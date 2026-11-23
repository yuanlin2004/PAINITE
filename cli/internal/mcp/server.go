// Package mcp is a Model Context Protocol server over stdio, so AI agents
// (Claude Code, Claude Desktop, and other MCP clients) can use a PAINITE
// server through `painite mcp`.
//
// It speaks JSON-RPC 2.0, one message per line on stdin and stdout, and
// implements the parts of MCP a tools-only server needs: initialize, ping,
// tools/list, tools/call, cancellation, and progress notifications.
// Standard library only, like the rest of the CLI.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// ProtocolVersions are the MCP revisions this server speaks, newest first.
var ProtocolVersions = []string{"2025-06-18", "2025-03-26", "2024-11-05"}

// JSON-RPC error codes.
const (
	codeParseError     = -32700
	codeInvalidRequest = -32600
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
)

// Tool is one tool the server offers.
type Tool struct {
	Name        string                                        `json:"name"`
	Title       string                                        `json:"title,omitempty"`
	Description string                                        `json:"description"`
	InputSchema map[string]any                                `json:"inputSchema"`
	Annotations map[string]any                                `json:"annotations,omitempty"`
	Handler     func(ctx context.Context, call *Call) *Result `json:"-"`
}

// Call is one tools/call request.
type Call struct {
	Args json.RawMessage
	// Progress reports a step to the client, if it asked for progress.
	Progress func(message string)
}

// Bind decodes the arguments into v, rejecting unknown fields.
func (c *Call) Bind(v any) error {
	if len(c.Args) == 0 || string(c.Args) == "null" {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(c.Args))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}

// Result is a tools/call result.
type Result struct {
	Content           []map[string]any `json:"content"`
	StructuredContent any              `json:"structuredContent,omitempty"`
	IsError           bool             `json:"isError,omitempty"`
}

// Server runs tools for one client.
type Server struct {
	Name, Version, Instructions string
	Tools                       []Tool

	out   io.Writer
	outMu sync.Mutex
	log   io.Writer

	mu       sync.Mutex
	inflight map[string]context.CancelFunc
	wg       sync.WaitGroup
}

type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Serve reads requests from in and writes responses to out until in closes
// or ctx ends. Tool calls run concurrently; when input ends, calls still
// running are canceled and Serve waits for them before returning.
func (s *Server) Serve(ctx context.Context, in io.Reader, out, log io.Writer) error {
	s.out, s.log = out, log
	s.inflight = map[string]context.CancelFunc{}
	ctx, cancel := context.WithCancel(ctx)
	defer func() {
		cancel()
		s.wg.Wait()
	}()

	r := bufio.NewReaderSize(in, 1<<20)
	for {
		line, err := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			s.handle(ctx, line)
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return nil
		}
	}
}

func (s *Server) handle(ctx context.Context, line []byte) {
	var m message
	if err := json.Unmarshal(line, &m); err != nil {
		s.reply(json.RawMessage("null"), nil, &rpcError{codeParseError, "parse error: " + err.Error()})
		return
	}
	isRequest := len(m.ID) > 0 && string(m.ID) != "null"
	if m.Method == "" {
		// A response to a request we never send, or junk. Nothing to do.
		if !isRequest {
			s.reply(json.RawMessage("null"), nil, &rpcError{codeInvalidRequest, "invalid request"})
		}
		return
	}
	if !isRequest {
		s.notification(m)
		return
	}
	switch m.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(m.Params, &p)
		version := ProtocolVersions[0]
		for _, v := range ProtocolVersions {
			if v == p.ProtocolVersion {
				version = v
			}
		}
		s.reply(m.ID, map[string]any{
			"protocolVersion": version,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
			"serverInfo":      map[string]any{"name": s.Name, "title": "PAINITE", "version": s.Version},
			"instructions":    s.Instructions,
		}, nil)
	case "ping":
		s.reply(m.ID, map[string]any{}, nil)
	case "tools/list":
		s.reply(m.ID, map[string]any{"tools": s.Tools}, nil)
	case "tools/call":
		s.call(ctx, m)
	case "resources/list":
		s.reply(m.ID, map[string]any{"resources": []any{}}, nil)
	case "prompts/list":
		s.reply(m.ID, map[string]any{"prompts": []any{}}, nil)
	default:
		s.reply(m.ID, nil, &rpcError{codeMethodNotFound, "method not found: " + m.Method})
	}
}

func (s *Server) notification(m message) {
	if m.Method != "notifications/cancelled" {
		return // initialized, roots changes, and so on: nothing to do.
	}
	var p struct {
		RequestID json.RawMessage `json:"requestId"`
	}
	if json.Unmarshal(m.Params, &p) != nil {
		return
	}
	s.mu.Lock()
	cancel := s.inflight[string(p.RequestID)]
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (s *Server) call(ctx context.Context, m message) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Meta      struct {
			ProgressToken json.RawMessage `json:"progressToken"`
		} `json:"_meta"`
	}
	if err := json.Unmarshal(m.Params, &p); err != nil {
		s.reply(m.ID, nil, &rpcError{codeInvalidParams, "invalid params: " + err.Error()})
		return
	}
	var tool *Tool
	for i := range s.Tools {
		if s.Tools[i].Name == p.Name {
			tool = &s.Tools[i]
		}
	}
	if tool == nil {
		s.reply(m.ID, nil, &rpcError{codeInvalidParams, "unknown tool: " + p.Name})
		return
	}

	cctx, cancel := context.WithCancel(ctx)
	key := string(m.ID)
	s.mu.Lock()
	s.inflight[key] = cancel
	s.mu.Unlock()

	c := &Call{Args: p.Arguments, Progress: func(string) {}}
	if tok := p.Meta.ProgressToken; len(tok) > 0 && string(tok) != "null" {
		var mu sync.Mutex
		n := 0
		c.Progress = func(msg string) {
			mu.Lock()
			n++
			step := n
			mu.Unlock()
			s.send(map[string]any{"jsonrpc": "2.0", "method": "notifications/progress",
				"params": map[string]any{"progressToken": tok, "progress": step, "message": msg}})
		}
	}

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer func() {
			s.mu.Lock()
			delete(s.inflight, key)
			s.mu.Unlock()
			cancel()
		}()
		res := s.run(cctx, tool, c)
		if cctx.Err() != nil && ctx.Err() == nil {
			// The client canceled this request: MCP says not to answer it.
			return
		}
		s.reply(m.ID, res, nil)
	}()
}

// run calls the handler, turning a panic into a tool error.
func (s *Server) run(ctx context.Context, tool *Tool, c *Call) (res *Result) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(s.log, "painite mcp: %s panicked: %v\n", tool.Name, r)
			res = ErrorResult("internal", fmt.Sprintf("%s failed unexpectedly: %v", tool.Name, r))
		}
	}()
	return tool.Handler(ctx, c)
}

func (s *Server) reply(id json.RawMessage, result any, e *rpcError) {
	msg := map[string]any{"jsonrpc": "2.0", "id": id}
	if e != nil {
		msg["error"] = e
	} else {
		msg["result"] = result
	}
	s.send(msg)
}

func (s *Server) send(msg any) {
	data, err := json.Marshal(msg)
	if err != nil {
		fmt.Fprintf(s.log, "painite mcp: encoding a message: %v\n", err)
		return
	}
	s.outMu.Lock()
	defer s.outMu.Unlock()
	_, _ = s.out.Write(append(data, '\n'))
}

// TextResult is a successful result with v as JSON text and structured content.
func TextResult(summary string, v any) *Result {
	data, _ := json.MarshalIndent(v, "", "  ")
	content := []map[string]any{}
	if summary != "" {
		content = append(content, map[string]any{"type": "text", "text": summary})
	}
	content = append(content, map[string]any{"type": "text", "text": string(data)})
	return &Result{Content: content, StructuredContent: v}
}

// ErrorResult is a failed tool call the agent can read and act on.
func ErrorResult(code, message string) *Result {
	v := map[string]any{"error": map[string]any{"code": code, "message": message}}
	return &Result{
		Content:           []map[string]any{{"type": "text", "text": code + ": " + message}},
		StructuredContent: v,
		IsError:           true,
	}
}
