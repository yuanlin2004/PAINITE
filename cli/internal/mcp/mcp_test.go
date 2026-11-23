package mcp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yuanlin2004/PAINITE/cli/internal/api"
)

// fakeServer implements enough of api/openapi.yaml for the tools.
type fakeServer struct {
	mu      sync.Mutex
	sources map[string][]byte
	jobs    map[string]map[string]any
	results map[string][]byte
	n       int
	hold    bool // keep jobs queued until released
}

func newFake(t *testing.T) (*fakeServer, *httptest.Server) {
	f := &fakeServer{sources: map[string][]byte{}, jobs: map[string]map[string]any{}, results: map[string][]byte{}}
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	return f, srv
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func notFound(w http.ResponseWriter) {
	writeJSON(w, 404, map[string]any{"error": map[string]any{"code": "not_found", "message": "no such thing"}})
}

func jpegBytes(w, h int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	var b bytes.Buffer
	_ = jpeg.Encode(&b, img, nil)
	return b.Bytes()
}

func (f *fakeServer) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer tok" && r.URL.Path != "/v1/health" {
		writeJSON(w, 401, map[string]any{"error": map[string]any{"code": "unauthorized", "message": "bad token"}})
		return
	}
	p := strings.TrimPrefix(r.URL.Path, "/v1")
	switch {
	case p == "/health":
		writeJSON(w, 200, map[string]any{"status": "ok", "version": "1.0.0",
			"gpu":   map[string]any{"available": true, "name": "Fake GPU"},
			"queue": map[string]any{"queued": 0, "running": 0, "capacity": 64}})
	case p == "/capabilities":
		writeJSON(w, 200, map[string]any{
			"operations": []any{map[string]any{"name": "denoise", "default_model": "denoise/x@1",
				"params": map[string]any{"strength": map[string]any{"min": 0, "max": 1, "default": 0.5}}}},
			"models":  []any{map[string]any{"id": "denoise/x@1", "operation": "denoise", "version": "1", "scale": 1}},
			"limits":  map[string]any{"max_upload_bytes": 1 << 29, "max_input_pixels": 1e8, "max_output_pixels": 2e8, "queue_capacity": 64},
			"formats": map[string]any{"input": []string{"jpeg", "tiff"}, "output": []string{"jpeg", "tiff"}}})
	case strings.HasPrefix(p, "/sources/"):
		sha := strings.TrimPrefix(p, "/sources/")
		switch r.Method {
		case http.MethodHead:
			if _, ok := f.sources[sha]; !ok {
				w.WriteHeader(404)
			}
		case http.MethodPut:
			data, _ := io.ReadAll(r.Body)
			sum := sha256.Sum256(data)
			if hex.EncodeToString(sum[:]) != sha {
				writeJSON(w, 400, map[string]any{"error": map[string]any{"code": "digest_mismatch", "message": "x"}})
				return
			}
			f.sources[sha] = data
			writeJSON(w, 200, map[string]any{"source_id": "sha256:" + sha, "bytes": len(data), "width": 64,
				"height": 64, "format": "jpeg"})
		}
	case p == "/jobs" && r.Method == http.MethodPost:
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		sha := strings.TrimPrefix(body["source_id"].(string), "sha256:")
		if _, ok := f.sources[sha]; !ok {
			notFound(w)
			return
		}
		f.n++
		id := fmt.Sprintf("job_%026d", f.n)
		op, model := "pipeline", "denoise/x@1+sharpen/y@1"
		if o, ok := body["operation"].(string); ok {
			op, model = o, o+"/x@1"
		}
		format := "jpeg"
		if out, ok := body["output"].(map[string]any); ok && out["format"] != nil {
			format = out["format"].(string)
		}
		job := map[string]any{"id": id, "state": "queued", "kind": body["kind"], "operation": op, "model": model,
			"source_id": body["source_id"], "progress": 0.0, "request": body, "format": format,
			"timings_ms": map[string]any{"total": 12.0}}
		if reg, ok := body["region"].(map[string]any); ok {
			job["region"] = reg
			job["faces"] = []any{map[string]any{"box": reg, "status": "enhanced", "size": 512}}
		}
		f.jobs[id] = job
		f.advance(id)
		writeJSON(w, 201, f.public(id))
	case strings.HasPrefix(p, "/jobs/"):
		rest := strings.TrimPrefix(p, "/jobs/")
		id, sub, _ := strings.Cut(rest, "/")
		job, ok := f.jobs[id]
		if !ok {
			notFound(w)
			return
		}
		switch {
		case sub == "" && r.Method == http.MethodGet:
			f.advance(id)
			writeJSON(w, 200, f.public(id))
		case sub == "cancel":
			if job["state"] == "queued" || job["state"] == "running" {
				job["state"] = "canceled"
			}
			writeJSON(w, 200, f.public(id))
		case sub == "result" && r.Method == http.MethodGet:
			data, ok := f.results[id]
			if !ok {
				writeJSON(w, 410, map[string]any{"error": map[string]any{"code": "result_unavailable", "message": "gone"}})
				return
			}
			sum := sha256.Sum256(data)
			w.Header().Set("X-Painite-SHA256", hex.EncodeToString(sum[:]))
			w.Header().Set("Content-Type", "image/jpeg")
			_, _ = w.Write(data)
		case sub == "result" && r.Method == http.MethodDelete:
			delete(f.results, id)
			job["deleted"] = true
			w.WriteHeader(204)
		}
	default:
		notFound(w)
	}
}

// advance moves a job one state forward per poll: queued, running, completed.
func (f *fakeServer) advance(id string) {
	job := f.jobs[id]
	if f.hold {
		return
	}
	switch job["state"] {
	case "queued":
		job["state"], job["progress"] = "running", 0.5
	case "running":
		job["state"], job["progress"] = "completed", 1.0
		req := job["request"].(map[string]any)
		if req["kind"] == "preview" {
			f.results[id] = jpegBytes(2000, 1000)
		} else {
			sha := strings.TrimPrefix(req["source_id"].(string), "sha256:")
			f.results[id] = append([]byte("enhanced:"), f.sources[sha]...)
		}
	}
}

func (f *fakeServer) public(id string) map[string]any {
	job := f.jobs[id]
	out := map[string]any{}
	for k, v := range job {
		if k != "request" && k != "format" && k != "deleted" {
			out[k] = v
		}
	}
	if data, ok := f.results[id]; ok || job["deleted"] == true {
		out["result"] = map[string]any{"bytes": len(data), "format": job["format"], "width": 2000, "height": 1000,
			"deleted": job["deleted"] == true}
	}
	return out
}

// session drives the MCP server over pipes.
type session struct {
	t     *testing.T
	in    *io.PipeWriter
	out   *bufio.Reader
	done  chan struct{}
	id    int
	notes []map[string]any
}

func start(t *testing.T, endpoint string) *session {
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	srv := NewServer("test", func() (*api.Client, error) { return api.New(endpoint, "tok", "") })
	s := &session{t: t, in: inW, out: bufio.NewReader(outR), done: make(chan struct{})}
	go func() {
		_ = srv.Serve(context.Background(), inR, outW, io.Discard)
		outW.Close()
		close(s.done)
	}()
	t.Cleanup(func() {
		inW.Close()
		<-s.done
	})
	s.request("initialize", map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{},
		"clientInfo": map[string]any{"name": "test", "version": "1"}})
	s.send(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	return s
}

func (s *session) send(v any) {
	data, _ := json.Marshal(v)
	if _, err := s.in.Write(append(data, '\n')); err != nil {
		s.t.Fatal(err)
	}
}

// request sends a request and returns its response, collecting notifications.
func (s *session) request(method string, params any) map[string]any {
	s.id++
	s.send(map[string]any{"jsonrpc": "2.0", "id": s.id, "method": method, "params": params})
	return s.wait(s.id)
}

func (s *session) wait(id int) map[string]any {
	for {
		line, err := s.out.ReadBytes('\n')
		if err != nil {
			s.t.Fatalf("reading response: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(line, &m); err != nil {
			s.t.Fatalf("bad JSON %q: %v", line, err)
		}
		if m["id"] == nil {
			s.notes = append(s.notes, m)
			continue
		}
		if int(m["id"].(float64)) == id {
			return m
		}
	}
}

// call runs a tool and returns its result.
func (s *session) call(name string, args map[string]any) map[string]any {
	resp := s.request("tools/call", map[string]any{"name": name, "arguments": args,
		"_meta": map[string]any{"progressToken": "p1"}})
	if resp["error"] != nil {
		s.t.Fatalf("%s: JSON-RPC error %v", name, resp["error"])
	}
	return resp["result"].(map[string]any)
}

func structured(r map[string]any) map[string]any { return r["structuredContent"].(map[string]any) }

func isError(r map[string]any) bool { b, _ := r["isError"].(bool); return b }

func errCode(r map[string]any) string {
	e, _ := structured(r)["error"].(map[string]any)
	if e == nil {
		return ""
	}
	return e["code"].(string)
}

func photo(t *testing.T, dir, name string) string {
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, jpegBytes(1200, 800), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestInitializeAndList(t *testing.T) {
	_, srv := newFake(t)
	s := start(t, srv.URL)
	// start() negotiated 2025-03-26; an unknown version gets the newest.
	resp := s.request("initialize", map[string]any{"protocolVersion": "1999-01-01"})
	if v := resp["result"].(map[string]any)["protocolVersion"]; v != ProtocolVersions[0] {
		t.Errorf("protocolVersion = %v", v)
	}
	tools := s.request("tools/list", nil)["result"].(map[string]any)["tools"].([]any)
	var names []string
	for _, tl := range tools {
		names = append(names, tl.(map[string]any)["name"].(string))
	}
	want := "server_status enhance_photos preview_enhancement submit_job get_job download_result cancel_job"
	if strings.Join(names, " ") != want {
		t.Errorf("tools = %v", names)
	}
	if resp := s.request("nope/nothing", nil); resp["error"].(map[string]any)["code"].(float64) != codeMethodNotFound {
		t.Errorf("unknown method: %v", resp)
	}
	if resp := s.request("tools/call", map[string]any{"name": "nope"}); resp["error"] == nil {
		t.Error("unknown tool should be a JSON-RPC error")
	}
	if resp := s.request("ping", nil); resp["result"] == nil {
		t.Error("ping")
	}
}

func TestStatus(t *testing.T) {
	_, srv := newFake(t)
	s := start(t, srv.URL)
	r := s.call("server_status", map[string]any{})
	if isError(r) {
		t.Fatalf("status failed: %v", r)
	}
	if !strings.Contains(r["content"].([]any)[0].(map[string]any)["text"].(string), "Fake GPU") {
		t.Errorf("summary: %v", r["content"])
	}
	if structured(r)["capabilities"] == nil {
		t.Error("no capabilities")
	}
}

func TestEnhancePhotos(t *testing.T) {
	_, srv := newFake(t)
	s := start(t, srv.URL)
	dir := t.TempDir()
	a, b := photo(t, dir, "a.jpg"), photo(t, dir, "b.jpg")
	out := filepath.Join(dir, "out") + "/"
	r := s.call("enhance_photos", map[string]any{"inputs": []string{a, b}, "output": out,
		"steps": []any{map[string]any{"operation": "denoise", "strength": 0.6}, map[string]any{"operation": "sharpen"}}})
	if isError(r) {
		t.Fatalf("enhance failed: %v", r)
	}
	if structured(r)["succeeded"].(float64) != 2 {
		t.Fatalf("results: %v", structured(r))
	}
	got, err := os.ReadFile(filepath.Join(dir, "out", "a.jpg"))
	if err != nil || !bytes.HasPrefix(got, []byte("enhanced:")) {
		t.Fatalf("output: %v", err)
	}
	if len(s.notes) == 0 {
		t.Error("expected progress notifications")
	}
	// A second run refuses to overwrite and submits nothing.
	r = s.call("enhance_photos", map[string]any{"inputs": []string{a}, "output": filepath.Join(dir, "out", "a.jpg"),
		"steps": []any{map[string]any{"operation": "denoise"}}})
	if !isError(r) || structured(r)["invalid"] == nil {
		t.Errorf("expected destination_exists: %v", r)
	}
}

func TestInvalidSteps(t *testing.T) {
	_, srv := newFake(t)
	s := start(t, srv.URL)
	dir := t.TempDir()
	a := photo(t, dir, "a.jpg")
	for _, steps := range [][]any{
		{},
		{map[string]any{"operation": "blur"}},
		{map[string]any{"operation": "upscale2x", "strength": 0.5}},
		{map[string]any{"operation": "upscale2x"}, map[string]any{"operation": "upscale2x"}},
		{map[string]any{"operation": "denoise", "strength": 2}},
		{map[string]any{"operation": "denoise"}, map[string]any{"operation": "denoise"},
			map[string]any{"operation": "denoise"}, map[string]any{"operation": "denoise"},
			map[string]any{"operation": "denoise"}},
	} {
		r := s.call("submit_job", map[string]any{"input": a, "steps": steps})
		if !isError(r) || errCode(r) != "invalid_params" {
			t.Errorf("%v: expected invalid_params, got %v", steps, r)
		}
	}
	r := s.call("submit_job", map[string]any{"input": a, "steps": []any{map[string]any{"operation": "denoise"}},
		"bogus": 1})
	if !isError(r) {
		t.Error("unknown argument should fail")
	}
	r = s.call("get_job", map[string]any{"job_id": "../sources/x"})
	if errCode(r) != "invalid_params" {
		t.Errorf("bad job id: %v", r)
	}
	r = s.call("submit_job", map[string]any{"input": filepath.Join(dir, "missing.jpg"),
		"steps": []any{map[string]any{"operation": "denoise"}}})
	if errCode(r) != "input_not_found" {
		t.Errorf("missing input: %v", r)
	}
}

func TestSubmitGetDownload(t *testing.T) {
	f, srv := newFake(t)
	f.hold = true
	s := start(t, srv.URL)
	dir := t.TempDir()
	a := photo(t, dir, "a.jpg")
	r := s.call("submit_job", map[string]any{"input": a, "steps": []any{map[string]any{"operation": "face"}}})
	if isError(r) {
		t.Fatalf("submit: %v", r)
	}
	id := structured(r)["job_id"].(string)

	r = s.call("download_result", map[string]any{"job_id": id, "output": filepath.Join(dir, "x.jpg")})
	if errCode(r) != "result_not_ready" {
		t.Errorf("download before done: %v", r)
	}
	f.mu.Lock()
	f.hold = false
	f.mu.Unlock()
	r = s.call("get_job", map[string]any{"job_id": id, "wait_seconds": 10})
	if structured(r)["state"] != "completed" {
		t.Fatalf("get_job: %v", r)
	}
	r = s.call("download_result", map[string]any{"job_id": id, "output": filepath.Join(dir, "x.tif")})
	if errCode(r) != "invalid_params" {
		t.Errorf("wrong extension: %v", r)
	}
	r = s.call("download_result", map[string]any{"job_id": id, "output": dir})
	if isError(r) {
		t.Fatalf("download: %v", r)
	}
	if out := structured(r)["output"].(string); out != filepath.Join(dir, id+".jpg") {
		t.Errorf("output = %s", out)
	}
	if structured(r)["deleted_from_server"] != true {
		t.Error("result should be deleted from the server")
	}
	r = s.call("download_result", map[string]any{"job_id": id, "output": filepath.Join(dir, "again.jpg")})
	if errCode(r) != "result_unavailable" {
		t.Errorf("second download: %v", r)
	}
}

func TestCancel(t *testing.T) {
	f, srv := newFake(t)
	f.hold = true
	s := start(t, srv.URL)
	a := photo(t, t.TempDir(), "a.jpg")
	r := s.call("submit_job", map[string]any{"input": a, "steps": []any{map[string]any{"operation": "denoise"}}})
	id := structured(r)["job_id"].(string)
	r = s.call("cancel_job", map[string]any{"job_id": id})
	if structured(r)["state"] != "canceled" {
		t.Errorf("cancel: %v", r)
	}
	r = s.call("download_result", map[string]any{"job_id": id, "output": "/tmp/x.jpg"})
	if errCode(r) != "canceled" {
		t.Errorf("download canceled job: %v", r)
	}
}

func TestPreview(t *testing.T) {
	_, srv := newFake(t)
	s := start(t, srv.URL)
	dir := t.TempDir()
	a := photo(t, dir, "a.jpg")
	saved := filepath.Join(dir, "prev.jpg")
	r := s.call("preview_enhancement", map[string]any{"input": a, "output": saved,
		"steps": []any{map[string]any{"operation": "face", "strength": 0.7}}})
	if isError(r) {
		t.Fatalf("preview: %v", r)
	}
	st := structured(r)
	reg := st["region"].(map[string]any)
	if reg["x"].(float64) != 88 || reg["y"].(float64) != 0 || reg["w"].(float64) != 1024 {
		t.Errorf("default region = %v", reg)
	}
	if st["faces"] == nil || st["saved_to"] != saved {
		t.Errorf("info = %v", st)
	}
	if text := r["content"].([]any)[0].(map[string]any)["text"].(string); !strings.Contains(text,
		"1 face found: 1 enhanced.") {
		t.Errorf("summary = %q", text)
	}
	var img map[string]any
	for _, c := range r["content"].([]any) {
		if c.(map[string]any)["type"] == "image" {
			img = c.(map[string]any)
		}
	}
	if img == nil {
		t.Fatal("no image content")
	}
	data, _ := base64.StdEncoding.DecodeString(img["data"].(string))
	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width != inlineImageSide || cfg.Height != 512 {
		t.Errorf("inline image %dx%d, %v", cfg.Width, cfg.Height, err)
	}
	if f, err := os.Open(saved); err != nil {
		t.Error(err)
	} else if cfg, _ := jpeg.DecodeConfig(f); cfg.Width != 2000 {
		t.Errorf("saved preview is %d wide, want the full 2000", cfg.Width)
	}
}

func TestCancelledRequest(t *testing.T) {
	f, srv := newFake(t)
	f.hold = true
	s := start(t, srv.URL)
	a := photo(t, t.TempDir(), "a.jpg")
	s.id++
	callID := s.id
	s.send(map[string]any{"jsonrpc": "2.0", "id": callID, "method": "tools/call", "params": map[string]any{
		"name": "preview_enhancement", "arguments": map[string]any{"input": a,
			"steps": []any{map[string]any{"operation": "denoise"}}}}})
	time.Sleep(300 * time.Millisecond)
	s.send(map[string]any{"jsonrpc": "2.0", "method": "notifications/cancelled",
		"params": map[string]any{"requestId": callID}})
	// The canceled call gets no response; the next request still works,
	// and the server job was canceled.
	if resp := s.request("ping", nil); resp["result"] == nil {
		t.Fatal("ping after cancel")
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		f.mu.Lock()
		state := f.jobs[fmt.Sprintf("job_%026d", 1)]["state"]
		f.mu.Unlock()
		if state == "canceled" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("server job state %v, want canceled", state)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
