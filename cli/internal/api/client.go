// Package api is a client for the PAINITE service contract (api/openapi.yaml).
package api

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// Error is a server error with a stable code, or a client-side failure.
type Error struct {
	Status  int    `json:"-"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// Retryable reports whether trying the same request again may succeed.
func (e *Error) Retryable() bool {
	switch e.Code {
	case "connection_failed", "queue_full", "internal":
		return true
	}
	return e.Status == http.StatusBadGateway || e.Status == http.StatusServiceUnavailable ||
		e.Status == http.StatusGatewayTimeout
}

// AsError converts any error to *Error.
func AsError(err error) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	if errors.Is(err, context.Canceled) {
		return &Error{Code: "canceled", Message: "interrupted"}
	}
	return &Error{Code: "connection_failed", Message: err.Error()}
}

type Health struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	GPU     struct {
		Available bool    `json:"available"`
		Name      *string `json:"name"`
	} `json:"gpu"`
	Queue struct {
		Queued   int `json:"queued"`
		Running  int `json:"running"`
		Capacity int `json:"capacity"`
	} `json:"queue"`
}

type ParamRange struct {
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Default float64 `json:"default"`
}

type Capabilities struct {
	Operations []struct {
		Name         string                `json:"name"`
		DefaultModel string                `json:"default_model"`
		Params       map[string]ParamRange `json:"params"`
	} `json:"operations"`
	Models []struct {
		ID          string `json:"id"`
		Operation   string `json:"operation"`
		Version     string `json:"version"`
		Scale       int    `json:"scale"`
		Placeholder bool   `json:"placeholder"`
	} `json:"models"`
	Limits struct {
		MaxUploadBytes  int64 `json:"max_upload_bytes"`
		MaxInputPixels  int64 `json:"max_input_pixels"`
		MaxOutputPixels int64 `json:"max_output_pixels"`
		QueueCapacity   int   `json:"queue_capacity"`
	} `json:"limits"`
	Formats struct {
		Input  []string `json:"input"`
		Output []string `json:"output"`
	} `json:"formats"`
}

type Source struct {
	SourceID string `json:"source_id"`
	Bytes    int64  `json:"bytes"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Format   string `json:"format"`
}

type JobRequest struct {
	SourceID  string         `json:"source_id"`
	Kind      string         `json:"kind"`
	Operation string         `json:"operation"`
	Params    map[string]any `json:"params,omitempty"`
	Model     string         `json:"model,omitempty"`
	Output    map[string]any `json:"output,omitempty"`
}

type Job struct {
	ID        string  `json:"id"`
	State     string  `json:"state"`
	Kind      string  `json:"kind"`
	Operation string  `json:"operation"`
	Model     string  `json:"model"`
	Progress  float64 `json:"progress"`
	Result    *struct {
		Bytes   int64  `json:"bytes"`
		SHA256  string `json:"sha256"`
		Format  string `json:"format"`
		Width   int    `json:"width"`
		Height  int    `json:"height"`
		Deleted bool   `json:"deleted"`
	} `json:"result"`
	TimingsMS map[string]float64 `json:"timings_ms"`
	Error     *Error             `json:"error"`
}

// Terminal reports whether the job has finished.
func (j *Job) Terminal() bool {
	return j.State == "completed" || j.State == "failed" || j.State == "canceled"
}

// Client talks to one server with one token.
type Client struct {
	base  string
	token string
	http  *http.Client
	// Backoff schedule for retryable failures.
	Retries int
}

// New creates a client. endpoint is the server root, e.g. https://spark1.local:8443.
func New(endpoint, token, caFile string) (*Client, error) {
	if endpoint == "" {
		return nil, &Error{Code: "not_configured", Message: "no endpoint configured; run `painite config set endpoint <url>`"}
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.DialContext = (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	tr.ResponseHeaderTimeout = 60 * time.Second
	tr.MaxIdleConnsPerHost = 8
	if caFile != "" {
		pem, err := os.ReadFile(caFile)
		if err != nil {
			return nil, &Error{Code: "not_configured", Message: fmt.Sprintf("reading CA file: %v", err)}
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, &Error{Code: "not_configured", Message: "CA file has no PEM certificates"}
		}
		tr.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
	}
	return &Client{
		base:    strings.TrimRight(endpoint, "/") + "/v1",
		token:   token,
		http:    &http.Client{Transport: tr},
		Retries: 6,
	}, nil
}

func (c *Client) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, body)
	if err != nil {
		return nil, err
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	req.Header.Set("User-Agent", "painite-cli")
	return req, nil
}

// do sends one request and decodes a JSON response or error. No retries.
func (c *Client) do(req *http.Request, out any) (*http.Response, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		if req.Context().Err() != nil {
			return nil, req.Context().Err()
		}
		return nil, &Error{Code: "connection_failed", Message: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return resp, decodeError(resp)
	}
	if out != nil && resp.StatusCode != http.StatusNoContent {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return resp, &Error{Status: resp.StatusCode, Code: "connection_failed", Message: "bad response: " + err.Error()}
		}
	}
	return resp, nil
}

func decodeError(resp *http.Response) error {
	var body struct {
		Error *Error `json:"error"`
	}
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if json.Unmarshal(data, &body) == nil && body.Error != nil && body.Error.Code != "" {
		body.Error.Status = resp.StatusCode
		return body.Error
	}
	code := "internal"
	switch resp.StatusCode {
	case 401:
		code = "unauthorized"
	case 404:
		code = "not_found"
	}
	return &Error{Status: resp.StatusCode, Code: code, Message: fmt.Sprintf("HTTP %d", resp.StatusCode)}
}

// Retry runs fn until it succeeds, fails permanently, or retries run out.
// Backoff is exponential with jitter, capped at 10 s.
func (c *Client) Retry(ctx context.Context, fn func() error) error {
	delay := 500 * time.Millisecond
	for attempt := 0; ; attempt++ {
		err := fn()
		if err == nil {
			return nil
		}
		e := AsError(err)
		if !e.Retryable() || attempt >= c.Retries || ctx.Err() != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		sleep := delay/2 + time.Duration(rand.Int64N(int64(delay)))
		select {
		case <-time.After(sleep):
		case <-ctx.Done():
			return ctx.Err()
		}
		delay = min(delay*2, 10*time.Second)
	}
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	return c.Retry(ctx, func() error {
		req, err := c.newRequest(ctx, http.MethodGet, path, nil)
		if err != nil {
			return err
		}
		_, err = c.do(req, out)
		return err
	})
}

func (c *Client) Health(ctx context.Context) (*Health, error) {
	var h Health
	return &h, c.getJSON(ctx, "/health", &h)
}

func (c *Client) Capabilities(ctx context.Context) (*Capabilities, error) {
	var caps Capabilities
	return &caps, c.getJSON(ctx, "/capabilities", &caps)
}

// HasSource reports whether the server already has this source for our token.
func (c *Client) HasSource(ctx context.Context, sha string) (bool, error) {
	var found bool
	err := c.Retry(ctx, func() error {
		req, err := c.newRequest(ctx, http.MethodHead, "/sources/"+sha, nil)
		if err != nil {
			return err
		}
		resp, err := c.do(req, nil)
		if err != nil {
			if e := AsError(err); e.Status == http.StatusNotFound {
				found = false
				return nil
			}
			return err
		}
		found = resp.StatusCode == http.StatusOK
		return nil
	})
	return found, err
}

// UploadSource uploads a file unless the server already has it.
func (c *Client) UploadSource(ctx context.Context, path, sha string, size int64) error {
	return c.Retry(ctx, func() error {
		if ok, err := c.HasSource(ctx, sha); err != nil || ok {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return &Error{Code: "input_unreadable", Message: err.Error()}
		}
		defer f.Close()
		req, err := c.newRequest(ctx, http.MethodPut, "/sources/"+sha, f)
		if err != nil {
			return err
		}
		req.ContentLength = size
		req.Header.Set("Content-Type", "application/octet-stream")
		_, err = c.do(req, nil)
		return err
	})
}

// CreateJob submits a job. The idempotency key makes retries safe.
func (c *Client) CreateJob(ctx context.Context, jr JobRequest, key string) (*Job, error) {
	body, err := json.Marshal(jr)
	if err != nil {
		return nil, err
	}
	var job Job
	err = c.Retry(ctx, func() error {
		req, err := c.newRequest(ctx, http.MethodPost, "/jobs", bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Idempotency-Key", key)
		_, err = c.do(req, &job)
		return err
	})
	return &job, err
}

func (c *Client) GetJob(ctx context.Context, id string) (*Job, error) {
	var job Job
	return &job, c.getJSON(ctx, "/jobs/"+id, &job)
}

// CancelJob asks the server to cancel. Best-effort; one attempt.
func (c *Client) CancelJob(ctx context.Context, id string) error {
	req, err := c.newRequest(ctx, http.MethodPost, "/jobs/"+id+"/cancel", nil)
	if err != nil {
		return err
	}
	_, err = c.do(req, nil)
	return err
}

// DeleteResult deletes the server's copy of a result. Best-effort; one attempt.
func (c *Client) DeleteResult(ctx context.Context, id string) error {
	req, err := c.newRequest(ctx, http.MethodDelete, "/jobs/"+id+"/result", nil)
	if err != nil {
		return err
	}
	_, err = c.do(req, nil)
	return err
}

// DownloadResult appends the result to f, resuming from f's current size
// after interruptions, and returns the server's SHA-256 for the whole file.
func (c *Client) DownloadResult(ctx context.Context, id string, f *os.File) (string, error) {
	var digest string
	err := c.Retry(ctx, func() error {
		offset, err := f.Seek(0, io.SeekEnd)
		if err != nil {
			return &Error{Code: "write_failed", Message: err.Error()}
		}
		req, err := c.newRequest(ctx, http.MethodGet, "/jobs/"+id+"/result", nil)
		if err != nil {
			return err
		}
		if offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		resp, err := c.http.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return &Error{Code: "connection_failed", Message: err.Error()}
		}
		defer resp.Body.Close()
		switch {
		case resp.StatusCode == http.StatusOK && offset > 0:
			// Server ignored the range; start over.
			if err := f.Truncate(0); err != nil {
				return &Error{Code: "write_failed", Message: err.Error()}
			}
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				return &Error{Code: "write_failed", Message: err.Error()}
			}
		case resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusPartialContent:
		case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
			// Already complete.
			digest = resp.Header.Get("X-Painite-SHA256")
			return nil
		default:
			return decodeError(resp)
		}
		digest = resp.Header.Get("X-Painite-SHA256")
		if _, err := io.Copy(fileWriter{f}, resp.Body); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			var we writeError
			if errors.As(err, &we) {
				return &Error{Code: "write_failed", Message: we.err.Error()}
			}
			// Network drop mid-body: retry resumes from what we have.
			return &Error{Code: "connection_failed", Message: err.Error()}
		}
		return nil
	})
	return digest, err
}

type writeError struct{ err error }

func (w writeError) Error() string { return w.err.Error() }

// fileWriter tags write errors so they are not mistaken for network errors.
type fileWriter struct{ f *os.File }

func (w fileWriter) Write(p []byte) (int, error) {
	n, err := w.f.Write(p)
	if err != nil {
		return n, writeError{err}
	}
	return n, nil
}
