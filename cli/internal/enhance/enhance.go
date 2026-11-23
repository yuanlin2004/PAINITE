// Package enhance runs a validated batch through the server: upload, submit,
// poll, download, and publish each result atomically.
package enhance

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/yuanlin2004/PAINITE/cli/internal/api"
	"github.com/yuanlin2004/PAINITE/cli/internal/plan"
)

// Options are the processing settings shared by every file.
type Options struct {
	// Steps run in order in one server job. One step is sent as a plain operation.
	Steps       []Step
	Model       string // only with one step
	JPEGQuality int
	Overwrite   bool
	Jobs        int // files in flight
	Progress    io.Writer
}

// Step is one operation and its strength (nil: server default).
type Step struct {
	Operation string
	Strength  *float64
}

// MaxSteps matches the server's limit on pipeline length.
const MaxSteps = 4

// KnownOps are the server's operations; StrengthOps are those that take a strength.
var (
	KnownOps = map[string]bool{"denoise": true, "sharpen": true, "unblur_motion": true, "unblur_focus": true,
		"upscale2x": true, "face": true}
	StrengthOps = map[string]bool{"denoise": true, "sharpen": true, "unblur_motion": true, "unblur_focus": true,
		"face": true}
)

// CheckSteps applies the server's pipeline rules: 1 to MaxSteps known
// operations, at most one upscale, and strengths in 0..1 only where they apply.
func CheckSteps(steps []Step) error {
	if len(steps) == 0 {
		return fmt.Errorf("at least one operation is required")
	}
	if len(steps) > MaxSteps {
		return fmt.Errorf("a pipeline has at most %d steps", MaxSteps)
	}
	upscales := 0
	for _, st := range steps {
		if !KnownOps[st.Operation] {
			return fmt.Errorf("unknown operation %q", st.Operation)
		}
		if st.Strength != nil {
			if !StrengthOps[st.Operation] {
				return fmt.Errorf("%s does not take a strength", st.Operation)
			}
			if *st.Strength < 0 || *st.Strength > 1 {
				return fmt.Errorf("strength for %s must be between 0 and 1", st.Operation)
			}
		}
		if st.Operation == "upscale2x" {
			upscales++
		}
	}
	if upscales > 1 {
		return fmt.Errorf("a pipeline can upscale at most once")
	}
	return nil
}

// Result is one line of --json output.
type Result struct {
	Input     string             `json:"input"`
	Output    string             `json:"output"`
	Status    string             `json:"status"` // succeeded, failed, invalid, not_run
	JobID     string             `json:"job_id,omitempty"`
	Model     string             `json:"model,omitempty"`
	TimingsMS map[string]float64 `json:"server_timings_ms,omitempty"`
	ClientMS  map[string]float64 `json:"client_timings_ms,omitempty"`
	Error     *api.Error         `json:"error,omitempty"`
}

type runner struct {
	c    *api.Client
	o    Options
	mu   sync.Mutex
	n    int
	done int
}

// Run processes the items (all valid) and returns results in input order.
func Run(ctx context.Context, c *api.Client, items []plan.Item, o Options) []Result {
	if o.Jobs < 1 {
		o.Jobs = 1
	}
	r := &runner{c: c, o: o, n: len(items)}
	results := make([]Result, len(items))
	sem := make(chan struct{}, o.Jobs)
	var wg sync.WaitGroup
	for i := range items {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				results[i] = Result{Input: items[i].Input, Output: items[i].Output, Status: "failed",
					Error: &api.Error{Code: "canceled", Message: "interrupted"}}
				return
			}
			defer func() { <-sem }()
			results[i] = r.one(ctx, i, items[i])
		}(i)
	}
	wg.Wait()
	return results
}

func (r *runner) logf(i int, input, format string, args ...any) {
	if r.o.Progress == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	fmt.Fprintf(r.o.Progress, "[%d/%d] %s: %s\n", i+1, r.n, filepath.Base(input), fmt.Sprintf(format, args...))
}

func (r *runner) one(ctx context.Context, i int, it plan.Item) (res Result) {
	res = Result{Input: it.Input, Output: it.Output, ClientMS: map[string]float64{}}
	fail := func(err error) Result {
		res.Status = "failed"
		res.Error = api.AsError(err)
		r.logf(i, it.Input, "failed: %s", res.Error.Error())
		return res
	}

	t0 := time.Now()
	job, err := Submit(ctx, r.c, it.Input, Request{Steps: r.o.Steps, Model: r.o.Model, Format: it.Format,
		JPEGQuality: r.o.JPEGQuality, Kind: "full"}, func(msg string) { r.logf(i, it.Input, "%s", msg) })
	if err != nil {
		return fail(err)
	}
	res.ClientMS["upload"] = ms(time.Since(t0))
	res.JobID = job.ID
	res.Model = job.Model

	t1 := time.Now()
	job, err = r.wait(ctx, i, it, job)
	if err != nil {
		if ctx.Err() != nil {
			cctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			_ = r.c.CancelJob(cctx, res.JobID)
			cancel()
		}
		return fail(err)
	}
	res.ClientMS["wait"] = ms(time.Since(t1))
	res.TimingsMS = job.TimingsMS
	if job.State != "completed" {
		if job.Error == nil {
			job.Error = &api.Error{Code: job.State, Message: "job " + job.State}
		}
		return fail(job.Error)
	}

	t2 := time.Now()
	if err := Download(ctx, r.c, job.ID, it.Output, r.o.Overwrite); err != nil {
		return fail(err)
	}
	res.ClientMS["download"] = ms(time.Since(t2))
	res.ClientMS["total"] = ms(time.Since(t0))
	dctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_ = r.c.DeleteResult(dctx, job.ID)
	cancel()

	res.Status = "succeeded"
	r.logf(i, it.Input, "done -> %s (server %.1fs: queue %.1fs, inference %.1fs; transfer %.1fs)",
		it.Output, job.TimingsMS["total"]/1000, job.TimingsMS["queue"]/1000, job.TimingsMS["inference"]/1000,
		(res.ClientMS["upload"]+res.ClientMS["download"])/1000)
	return res
}

func (r *runner) wait(ctx context.Context, i int, it plan.Item, job *api.Job) (*api.Job, error) {
	lastState, lastPct := "", -1
	return Wait(ctx, r.c, job, func(job *api.Job) {
		if job.State != lastState || (job.State == "running" && int(job.Progress*10) != lastPct) {
			if job.State == "running" {
				r.logf(i, it.Input, "running %d%%", int(job.Progress*100))
			} else if !job.Terminal() {
				r.logf(i, it.Input, "%s", job.State)
			}
			lastState, lastPct = job.State, int(job.Progress*10)
		}
	})
}

// Wait polls a job until it finishes, calling update (if not nil) with each
// state seen, the first one included. Polling backs off from 200 ms to 2 s.
func Wait(ctx context.Context, c *api.Client, job *api.Job, update func(*api.Job)) (*api.Job, error) {
	delay := 200 * time.Millisecond
	for {
		if update != nil {
			update(job)
		}
		if job.Terminal() {
			return job, nil
		}
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		delay = min(delay*3/2, 2*time.Second)
		next, err := c.GetJob(ctx, job.ID)
		if err != nil {
			return nil, err
		}
		job = next
	}
}

// Download writes a job's result to a temp file beside dest, verifies it, and
// renames it into place, so a failure never leaves a partial or damaged file.
// Without overwrite, an existing dest is left alone and reported.
func Download(ctx context.Context, c *api.Client, jobID, dest string, overwrite bool) error {
	dir := filepath.Dir(dest)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(dest)+".painite-*.part")
	if err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	defer tmp.Close()

	want, err := c.DownloadResult(ctx, jobID, tmp)
	if err != nil {
		return err
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	h := sha256.New()
	if _, err := io.Copy(h, tmp); err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	if want != "" && hex.EncodeToString(h.Sum(nil)) != want {
		return &api.Error{Code: "download_corrupt", Message: "downloaded result does not match the server's checksum"}
	}
	// CreateTemp makes 0600 files; results are ordinary images.
	if err := tmp.Chmod(0o644); err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	if err := tmp.Sync(); err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	if err := tmp.Close(); err != nil {
		return &api.Error{Code: "write_failed", Message: err.Error()}
	}
	if overwrite {
		if err := os.Rename(tmpName, dest); err != nil {
			return &api.Error{Code: "write_failed", Message: err.Error()}
		}
		return nil
	}
	// No overwrite: a hard link fails if the destination appeared meanwhile.
	if err := os.Link(tmpName, dest); err != nil {
		if _, statErr := os.Lstat(dest); statErr == nil {
			return &api.Error{Code: "destination_exists", Message: "output appeared while processing; not overwritten"}
		}
		// File systems without hard links: fall back to rename.
		if err := os.Rename(tmpName, dest); err != nil {
			return &api.Error{Code: "write_failed", Message: err.Error()}
		}
	}
	return nil
}

// Request describes one job to submit for one input file.
type Request struct {
	Steps       []Step
	Model       string // only with one step
	Kind        string // full or preview
	Format      string // output format: jpeg or tiff
	JPEGQuality int
	Region      *api.Region // preview only
}

// Submit uploads the input unless the server already has it, then creates
// the job. log, if not nil, receives short progress messages.
func Submit(ctx context.Context, c *api.Client, input string, rq Request, log func(string)) (*api.Job, error) {
	if log == nil {
		log = func(string) {}
	}
	sha, size, err := hashFile(input)
	if err != nil {
		return nil, &api.Error{Code: "input_unreadable", Message: err.Error()}
	}
	have, err := c.HasSource(ctx, sha)
	if err != nil {
		return nil, err
	}
	if have {
		log("already on server, skipping upload")
	} else {
		log(fmt.Sprintf("uploading %.1f MB", float64(size)/1e6))
		if err := c.UploadSource(ctx, input, sha, size); err != nil {
			return nil, err
		}
	}
	jr := api.JobRequest{SourceID: "sha256:" + sha, Kind: rq.Kind, Region: rq.Region}
	if rq.Format != "" || rq.JPEGQuality != 0 {
		jr.Output = map[string]any{}
		if rq.Format != "" {
			jr.Output["format"] = rq.Format
		}
		if rq.JPEGQuality != 0 {
			jr.Output["jpeg_quality"] = rq.JPEGQuality
		}
	}
	if len(rq.Steps) == 1 {
		st := rq.Steps[0]
		jr.Operation, jr.Model = st.Operation, rq.Model
		if st.Strength != nil {
			jr.Params = map[string]any{"strength": *st.Strength}
		}
	} else {
		for _, st := range rq.Steps {
			step := api.Step{Operation: st.Operation}
			if st.Strength != nil {
				step.Params = map[string]any{"strength": *st.Strength}
			}
			jr.Steps = append(jr.Steps, step)
		}
	}
	return c.CreateJob(ctx, jr, newKey())
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func newKey() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "cli-" + hex.EncodeToString(b)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
