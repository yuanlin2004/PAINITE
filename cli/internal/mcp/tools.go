package mcp

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/yuanlin2004/PAINITE/cli/internal/api"
	"github.com/yuanlin2004/PAINITE/cli/internal/enhance"
	"github.com/yuanlin2004/PAINITE/cli/internal/plan"
)

// ClientFunc makes an API client from the current settings. It runs on every
// tool call, so settings changed while the server runs take effect.
type ClientFunc func() (*api.Client, error)

// Limits for agent-facing tools.
const (
	maxInputs       = 50
	maxPreviewSide  = 2048 // requested region side, before the server snaps it to tiles
	defaultPreview  = 1024
	inlineImageSide = 1024 // long edge of the image returned inline, to keep agent context small
	maxWaitSeconds  = 120
)

const instructions = `PAINITE enhances photos (JPEG or TIFF files on this computer) on a PAINITE inference server: ` +
	`denoise, sharpen, unblur_motion (camera shake), unblur_focus (missed focus), face (restore faces) and upscale2x. ` +
	`Use absolute file paths. Call server_status first to see the operations, their strength ranges and the models. ` +
	`enhance_photos is the simple path: it uploads, runs and saves results, and returns when they are written. ` +
	`preview_enhancement renders one region and returns it as an image so you can judge an effect before a full run. ` +
	`For long batches use submit_job, then get_job (with wait_seconds) and download_result. ` +
	`Originals are never changed, and existing files are kept unless overwrite is true.`

// NewServer returns the PAINITE MCP server.
func NewServer(version string, client ClientFunc) *Server {
	t := &tools{client: client}
	return &Server{
		Name:         "painite",
		Version:      version,
		Instructions: instructions,
		Tools: []Tool{
			{
				Name:  "server_status",
				Title: "PAINITE server status",
				Description: "Check the PAINITE server: health, GPU, queue, the operations it offers with their " +
					"parameter ranges and default models, installed models, and limits.",
				InputSchema: object(nil, nil),
				Annotations: map[string]any{"readOnlyHint": true, "openWorldHint": false},
				Handler:     t.status,
			},
			{
				Name:  "enhance_photos",
				Title: "Enhance photos",
				Description: "Enhance one or more JPEG/TIFF photos and save the results. Runs one operation or a " +
					"pipeline of up to 4 steps in order in one server job (at most one upscale2x). Returns when " +
					"every result is saved, with per-photo status, output path, job ID and timings. Everything is " +
					"validated before anything is uploaded. A 24 MP photo takes about 30 seconds per step on a " +
					"DGX Spark; for many photos or a slow server prefer submit_job.",
				InputSchema: object(map[string]any{
					"inputs": map[string]any{"type": "array", "minItems": 1, "maxItems": maxInputs,
						"items":       map[string]any{"type": "string"},
						"description": "Absolute paths of the photos (.jpg, .jpeg, .tif, .tiff)."},
					"output": map[string]any{"type": "string", "description": "Output file for one input, or a " +
						"directory (created if needed; end it with / to be explicit). In a directory each result " +
						"keeps its input's file name."},
					"steps":        stepsSchema,
					"model":        modelSchema,
					"format":       formatSchema,
					"jpeg_quality": qualitySchema,
					"overwrite": map[string]any{"type": "boolean", "default": false,
						"description": "Replace existing output files."},
				}, []string{"inputs", "output", "steps"}),
				Annotations: map[string]any{"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
				Handler:     t.enhancePhotos,
			},
			{
				Name:  "preview_enhancement",
				Title: "Preview an enhancement",
				Description: "Render the enhancement for one region of a photo and return it as an image, so you " +
					"can see the effect before a full run. The server grows the region to its tile grid; the " +
					"preview is pixel-identical to the same region of a full result. The returned image is " +
					"scaled down to fit 1024 px; pass output to also save it at full size. For face, faces " +
					"lists every face in the whole photo in full-photo coordinates, not only those in the " +
					"region, and the summary counts them. A new preview of a photo replaces any unfinished earlier preview of the same photo.",
				InputSchema: object(map[string]any{
					"input": map[string]any{"type": "string", "description": "Absolute path of the photo."},
					"steps": stepsSchema,
					"model": modelSchema,
					"region": map[string]any{
						"type": "object",
						"description": fmt.Sprintf("Area in displayed (rotation-applied) pixels. Default: a "+
							"%d x %d square at the center. Each side at most %d.", defaultPreview, defaultPreview,
							maxPreviewSide),
						"properties": map[string]any{
							"x": map[string]any{"type": "integer", "minimum": 0},
							"y": map[string]any{"type": "integer", "minimum": 0},
							"w": map[string]any{"type": "integer", "minimum": 1, "maximum": maxPreviewSide},
							"h": map[string]any{"type": "integer", "minimum": 1, "maximum": maxPreviewSide},
						},
						"required":             []string{"x", "y", "w", "h"},
						"additionalProperties": false,
					},
					"output": map[string]any{"type": "string",
						"description": "Optional .jpg path to save the full-size preview."},
					"overwrite": map[string]any{"type": "boolean", "default": false},
				}, []string{"input", "steps"}),
				Annotations: map[string]any{"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
				Handler:     t.preview,
			},
			{
				Name:  "submit_job",
				Title: "Submit an enhancement job",
				Description: "Upload a photo (skipped if the server already has it) and start a full enhancement " +
					"job without waiting. Returns the job ID. Follow with get_job and download_result.",
				InputSchema: object(map[string]any{
					"input":        map[string]any{"type": "string", "description": "Absolute path of the photo."},
					"steps":        stepsSchema,
					"model":        modelSchema,
					"format":       formatSchema,
					"jpeg_quality": qualitySchema,
				}, []string{"input", "steps"}),
				Annotations: map[string]any{"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
				Handler:     t.submit,
			},
			{
				Name:  "get_job",
				Title: "Get job status",
				Description: "Get a job's state (queued, running, completed, failed, canceled), progress, models, " +
					"timings, faces found, result size and error. With wait_seconds, waits up to that long for " +
					"the job to finish before answering.",
				InputSchema: object(map[string]any{
					"job_id": jobIDSchema,
					"wait_seconds": map[string]any{"type": "integer", "minimum": 0, "maximum": maxWaitSeconds,
						"default": 0, "description": "Wait up to this many seconds for the job to finish."},
				}, []string{"job_id"}),
				Annotations: map[string]any{"readOnlyHint": true, "openWorldHint": false},
				Handler:     t.getJob,
			},
			{
				Name:  "download_result",
				Title: "Download a job's result",
				Description: "Save a completed job's result to a file, verified against the server's checksum. " +
					"By default the server's copy is then deleted.",
				InputSchema: object(map[string]any{
					"job_id": jobIDSchema,
					"output": map[string]any{"type": "string", "description": "Output file path, or an existing " +
						"directory (the file is named after the job). The extension should match the result " +
						"format (.jpg or .tif)."},
					"overwrite": map[string]any{"type": "boolean", "default": false},
					"delete_from_server": map[string]any{"type": "boolean", "default": true,
						"description": "Delete the server's copy after a successful download."},
				}, []string{"job_id", "output"}),
				Annotations: map[string]any{"readOnlyHint": false, "destructiveHint": false, "openWorldHint": false},
				Handler:     t.download,
			},
			{
				Name:        "cancel_job",
				Title:       "Cancel a job",
				Description: "Cancel a queued or running job. Canceling a finished job does nothing.",
				InputSchema: object(map[string]any{"job_id": jobIDSchema}, []string{"job_id"}),
				Annotations: map[string]any{"readOnlyHint": false, "destructiveHint": true, "idempotentHint": true,
					"openWorldHint": false},
				Handler: t.cancel,
			},
		},
	}
}

var (
	ops         = []string{"denoise", "sharpen", "unblur_motion", "unblur_focus", "face", "upscale2x"}
	stepsSchema = map[string]any{
		"type": "array", "minItems": 1, "maxItems": enhance.MaxSteps,
		"description": "Operations to run, in order. One step is a single operation; 2 to 4 make a pipeline " +
			"(at most one upscale2x). denoise: high-ISO noise. sharpen: soft detail. unblur_motion: camera shake " +
			"or motion blur. unblur_focus: missed focus. face: restores faces, leaves the rest unchanged. " +
			"upscale2x: doubles width and height.",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"operation": map[string]any{"type": "string", "enum": ops},
				"strength": map[string]any{"type": "number", "minimum": 0, "maximum": 1,
					"description": "0 = no change, 1 = full effect. Not for upscale2x. Server defaults: 0.5 " +
						"for denoise, sharpen and face; 1.0 for unblur."},
			},
			"required":             []string{"operation"},
			"additionalProperties": false,
		},
	}
	modelSchema = map[string]any{"type": "string",
		"description": "Model ID to pin (from server_status). Single operation only; default: the server's default."}
	formatSchema = map[string]any{"type": "string", "enum": []string{"jpeg", "tiff"},
		"description": "Output format. Default: same as the input (or from the output file's extension)."}
	qualitySchema = map[string]any{"type": "integer", "minimum": 1, "maximum": 100, "default": 92,
		"description": "JPEG quality for JPEG output."}
	jobIDSchema = map[string]any{"type": "string", "pattern": "^job_[0-9a-z]{26}$",
		"description": "Job ID from submit_job or enhance_photos."}
)

func object(props map[string]any, required []string) map[string]any {
	if props == nil {
		props = map[string]any{}
	}
	o := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		o["required"] = required
	}
	return o
}

type tools struct {
	client ClientFunc
}

type stepArg struct {
	Operation string   `json:"operation"`
	Strength  *float64 `json:"strength,omitempty"`
}

func toSteps(args []stepArg, model string) ([]enhance.Step, *Result) {
	steps := make([]enhance.Step, len(args))
	for i, a := range args {
		steps[i] = enhance.Step{Operation: a.Operation, Strength: a.Strength}
	}
	if err := enhance.CheckSteps(steps); err != nil {
		return nil, ErrorResult("invalid_params", err.Error())
	}
	if model != "" && len(steps) > 1 {
		return nil, ErrorResult("invalid_params", "model can be pinned only for a single operation")
	}
	return steps, nil
}

// resolve expands a leading ~ and makes the path absolute against the
// server's working directory.
func resolve(p string) string {
	p = strings.TrimSpace(p)
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			p = filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	if abs, err := filepath.Abs(p); err == nil && p != "" {
		// Keep a trailing slash: it marks an output directory.
		if strings.HasSuffix(p, "/") || strings.HasSuffix(p, string(os.PathSeparator)) {
			return abs + string(os.PathSeparator)
		}
		return abs
	}
	return p
}

var extFormat = map[string]string{".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff"}

// checkInput confirms the photo is a readable JPEG or TIFF and returns its format.
func checkInput(path string) (string, *Result) {
	st, err := os.Stat(path)
	switch {
	case os.IsNotExist(err):
		return "", ErrorResult("input_not_found", path+" does not exist")
	case err != nil:
		return "", ErrorResult("input_unreadable", err.Error())
	case !st.Mode().IsRegular():
		return "", ErrorResult("input_unreadable", path+" is not a regular file")
	}
	f := extFormat[strings.ToLower(filepath.Ext(path))]
	if f == "" {
		return "", ErrorResult("unsupported_format", "input must be a JPEG (.jpg, .jpeg) or TIFF (.tif, .tiff)")
	}
	return f, nil
}

func apiError(err error) *Result {
	e := api.AsError(err)
	return ErrorResult(e.Code, e.Message)
}

func (t *tools) status(ctx context.Context, c *Call) *Result {
	if err := c.Bind(&struct{}{}); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	client.Retries = 1
	start := time.Now()
	health, err := client.Health(ctx)
	if err != nil {
		return apiError(err)
	}
	rtt := time.Since(start)
	caps, err := client.Capabilities(ctx)
	if err != nil {
		return apiError(err)
	}
	gpu := "none"
	if health.GPU.Available && health.GPU.Name != nil {
		gpu = *health.GPU.Name
	}
	summary := fmt.Sprintf("Server ok, version %s, GPU %s, %d queued / %d running, %d ms round trip.",
		health.Version, gpu, health.Queue.Queued, health.Queue.Running, rtt.Milliseconds())
	return TextResult(summary, map[string]any{
		"health": health, "capabilities": caps, "round_trip_ms": float64(rtt.Microseconds()) / 1000,
	})
}

// lineWriter turns the batch runner's progress lines into progress notifications.
type lineWriter struct {
	mu  sync.Mutex
	buf []byte
	fn  func(string)
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			return len(p), nil
		}
		w.fn(string(w.buf[:i]))
		w.buf = w.buf[i+1:]
	}
}

func (t *tools) enhancePhotos(ctx context.Context, c *Call) *Result {
	var a struct {
		Inputs      []string  `json:"inputs"`
		Output      string    `json:"output"`
		Steps       []stepArg `json:"steps"`
		Model       string    `json:"model"`
		Format      string    `json:"format"`
		JPEGQuality int       `json:"jpeg_quality"`
		Overwrite   bool      `json:"overwrite"`
	}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	if len(a.Inputs) == 0 || len(a.Inputs) > maxInputs {
		return ErrorResult("invalid_params", fmt.Sprintf("give 1 to %d inputs", maxInputs))
	}
	if strings.TrimSpace(a.Output) == "" {
		return ErrorResult("invalid_params", "output is required")
	}
	if a.JPEGQuality == 0 {
		a.JPEGQuality = 92
	}
	if a.JPEGQuality < 1 || a.JPEGQuality > 100 {
		return ErrorResult("invalid_params", "jpeg_quality must be between 1 and 100")
	}
	steps, bad := toSteps(a.Steps, a.Model)
	if bad != nil {
		return bad
	}
	inputs := make([]string, len(a.Inputs))
	for i, in := range a.Inputs {
		inputs[i] = resolve(in)
	}
	items, ok, err := plan.Build(inputs, plan.Options{Output: resolve(a.Output), Format: a.Format,
		Overwrite: a.Overwrite})
	if err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	if !ok {
		var probs []map[string]any
		for _, it := range items {
			if it.Err != nil {
				probs = append(probs, map[string]any{"input": it.Input, "output": it.Output,
					"error": it.Err})
			}
		}
		r := TextResult("Nothing was submitted: fix these inputs and try again.", map[string]any{"invalid": probs})
		r.IsError = true
		return r
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	dirs := map[string]bool{}
	for _, it := range items {
		dirs[filepath.Dir(it.Output)] = true
	}
	for d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return ErrorResult("write_failed", fmt.Sprintf("creating output directory: %v", err))
		}
	}
	results := enhance.Run(ctx, client, items, enhance.Options{
		Steps: steps, Model: a.Model, JPEGQuality: a.JPEGQuality, Overwrite: a.Overwrite,
		Jobs: 4, Progress: &lineWriter{fn: c.Progress},
	})
	failed := 0
	for _, r := range results {
		if r.Status != "succeeded" {
			failed++
		}
	}
	summary := fmt.Sprintf("%d of %d photos enhanced.", len(results)-failed, len(results))
	if failed > 0 {
		summary += fmt.Sprintf(" %d failed; see error in each result.", failed)
	}
	r := TextResult(summary, map[string]any{"results": results, "succeeded": len(results) - failed,
		"failed": failed})
	r.IsError = failed == len(results)
	return r
}

func (t *tools) preview(ctx context.Context, c *Call) *Result {
	var a struct {
		Input     string      `json:"input"`
		Steps     []stepArg   `json:"steps"`
		Model     string      `json:"model"`
		Region    *api.Region `json:"region"`
		Output    string      `json:"output"`
		Overwrite bool        `json:"overwrite"`
	}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	steps, bad := toSteps(a.Steps, a.Model)
	if bad != nil {
		return bad
	}
	input := resolve(a.Input)
	format, bad := checkInput(input)
	if bad != nil {
		return bad
	}
	output := ""
	if a.Output != "" {
		output = resolve(a.Output)
		if extFormat[strings.ToLower(filepath.Ext(output))] != "jpeg" {
			return ErrorResult("invalid_params", "output for a preview must end in .jpg or .jpeg")
		}
		if _, err := os.Stat(output); err == nil && !a.Overwrite {
			return ErrorResult("destination_exists", output+" exists; pass overwrite to replace it")
		}
	}
	region := a.Region
	if region == nil {
		region = centerRegion(input, format)
	}
	if region.X < 0 || region.Y < 0 || region.W < 1 || region.H < 1 ||
		region.W > maxPreviewSide || region.H > maxPreviewSide {
		return ErrorResult("invalid_params", fmt.Sprintf("region needs x, y >= 0 and w, h from 1 to %d",
			maxPreviewSide))
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	job, err := enhance.Submit(ctx, client, input, enhance.Request{Steps: steps, Model: a.Model, Kind: "preview",
		Format: "jpeg", JPEGQuality: 90, Region: region}, c.Progress)
	if err != nil {
		return apiError(err)
	}
	id := job.ID
	job, err = waitJob(ctx, client, job, c.Progress)
	if err != nil {
		if ctx.Err() != nil {
			cctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			_ = client.CancelJob(cctx, id)
			cancel()
		}
		return apiError(err)
	}
	if job.State != "completed" {
		return jobFailed(job)
	}

	tmp, err := os.CreateTemp("", "painite-preview-*.jpg")
	if err != nil {
		return ErrorResult("write_failed", err.Error())
	}
	tmpName := tmp.Name()
	tmp.Close()
	defer os.Remove(tmpName)
	if err := enhance.Download(ctx, client, job.ID, tmpName, true); err != nil {
		return apiError(err)
	}
	dctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_ = client.DeleteResult(dctx, job.ID)
	cancel()
	data, err := os.ReadFile(tmpName)
	if err != nil {
		return ErrorResult("write_failed", err.Error())
	}
	if output != "" {
		if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
			return ErrorResult("write_failed", err.Error())
		}
		if err := os.WriteFile(output, data, 0o644); err != nil {
			return ErrorResult("write_failed", err.Error())
		}
	}
	inline, w, h, err := fitJPEG(data, inlineImageSide)
	if err != nil {
		return ErrorResult("processing_failed", "decoding the preview: "+err.Error())
	}

	full, _ := client.GetJobJSON(ctx, job.ID)
	info := map[string]any{"job_id": job.ID, "model": job.Model, "region": job.Region,
		"timings_ms": job.TimingsMS, "image_size": map[string]int{"width": w, "height": h}}
	if job.Result != nil {
		info["preview_size"] = map[string]int{"width": job.Result.Width, "height": job.Result.Height}
	}
	if full != nil && full["faces"] != nil {
		info["faces"] = full["faces"]
	}
	if output != "" {
		info["saved_to"] = output
	}
	summary := "Preview with " + job.Model + "."
	if rg := job.Region; rg != nil {
		summary = fmt.Sprintf("Preview of x=%d y=%d w=%d h=%d with %s.", rg.X, rg.Y, rg.W, rg.H, job.Model)
	}
	if faces, ok := info["faces"].([]any); ok {
		summary += " " + faceSummary(faces)
	}
	r := TextResult(summary, info)
	r.Content = append(r.Content, map[string]any{"type": "image", "mimeType": "image/jpeg",
		"data": base64.StdEncoding.EncodeToString(inline)})
	return r
}

// faceSummary counts faces by status, for example "4 faces found: 3 enhanced, 1 too_small.".
func faceSummary(faces []any) string {
	if len(faces) == 0 {
		return "No faces found."
	}
	counts := map[string]int{}
	var order []string
	for _, f := range faces {
		st, _ := f.(map[string]any)["status"].(string)
		if counts[st] == 0 {
			order = append(order, st)
		}
		counts[st]++
	}
	parts := make([]string, len(order))
	for i, st := range order {
		parts[i] = fmt.Sprintf("%d %s", counts[st], st)
	}
	noun := "faces"
	if len(faces) == 1 {
		noun = "face"
	}
	return fmt.Sprintf("%d %s found: %s.", len(faces), noun, strings.Join(parts, ", "))
}

// centerRegion is a default preview square at the center of a JPEG, or at
// the top left when the size is unknown (TIFF). The server clips regions
// that run past the edge.
func centerRegion(path, format string) *api.Region {
	r := &api.Region{W: defaultPreview, H: defaultPreview}
	if format != "jpeg" {
		return r
	}
	f, err := os.Open(path)
	if err != nil {
		return r
	}
	defer f.Close()
	cfg, err := jpeg.DecodeConfig(f)
	if err != nil {
		return r
	}
	// Stored dimensions; with EXIF rotation the center moves, but the
	// square stays inside the image for any usual aspect ratio.
	r.X = max(0, (cfg.Width-defaultPreview)/2)
	r.Y = max(0, (cfg.Height-defaultPreview)/2)
	return r
}

// fitJPEG scales a JPEG down so its long edge is at most side, by area
// averaging, and re-encodes it. Small images are returned as they are.
func fitJPEG(data []byte, side int) ([]byte, int, int, error) {
	src, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, err
	}
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw <= side && sh <= side {
		return data, sw, sh, nil
	}
	scale := float64(side) / float64(max(sw, sh))
	dw, dh := max(1, int(float64(sw)*scale+0.5)), max(1, int(float64(sh)*scale+0.5))
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		y0, y1 := y*sh/dh, max((y+1)*sh/dh, y*sh/dh+1)
		for x := 0; x < dw; x++ {
			x0, x1 := x*sw/dw, max((x+1)*sw/dw, x*sw/dw+1)
			var r, g, bl, n uint64
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					cr, cg, cb, _ := src.At(b.Min.X+xx, b.Min.Y+yy).RGBA()
					r, g, bl, n = r+uint64(cr), g+uint64(cg), bl+uint64(cb), n+1
				}
			}
			i := dst.PixOffset(x, y)
			dst.Pix[i] = uint8(r / n >> 8)
			dst.Pix[i+1] = uint8(g / n >> 8)
			dst.Pix[i+2] = uint8(bl / n >> 8)
			dst.Pix[i+3] = 255
		}
	}
	var out bytes.Buffer
	if err := jpeg.Encode(&out, dst, &jpeg.Options{Quality: 88}); err != nil {
		return nil, 0, 0, err
	}
	return out.Bytes(), dw, dh, nil
}

// waitJob polls until the job ends, reporting state changes as progress.
func waitJob(ctx context.Context, client *api.Client, job *api.Job, progress func(string)) (*api.Job, error) {
	last := ""
	return enhance.Wait(ctx, client, job, func(j *api.Job) {
		s := j.State
		if s == "running" {
			s = fmt.Sprintf("running %d%%", int(j.Progress*10)*10)
		}
		if s != last {
			progress(s)
			last = s
		}
	})
}

var jobIDPattern = regexp.MustCompile(`^job_[0-9a-z]{26}$`)

func checkJobID(id string) *Result {
	if !jobIDPattern.MatchString(id) {
		return ErrorResult("invalid_params", fmt.Sprintf("%q is not a job ID (job_ followed by 26 characters)", id))
	}
	return nil
}

func jobFailed(job *api.Job) *Result {
	e := job.Error
	if e == nil {
		e = &api.Error{Code: job.State, Message: "job " + job.State}
	}
	r := ErrorResult(e.Code, e.Message)
	r.StructuredContent = map[string]any{"job_id": job.ID, "state": job.State, "error": e}
	return r
}

func (t *tools) submit(ctx context.Context, c *Call) *Result {
	var a struct {
		Input       string    `json:"input"`
		Steps       []stepArg `json:"steps"`
		Model       string    `json:"model"`
		Format      string    `json:"format"`
		JPEGQuality int       `json:"jpeg_quality"`
	}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	steps, bad := toSteps(a.Steps, a.Model)
	if bad != nil {
		return bad
	}
	input := resolve(a.Input)
	format, bad := checkInput(input)
	if bad != nil {
		return bad
	}
	if a.Format != "" {
		if a.Format != "jpeg" && a.Format != "tiff" {
			return ErrorResult("invalid_params", "format must be jpeg or tiff")
		}
		format = a.Format
	}
	if a.JPEGQuality == 0 {
		a.JPEGQuality = 92
	}
	if a.JPEGQuality < 1 || a.JPEGQuality > 100 {
		return ErrorResult("invalid_params", "jpeg_quality must be between 1 and 100")
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	job, err := enhance.Submit(ctx, client, input, enhance.Request{Steps: steps, Model: a.Model, Kind: "full",
		Format: format, JPEGQuality: a.JPEGQuality}, c.Progress)
	if err != nil {
		return apiError(err)
	}
	ext := map[string]string{"jpeg": ".jpg", "tiff": ".tif"}[format]
	return TextResult(fmt.Sprintf("Submitted %s (%s). The result will be %s; save it with a %s file name.",
		job.ID, job.State, format, ext), map[string]any{"job_id": job.ID, "state": job.State, "model": job.Model,
		"input": input, "format": format})
}

func (t *tools) getJob(ctx context.Context, c *Call) *Result {
	var a struct {
		JobID       string `json:"job_id"`
		WaitSeconds int    `json:"wait_seconds"`
	}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	if bad := checkJobID(a.JobID); bad != nil {
		return bad
	}
	if a.WaitSeconds < 0 || a.WaitSeconds > maxWaitSeconds {
		return ErrorResult("invalid_params", fmt.Sprintf("wait_seconds must be 0 to %d", maxWaitSeconds))
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	if a.WaitSeconds > 0 {
		job, err := client.GetJob(ctx, a.JobID)
		if err != nil {
			return apiError(err)
		}
		wctx, cancel := context.WithTimeout(ctx, time.Duration(a.WaitSeconds)*time.Second)
		_, err = waitJob(wctx, client, job, c.Progress)
		cancel()
		if err != nil && ctx.Err() != nil {
			return apiError(ctx.Err())
		}
		if err != nil && wctx.Err() == nil {
			return apiError(err)
		}
	}
	job, err := client.GetJobJSON(ctx, a.JobID)
	if err != nil {
		return apiError(err)
	}
	summary := fmt.Sprintf("%s is %v.", a.JobID, job["state"])
	if job["state"] == "running" {
		summary = fmt.Sprintf("%s is running, %.0f%% done.", a.JobID, toFloat(job["progress"])*100)
	}
	return TextResult(summary, job)
}

func toFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

func (t *tools) download(ctx context.Context, c *Call) *Result {
	a := struct {
		JobID            string `json:"job_id"`
		Output           string `json:"output"`
		Overwrite        bool   `json:"overwrite"`
		DeleteFromServer *bool  `json:"delete_from_server"`
	}{}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	if bad := checkJobID(a.JobID); bad != nil {
		return bad
	}
	if strings.TrimSpace(a.Output) == "" {
		return ErrorResult("invalid_params", "output is required")
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	job, err := client.GetJob(ctx, a.JobID)
	if err != nil {
		return apiError(err)
	}
	if job.State != "completed" {
		if job.Terminal() {
			return jobFailed(job)
		}
		return ErrorResult("result_not_ready", fmt.Sprintf("%s is %s; wait with get_job first", job.ID, job.State))
	}
	if job.Result == nil || job.Result.Deleted {
		return ErrorResult("result_unavailable", "the result was deleted or has expired")
	}
	ext := map[string]string{"jpeg": ".jpg", "tiff": ".tif"}[job.Result.Format]
	out := resolve(a.Output)
	if st, err := os.Stat(out); (err == nil && st.IsDir()) || strings.HasSuffix(out, string(os.PathSeparator)) {
		out = filepath.Join(out, job.ID+ext)
	} else if extFormat[strings.ToLower(filepath.Ext(out))] != job.Result.Format {
		return ErrorResult("invalid_params", fmt.Sprintf("the result is %s; use a %s file name",
			job.Result.Format, ext))
	}
	if st, err := os.Stat(out); err == nil {
		if st.IsDir() {
			return ErrorResult("destination_exists", out+" is a directory")
		}
		if !a.Overwrite {
			return ErrorResult("destination_exists", out+" exists; pass overwrite to replace it")
		}
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return ErrorResult("write_failed", err.Error())
	}
	if err := enhance.Download(ctx, client, job.ID, out, a.Overwrite); err != nil {
		return apiError(err)
	}
	deleted := false
	if a.DeleteFromServer == nil || *a.DeleteFromServer {
		dctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		deleted = client.DeleteResult(dctx, job.ID) == nil
		cancel()
	}
	return TextResult(fmt.Sprintf("Saved %s.", out), map[string]any{"job_id": job.ID, "output": out,
		"bytes": job.Result.Bytes, "width": job.Result.Width, "height": job.Result.Height,
		"format": job.Result.Format, "deleted_from_server": deleted})
}

func (t *tools) cancel(ctx context.Context, c *Call) *Result {
	var a struct {
		JobID string `json:"job_id"`
	}
	if err := c.Bind(&a); err != nil {
		return ErrorResult("invalid_params", err.Error())
	}
	if bad := checkJobID(a.JobID); bad != nil {
		return bad
	}
	client, err := t.client()
	if err != nil {
		return apiError(err)
	}
	if err := client.CancelJob(ctx, a.JobID); err != nil {
		return apiError(err)
	}
	// A running job stops at its next tile; wait briefly for the final state.
	if job, err := client.GetJob(ctx, a.JobID); err == nil {
		wctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		_, _ = enhance.Wait(wctx, client, job, nil)
		cancel()
	}
	job, err := client.GetJobJSON(ctx, a.JobID)
	if err != nil {
		return apiError(err)
	}
	summary := fmt.Sprintf("%s is %v.", a.JobID, job["state"])
	if job["state"] == "queued" || job["state"] == "running" {
		summary = fmt.Sprintf("Cancel requested; %s is still %v. Check again with get_job.", a.JobID, job["state"])
	}
	return TextResult(summary, job)
}
