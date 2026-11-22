// Command painite enhances photos on a PAINITE inference server.
//
// Exit codes: 0 all files succeeded, 1 any file failed, 2 invalid invocation.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/yuanlin2004/PAINITE/cli/internal/api"
	"github.com/yuanlin2004/PAINITE/cli/internal/config"
	"github.com/yuanlin2004/PAINITE/cli/internal/enhance"
	"github.com/yuanlin2004/PAINITE/cli/internal/plan"
)

// version is set at build time with -ldflags "-X main.version=...".
var version = "dev"

const (
	exitOK      = 0
	exitFailed  = 1
	exitInvalid = 2
)

const usage = `painite enhances photos on a PAINITE inference server.

Usage:
  painite enhance --op <denoise|sharpen|unblur_motion|unblur_focus|upscale2x> [--strength 0..1] -o <file|dir> [flags] <input>...
  painite enhance --op denoise:0.6,sharpen:0.3,upscale2x -o <file|dir> [flags] <input>...   (pipeline)
  painite trust --fingerprint SHA256:<hex>
  painite status [--json]
  painite config set <key> <value>
  painite config get <key>
  painite config show
  painite version

Run "painite <command> -h" for a command's flags.

Exit codes: 0 all succeeded, 1 any file failed, 2 invalid invocation.
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return exitInvalid
	}
	switch args[0] {
	case "enhance":
		return cmdEnhance(args[1:], stdout, stderr)
	case "status":
		return cmdStatus(args[1:], stdout, stderr)
	case "config":
		return cmdConfig(args[1:], stdout, stderr)
	case "trust":
		return cmdTrust(args[1:], stdout, stderr)
	case "version", "--version":
		fmt.Fprintf(stdout, "painite %s\n", version)
		return exitOK
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
		return exitOK
	}
	fmt.Fprintf(stderr, "painite: unknown command %q\n\n%s", args[0], usage)
	return exitInvalid
}

// parseInterspersed lets flags appear before or after positional arguments.
func parseInterspersed(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return pos, nil
		}
		// flag stops at "--": everything after it is positional.
		if consumed := len(args) - len(rest); consumed > 0 && args[consumed-1] == "--" {
			return append(pos, rest...), nil
		}
		pos = append(pos, rest[0])
		args = rest[1:]
	}
}

type connFlags struct {
	endpoint, token, caFile string
}

func (c *connFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&c.endpoint, "endpoint", "", "server URL (overrides config and PAINITE_ENDPOINT)")
	fs.StringVar(&c.token, "token", "", "client token (overrides config and PAINITE_TOKEN)")
	fs.StringVar(&c.caFile, "ca-file", "", "CA certificate PEM (overrides config and PAINITE_CA_FILE)")
}

func (c *connFlags) client() (*api.Client, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	if c.endpoint != "" {
		cfg.Endpoint = c.endpoint
	}
	if c.token != "" {
		cfg.Token = c.token
	}
	if c.caFile != "" {
		cfg.CAFile = c.caFile
	}
	return api.New(cfg.Endpoint, cfg.Token, cfg.CAFile)
}

// optionalFloat records whether a float flag was given.
type optionalFloat struct{ v *float64 }

func (o *optionalFloat) String() string {
	if o.v == nil {
		return ""
	}
	return fmt.Sprint(*o.v)
}

func (o *optionalFloat) Set(s string) error {
	var f float64
	if _, err := fmt.Sscan(s, &f); err != nil {
		return fmt.Errorf("not a number: %s", s)
	}
	o.v = &f
	return nil
}

func cmdEnhance(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("painite enhance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		op, model, format, output string
		quality, jobs             int
		overwrite, asJSON, quiet  bool
		strength                  optionalFloat
		conn                      connFlags
	)
	fs.StringVar(&op, "op", "", "operation: denoise, sharpen, unblur_motion, unblur_focus, or upscale2x (required); a comma-separated list runs\n"+
		"a pipeline in one job, and op:strength sets one step's strength, e.g. denoise:0.6,sharpen:0.3,upscale2x")
	fs.Var(&strength, "strength", "strength 0..1 for denoise, sharpen and unblur steps without their own (default: server default)")
	fs.StringVar(&model, "model", "", "model ID to pin, single operation only (default: server default for the operation)")
	fs.StringVar(&format, "format", "", "output format: jpeg or tiff (default: same as input, or from -o's extension)")
	fs.IntVar(&quality, "jpeg-quality", 92, "JPEG quality 1..100")
	fs.StringVar(&output, "o", "", "output file, or directory for several inputs (required)")
	fs.StringVar(&output, "output", "", "same as -o")
	fs.BoolVar(&overwrite, "overwrite", false, "replace existing output files")
	fs.BoolVar(&asJSON, "json", false, "print one JSON result per input on stdout")
	fs.BoolVar(&quiet, "quiet", false, "no progress on stderr")
	fs.IntVar(&jobs, "jobs", 4, "files in flight at once")
	conn.register(fs)

	inputs, err := parseInterspersed(fs, args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitInvalid
	}
	invalid := func(msg string) int {
		fmt.Fprintf(stderr, "painite: %s\n", msg)
		if asJSON {
			writeJSON(stdout, notRun(inputs, nil, &api.Error{Code: "invalid_invocation", Message: msg}))
		}
		return exitInvalid
	}
	if strength.v != nil && (*strength.v < 0 || *strength.v > 1) {
		return invalid("--strength must be between 0 and 1")
	}
	steps, err := parseSteps(op, strength.v)
	if err != nil {
		return invalid(err.Error())
	}
	if model != "" && len(steps) > 1 {
		return invalid("--model works only with a single operation")
	}
	if quality < 1 || quality > 100 {
		return invalid("--jpeg-quality must be between 1 and 100")
	}
	if jobs < 1 || jobs > 16 {
		return invalid("--jobs must be between 1 and 16")
	}

	items, ok, err := plan.Build(inputs, plan.Options{Output: output, Format: format, Overwrite: overwrite})
	if err != nil {
		return invalid(err.Error())
	}
	if !ok {
		for _, it := range items {
			if it.Err != nil {
				fmt.Fprintf(stderr, "painite: %s: %s\n", it.Input, it.Err.Message)
			}
		}
		fmt.Fprintln(stderr, "painite: nothing was submitted")
		if asJSON {
			writeJSON(stdout, notRun(inputs, items, nil))
		}
		return exitInvalid
	}

	client, err := conn.client()
	if err != nil {
		return invalid(err.Error())
	}
	dirs := map[string]bool{}
	for _, it := range items {
		dirs[filepath.Dir(it.Output)] = true
	}
	for d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return invalid(fmt.Sprintf("creating output directory: %v", err))
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var progress io.Writer = stderr
	if quiet {
		progress = nil
	}
	results := enhance.Run(ctx, client, items, enhance.Options{
		Steps: steps, Model: model, JPEGQuality: quality, Overwrite: overwrite,
		Jobs: jobs, Progress: progress,
	})

	failed := 0
	for _, r := range results {
		if r.Status != "succeeded" {
			failed++
		}
	}
	if asJSON {
		writeJSON(stdout, results)
	}
	if !quiet && len(results) > 1 {
		fmt.Fprintf(stderr, "painite: %d succeeded, %d failed\n", len(results)-failed, failed)
	}
	if failed > 0 {
		return exitFailed
	}
	return exitOK
}

// notRun builds JSON results for a batch rejected before submission.
func notRun(inputs []string, items []plan.Item, all *api.Error) []enhance.Result {
	out := make([]enhance.Result, len(inputs))
	for i, in := range inputs {
		out[i] = enhance.Result{Input: in, Status: "not_run", Error: all}
		if items != nil {
			out[i].Output = items[i].Output
			if e := items[i].Err; e != nil {
				out[i].Status = "invalid"
				out[i].Error = &api.Error{Code: e.Code, Message: e.Message}
			}
		}
	}
	return out
}

func writeJSON(w io.Writer, results []enhance.Result) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(map[string]any{"results": results})
}

func cmdStatus(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("painite status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var asJSON bool
	var conn connFlags
	fs.BoolVar(&asJSON, "json", false, "print JSON")
	conn.register(fs)
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitInvalid
	}
	client, err := conn.client()
	if err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitInvalid
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client.Retries = 1
	start := time.Now()
	health, err := client.Health(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "painite: cannot reach server: %v\n", err)
		return exitFailed
	}
	rtt := time.Since(start)
	caps, capsErr := client.Capabilities(ctx)

	if asJSON {
		out := map[string]any{"health": health, "round_trip_ms": float64(rtt.Microseconds()) / 1000}
		if capsErr == nil {
			out["capabilities"] = caps
		} else {
			out["capabilities_error"] = api.AsError(capsErr)
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
	} else {
		gpu := "none"
		if health.GPU.Available && health.GPU.Name != nil {
			gpu = *health.GPU.Name
		}
		fmt.Fprintf(stdout, "Server:  ok (version %s, %d ms round trip)\n", health.Version, rtt.Milliseconds())
		fmt.Fprintf(stdout, "GPU:     %s\n", gpu)
		fmt.Fprintf(stdout, "Queue:   %d queued, %d running, capacity %d\n", health.Queue.Queued,
			health.Queue.Running, health.Queue.Capacity)
		if capsErr == nil {
			fmt.Fprintln(stdout, "Models:")
			sort.Slice(caps.Models, func(i, j int) bool { return caps.Models[i].ID < caps.Models[j].ID })
			for _, m := range caps.Models {
				note := ""
				if m.Placeholder {
					note = " (placeholder)"
				}
				fmt.Fprintf(stdout, "  %s%s\n", m.ID, note)
			}
		}
	}
	if capsErr != nil {
		fmt.Fprintf(stderr, "painite: capabilities: %v\n", capsErr)
		return exitFailed
	}
	return exitOK
}

func cmdConfig(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, "usage: painite config set <key> <value> | get <key> | show | path\n")
		return exitInvalid
	}
	cfg, err := config.LoadFile()
	if err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	switch args[0] {
	case "set":
		if len(args) != 3 {
			fmt.Fprint(stderr, "usage: painite config set <key> <value>\n")
			return exitInvalid
		}
		if err := cfg.Set(args[1], args[2]); err != nil {
			fmt.Fprintf(stderr, "painite: %v\n", err)
			return exitInvalid
		}
		if err := config.Save(cfg); err != nil {
			fmt.Fprintf(stderr, "painite: %v\n", err)
			return exitFailed
		}
	case "get":
		if len(args) != 2 {
			fmt.Fprint(stderr, "usage: painite config get <key>\n")
			return exitInvalid
		}
		v, err := cfg.Get(args[1])
		if err != nil {
			fmt.Fprintf(stderr, "painite: %v\n", err)
			return exitInvalid
		}
		fmt.Fprintln(stdout, v)
	case "show":
		p, _ := config.Path()
		fmt.Fprintf(stdout, "file:     %s\nendpoint: %s\ntoken:    %s\nca_file:  %s\n", p, cfg.Endpoint,
			cfg.MaskedToken(), cfg.CAFile)
	case "path":
		p, _ := config.Path()
		fmt.Fprintln(stdout, p)
	default:
		fmt.Fprintf(stderr, "painite: unknown config command %q\n", strings.Join(args, " "))
		return exitInvalid
	}
	return exitOK
}

// cmdTrust pins the server's private CA after checking its fingerprint
// against the one the operator printed on the server.
func cmdTrust(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("painite trust", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var fingerprint, endpoint string
	fs.StringVar(&fingerprint, "fingerprint", "", "CA fingerprint from `painite-server admin fingerprint`")
	fs.StringVar(&endpoint, "endpoint", "", "server URL (default: configured endpoint)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return exitOK
		}
		return exitInvalid
	}
	cfg, err := config.LoadFile()
	if err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	if endpoint != "" {
		if err := cfg.Set("endpoint", endpoint); err != nil {
			fmt.Fprintf(stderr, "painite: %v\n", err)
			return exitInvalid
		}
	}
	if cfg.Endpoint == "" {
		fmt.Fprintln(stderr, "painite: no endpoint; pass --endpoint or run `painite config set endpoint <url>`")
		return exitInvalid
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pemBytes, got, err := api.FetchCA(ctx, cfg.Endpoint)
	if err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	if fingerprint == "" {
		fmt.Fprintln(stdout, got)
		fmt.Fprintln(stderr, "painite: compare this with `painite-server admin fingerprint` on the server, "+
			"then rerun with --fingerprint")
		return exitFailed
	}
	if api.NormalizeFingerprint(fingerprint) != got {
		fmt.Fprintf(stderr, "painite: fingerprint mismatch: server presented %s; not trusted\n", got)
		return exitFailed
	}
	cfgPath, err := config.Path()
	if err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	caPath := filepath.Join(filepath.Dir(cfgPath), "ca.pem")
	if err := os.MkdirAll(filepath.Dir(caPath), 0o700); err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	if err := os.WriteFile(caPath, pemBytes, 0o644); err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	cfg.CAFile = caPath
	if err := config.Save(cfg); err != nil {
		fmt.Fprintf(stderr, "painite: %v\n", err)
		return exitFailed
	}
	// Confirm the server's certificate now verifies, including its name.
	client, err := api.New(cfg.Endpoint, cfg.Token, cfg.CAFile)
	if err == nil {
		client.Retries = 0
		_, err = client.Health(ctx)
	}
	if err != nil {
		fmt.Fprintf(stderr, "painite: CA saved to %s, but the connection still fails: %v\n", caPath, err)
		return exitFailed
	}
	fmt.Fprintf(stderr, "Trusted %s for %s (saved to %s).\n", got, cfg.Endpoint, caPath)
	return exitOK
}

// maxSteps matches the server's limit on pipeline length.
const maxSteps = 4

// strengthOps take a strength; other known operations don't.
var strengthOps = map[string]bool{"denoise": true, "sharpen": true, "unblur_motion": true, "unblur_focus": true}
var knownOps = map[string]bool{"denoise": true, "sharpen": true, "unblur_motion": true, "unblur_focus": true,
	"upscale2x": true}

const opList = "denoise, sharpen, unblur_motion, unblur_focus, or upscale2x"

// parseSteps reads --op: "denoise", or a pipeline "denoise:0.6,sharpen,upscale2x".
// defaultStrength applies to strength steps that don't set their own.
func parseSteps(spec string, defaultStrength *float64) ([]enhance.Step, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, fmt.Errorf("--op is required (%s)", opList)
	}
	var steps []enhance.Step
	upscales := 0
	for _, part := range strings.Split(spec, ",") {
		name, val, hasVal := strings.Cut(strings.TrimSpace(part), ":")
		if name == "unblur" {
			return nil, fmt.Errorf("unblur comes in two kinds: unblur_motion (camera shake, motion blur) or " +
				"unblur_focus (out of focus)")
		}
		if !knownOps[name] {
			return nil, fmt.Errorf("unknown operation %q in --op (use %s)", name, opList)
		}
		st := enhance.Step{Operation: name}
		if hasVal {
			if !strengthOps[name] {
				return nil, fmt.Errorf("%s does not take a strength", name)
			}
			var f float64
			if _, err := fmt.Sscan(val, &f); err != nil || f < 0 || f > 1 {
				return nil, fmt.Errorf("strength for %s must be a number between 0 and 1", name)
			}
			st.Strength = &f
		} else if strengthOps[name] {
			st.Strength = defaultStrength
		}
		if name == "upscale2x" {
			upscales++
		}
		steps = append(steps, st)
	}
	if len(steps) > maxSteps {
		return nil, fmt.Errorf("a pipeline has at most %d steps", maxSteps)
	}
	if upscales > 1 {
		return nil, fmt.Errorf("a pipeline can upscale at most once")
	}
	if defaultStrength != nil {
		any := false
		for _, st := range steps {
			any = any || strengthOps[st.Operation]
		}
		if !any {
			return nil, fmt.Errorf("--strength applies only to denoise, sharpen, unblur_motion and unblur_focus")
		}
	}
	return steps, nil
}
