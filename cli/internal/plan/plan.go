// Package plan resolves inputs to output paths and validates the whole batch
// before anything is uploaded (DESIGN.md section 6, PRD "Standalone CLI").
package plan

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Item is one input and where its result goes.
type Item struct {
	Input  string // as given on the command line
	Output string // as it will be written
	Format string // jpeg or tiff, sent to the server
	Err    *Error // validation failure, nil if valid
}

// Error is a validation failure with a stable code.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// Options are the flags that affect planning.
type Options struct {
	Output    string // -o: file or directory
	Format    string // --format: "", "jpeg", or "tiff"
	Overwrite bool
}

var inputFormats = map[string]string{".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff"}
var formatExt = map[string]string{"jpeg": ".jpg", "tiff": ".tif"}

// Build resolves every input. It returns ok=false if any item is invalid, in
// which case nothing should be submitted.
func Build(inputs []string, o Options) (items []Item, ok bool, err error) {
	if len(inputs) == 0 {
		return nil, false, errors.New("no input files")
	}
	if o.Output == "" {
		return nil, false, errors.New("an output path is required (-o)")
	}
	if o.Format != "" && formatExt[o.Format] == "" {
		return nil, false, fmt.Errorf("unsupported --format %q (use jpeg or tiff)", o.Format)
	}

	dirMode := len(inputs) > 1 || strings.HasSuffix(o.Output, "/") || strings.HasSuffix(o.Output, string(os.PathSeparator))
	if st, statErr := os.Stat(o.Output); statErr == nil && st.IsDir() {
		dirMode = true
	} else if statErr == nil && dirMode {
		return nil, false, fmt.Errorf("output %s is a file, but several inputs need an output directory", o.Output)
	}

	items = make([]Item, len(inputs))
	for i, in := range inputs {
		it := &items[i]
		it.Input = in
		inFmt := inputFormats[strings.ToLower(filepath.Ext(in))]

		if dirMode {
			format := o.Format
			ext := filepath.Ext(in)
			if format == "" {
				format = inFmt
			} else if format != inFmt {
				ext = formatExt[format]
			}
			it.Format = format
			it.Output = filepath.Join(o.Output, strings.TrimSuffix(filepath.Base(in), filepath.Ext(in))+ext)
		} else {
			it.Output = o.Output
			outFmt := inputFormats[strings.ToLower(filepath.Ext(o.Output))]
			switch {
			case outFmt == "":
				it.Err = &Error{"unsupported_format", fmt.Sprintf("output %s must end in .jpg, .jpeg, .tif, or .tiff", o.Output)}
			case o.Format != "" && o.Format != outFmt:
				it.Err = &Error{"unsupported_format", fmt.Sprintf("--format %s does not match output extension %s", o.Format, filepath.Ext(o.Output))}
			}
			it.Format = outFmt
		}
		if it.Err != nil {
			continue
		}

		st, statErr := os.Stat(in)
		switch {
		case errors.Is(statErr, fs.ErrNotExist):
			it.Err = &Error{"input_not_found", "input file does not exist"}
		case statErr != nil:
			it.Err = &Error{"input_unreadable", statErr.Error()}
		case !st.Mode().IsRegular():
			it.Err = &Error{"input_unreadable", "input is not a regular file"}
		case inFmt == "":
			it.Err = &Error{"unsupported_format", "input must be a JPEG (.jpg, .jpeg) or TIFF (.tif, .tiff)"}
		}
	}

	// Collisions and source-as-destination, compared on canonical paths.
	keys := make([]string, len(items))
	inKeys := map[string]bool{}
	for i := range items {
		keys[i] = canonical(items[i].Output)
		inKeys[canonical(items[i].Input)] = true
	}
	byOut := map[string][]int{}
	for i, k := range keys {
		byOut[k] = append(byOut[k], i)
	}
	for i := range items {
		it := &items[i]
		if it.Err != nil {
			continue
		}
		if n := len(byOut[keys[i]]); n > 1 {
			it.Err = &Error{"output_collision", fmt.Sprintf("%d inputs resolve to the same output %s", n, it.Output)}
			continue
		}
		if inKeys[keys[i]] || sameFileAsAnyInput(it.Output, items) {
			it.Err = &Error{"source_is_destination", "output would overwrite an input file"}
			continue
		}
		if st, err := os.Stat(it.Output); err == nil {
			if st.IsDir() {
				it.Err = &Error{"destination_exists", "output path is a directory"}
			} else if !o.Overwrite {
				it.Err = &Error{"destination_exists", "output exists; pass --overwrite to replace it"}
			}
		}
	}

	ok = true
	for i := range items {
		if items[i].Err != nil {
			ok = false
		}
	}
	return items, ok, nil
}

// canonical returns an absolute, symlink-resolved path for comparison. On
// macOS the default file system is case-insensitive, so case is folded.
func canonical(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		abs = p
	}
	dir, base := filepath.Split(abs)
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	} else {
		abs = filepath.Join(dir, base)
	}
	if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
		abs = strings.ToLower(abs)
	}
	return abs
}

func sameFileAsAnyInput(out string, items []Item) bool {
	ost, err := os.Stat(out)
	if err != nil {
		return false
	}
	for _, it := range items {
		if ist, err := os.Stat(it.Input); err == nil && os.SameFile(ost, ist) {
			return true
		}
	}
	return false
}
