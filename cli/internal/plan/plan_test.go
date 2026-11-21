package plan

import (
	"os"
	"path/filepath"
	"testing"
)

func touch(t *testing.T, p string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func codes(items []Item) []string {
	out := make([]string, len(items))
	for i, it := range items {
		if it.Err != nil {
			out[i] = it.Err.Code
		}
	}
	return out
}

func eq(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestSingleFileOutput(t *testing.T) {
	d := t.TempDir()
	in := touch(t, filepath.Join(d, "a.jpg"))
	items, ok, err := Build([]string{in}, Options{Output: filepath.Join(d, "out.tif")})
	if err != nil || !ok {
		t.Fatalf("ok=%v err=%v items=%+v", ok, err, items)
	}
	if items[0].Format != "tiff" || items[0].Output != filepath.Join(d, "out.tif") {
		t.Fatalf("%+v", items[0])
	}
}

func TestDirectoryModeKeepsNamesAndChangesExtForFormat(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "a.JPG"))
	b := touch(t, filepath.Join(d, "b.tiff"))
	out := filepath.Join(d, "out") + "/"
	items, ok, _ := Build([]string{a, b}, Options{Output: out})
	if !ok {
		t.Fatalf("%+v", items)
	}
	if items[0].Output != filepath.Join(d, "out", "a.JPG") || items[1].Output != filepath.Join(d, "out", "b.tiff") {
		t.Fatalf("%+v", items)
	}
	items, _, _ = Build([]string{a, b}, Options{Output: out, Format: "tiff"})
	if items[0].Output != filepath.Join(d, "out", "a.tif") || items[1].Output != filepath.Join(d, "out", "b.tiff") {
		t.Fatalf("%+v", items)
	}
}

func TestCollisionRejectedEvenWithOverwrite(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "x", "img.jpg"))
	b := touch(t, filepath.Join(d, "y", "img.jpg"))
	c := touch(t, filepath.Join(d, "y", "other.jpg"))
	items, ok, _ := Build([]string{a, b, c}, Options{Output: filepath.Join(d, "out"), Overwrite: true})
	if ok {
		t.Fatal("expected invalid")
	}
	eq(t, codes(items), []string{"output_collision", "output_collision", ""})
}

func TestSameInputTwiceCollides(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "img.jpg"))
	items, ok, _ := Build([]string{a, a}, Options{Output: filepath.Join(d, "out")})
	if ok {
		t.Fatal("expected invalid")
	}
	eq(t, codes(items), []string{"output_collision", "output_collision"})
}

func TestSourceIsDestination(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "img.jpg"))
	_, ok, _ := Build([]string{a}, Options{Output: a, Overwrite: true})
	if ok {
		t.Fatal("expected invalid")
	}
	items, ok, _ := Build([]string{a}, Options{Output: d, Overwrite: true})
	if ok {
		t.Fatal("expected invalid when writing into the source's directory with the same name")
	}
	eq(t, codes(items), []string{"source_is_destination"})
}

func TestExistingDestinationNeedsOverwrite(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "a.jpg"))
	out := touch(t, filepath.Join(d, "out", "a.jpg"))
	items, ok, _ := Build([]string{a}, Options{Output: filepath.Dir(out)})
	if ok {
		t.Fatal("expected invalid")
	}
	eq(t, codes(items), []string{"destination_exists"})
	if _, ok, _ := Build([]string{a}, Options{Output: filepath.Dir(out), Overwrite: true}); !ok {
		t.Fatal("expected valid with overwrite")
	}
}

func TestBadInputs(t *testing.T) {
	d := t.TempDir()
	png := touch(t, filepath.Join(d, "a.png"))
	items, ok, _ := Build([]string{png, filepath.Join(d, "missing.jpg")}, Options{Output: filepath.Join(d, "out")})
	if ok {
		t.Fatal("expected invalid")
	}
	eq(t, codes(items), []string{"unsupported_format", "input_not_found"})
}

func TestSingleFileExtensionChecks(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "a.jpg"))
	items, _, _ := Build([]string{a}, Options{Output: filepath.Join(d, "out.png")})
	eq(t, codes(items), []string{"unsupported_format"})
	items, _, _ = Build([]string{a}, Options{Output: filepath.Join(d, "out.jpg"), Format: "tiff"})
	eq(t, codes(items), []string{"unsupported_format"})
}

func TestMultipleInputsToExistingFileIsAnError(t *testing.T) {
	d := t.TempDir()
	a := touch(t, filepath.Join(d, "a.jpg"))
	b := touch(t, filepath.Join(d, "b.jpg"))
	f := touch(t, filepath.Join(d, "f.jpg"))
	if _, _, err := Build([]string{a, b}, Options{Output: f}); err == nil {
		t.Fatal("expected error")
	}
}
