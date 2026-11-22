package api

import "testing"

func TestNormalizeFingerprint(t *testing.T) {
	want := "SHA256:abcd01"
	for _, in := range []string{"SHA256:ABCD01", "sha256:abcd01", "ab:cd:01", " abcd01 "} {
		if got := NormalizeFingerprint(in); got != want {
			t.Errorf("%q -> %q", in, got)
		}
	}
}
