package main

import "testing"

func TestParseSteps(t *testing.T) {
	half := 0.5
	steps, err := parseSteps("denoise:0.6, sharpen,upscale2x", &half)
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 3 || steps[0].Operation != "denoise" || *steps[0].Strength != 0.6 ||
		*steps[1].Strength != 0.5 || steps[2].Strength != nil {
		t.Fatalf("got %+v", steps)
	}
	one, err := parseSteps("sharpen", nil)
	if err != nil || len(one) != 1 || one[0].Strength != nil {
		t.Fatalf("got %+v, %v", one, err)
	}
	for _, bad := range []string{"", "blur", "upscale2x:0.5", "denoise:2", "denoise:x", "upscale2x,upscale2x",
		"denoise,denoise,denoise,denoise,denoise"} {
		if _, err := parseSteps(bad, nil); err == nil {
			t.Errorf("%q: expected an error", bad)
		}
	}
	ub, err := parseSteps("unblur:0.8,denoise", nil)
	if err != nil || len(ub) != 2 || ub[0].Operation != "unblur" || *ub[0].Strength != 0.8 {
		t.Fatalf("got %+v, %v", ub, err)
	}
	if _, err := parseSteps("upscale2x", &half); err == nil {
		t.Error("--strength with only upscale2x should fail")
	}
}
