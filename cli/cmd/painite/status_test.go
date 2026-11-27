package main

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/yuanlin2004/PAINITE/cli/internal/api"
)

func TestPlaceholderOperations(t *testing.T) {
	var caps api.Capabilities
	err := json.Unmarshal([]byte(`{
		"operations": [{"name": "sharpen", "default_model": "sharpen/identity@0.1.0"},
		               {"name": "denoise", "default_model": "denoise/scunet@1.0.0"},
		               {"name": "face", "default_model": "face/identity@0.1.0"}],
		"models": [{"id": "sharpen/identity@0.1.0", "placeholder": true},
		           {"id": "denoise/identity@0.1.0", "placeholder": true},
		           {"id": "denoise/scunet@1.0.0"},
		           {"id": "face/identity@0.1.0", "placeholder": true}]}`), &caps)
	if err != nil {
		t.Fatal(err)
	}
	if got := placeholderOperations(&caps); !reflect.DeepEqual(got, []string{"face", "sharpen"}) {
		t.Fatalf("got %v", got)
	}
}
