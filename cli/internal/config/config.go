// Package config loads and saves the CLI's endpoint and credentials.
//
// Precedence, highest first: command-line flags, PAINITE_* environment
// variables, the config file (PAINITE_CONFIG, or <user config dir>/painite/config.json).
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Config is what the CLI needs to reach the server.
type Config struct {
	Endpoint string `json:"endpoint,omitempty"`
	Token    string `json:"token,omitempty"`
	CAFile   string `json:"ca_file,omitempty"`
}

// Keys lists the settable keys and what they mean.
var Keys = map[string]string{
	"endpoint": "server URL, for example https://spark1.local:8443",
	"token":    "client token from `painite-server token create`",
	"ca_file":  "PEM file of the CA that signed the server certificate",
}

// Path returns the config file path.
func Path() (string, error) {
	if p := os.Getenv("PAINITE_CONFIG"); p != "" {
		return p, nil
	}
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "painite", "config.json"), nil
}

// LoadFile reads the config file. A missing file is an empty config.
func LoadFile() (Config, error) {
	var c Config
	p, err := Path()
	if err != nil {
		return c, err
	}
	data, err := os.ReadFile(p)
	if errors.Is(err, fs.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(data, &c); err != nil {
		return c, fmt.Errorf("%s: %w", p, err)
	}
	return c, nil
}

// Load reads the file and applies environment overrides.
func Load() (Config, error) {
	c, err := LoadFile()
	if err != nil {
		return c, err
	}
	if v := os.Getenv("PAINITE_ENDPOINT"); v != "" {
		c.Endpoint = v
	}
	if v := os.Getenv("PAINITE_TOKEN"); v != "" {
		c.Token = v
	}
	if v := os.Getenv("PAINITE_CA_FILE"); v != "" {
		c.CAFile = v
	}
	return c, nil
}

// Save writes the config file with owner-only permissions.
func Save(c Config) error {
	p, err := Path()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

// Get returns one key's value.
func (c Config) Get(key string) (string, error) {
	switch key {
	case "endpoint":
		return c.Endpoint, nil
	case "token":
		return c.Token, nil
	case "ca_file":
		return c.CAFile, nil
	}
	return "", unknownKey(key)
}

// Set updates one key.
func (c *Config) Set(key, value string) error {
	switch key {
	case "endpoint":
		value = strings.TrimRight(value, "/")
		if !strings.HasPrefix(value, "http://") && !strings.HasPrefix(value, "https://") {
			return fmt.Errorf("endpoint must start with https:// or http://")
		}
		c.Endpoint = value
	case "token":
		c.Token = value
	case "ca_file":
		if value != "" {
			abs, err := filepath.Abs(value)
			if err != nil {
				return err
			}
			value = abs
		}
		c.CAFile = value
	default:
		return unknownKey(key)
	}
	return nil
}

func unknownKey(key string) error {
	names := make([]string, 0, len(Keys))
	for k := range Keys {
		names = append(names, k)
	}
	sort.Strings(names)
	return fmt.Errorf("unknown key %q (keys: %s)", key, strings.Join(names, ", "))
}

// MaskedToken shows only the start of the token.
func (c Config) MaskedToken() string {
	if c.Token == "" {
		return ""
	}
	if len(c.Token) <= 8 {
		return "****"
	}
	return c.Token[:8] + "…"
}
