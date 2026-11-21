package api

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// FetchCA downloads the server's private CA certificate without verifying the
// TLS connection. The caller must check the fingerprint against the one the
// operator printed with `painite-server admin fingerprint` before trusting it.
func FetchCA(ctx context.Context, endpoint string) (pemBytes []byte, fingerprint string, err error) {
	if !strings.HasPrefix(endpoint, "https://") {
		return nil, "", fmt.Errorf("endpoint must use https:// to pin a CA")
	}
	tr := &http.Transport{
		TLSClientConfig:       &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // fingerprint is checked by the caller
		ResponseHeaderTimeout: 30 * time.Second,
	}
	c := &http.Client{Transport: tr, Timeout: 60 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(endpoint, "/")+"/v1/ca.pem", nil)
	if err != nil {
		return nil, "", err
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("fetching CA: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("fetching CA: %w", decodeError(resp))
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if err != nil {
		return nil, "", err
	}
	// Keep only the first certificate: anything appended after it would be
	// trusted too without being covered by the fingerprint.
	block, _ := pem.Decode(data)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, "", fmt.Errorf("server sent no PEM certificate")
	}
	only := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: block.Bytes})
	fp, err := Fingerprint(only)
	return only, fp, err
}

// Fingerprint returns "SHA256:<hex>" of the first certificate in a PEM blob.
func Fingerprint(pemBytes []byte) (string, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil || block.Type != "CERTIFICATE" {
		return "", fmt.Errorf("no PEM certificate found")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return "", err
	}
	if !cert.IsCA {
		return "", fmt.Errorf("certificate is not a CA")
	}
	sum := sha256.Sum256(cert.Raw)
	return "SHA256:" + hex.EncodeToString(sum[:]), nil
}

// NormalizeFingerprint accepts "SHA256:ab12…", "ab:12:…", any case.
func NormalizeFingerprint(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimPrefix(s, "sha256:")
	s = strings.ReplaceAll(s, ":", "")
	return "SHA256:" + s
}
