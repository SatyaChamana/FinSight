package main

import (
	"log/slog"
	"strings"
	"testing"
)

func TestParseLogLevel(t *testing.T) {
	cases := map[string]slog.Level{
		"debug":   slog.LevelDebug,
		"DEBUG":   slog.LevelDebug,
		"info":    slog.LevelInfo,
		"warn":    slog.LevelWarn,
		"warning": slog.LevelWarn,
		"error":   slog.LevelError,
		"":        slog.LevelInfo,
		"bogus":   slog.LevelInfo,
	}
	for input, want := range cases {
		t.Run(strings.ToLower(input), func(t *testing.T) {
			if got := parseLogLevel(input); got != want {
				t.Fatalf("parseLogLevel(%q): got %v, want %v", input, got, want)
			}
		})
	}
}

func TestGetEnv(t *testing.T) {
	const key = "FINSIGHT_TEST_VAR"
	t.Setenv(key, "set-value")
	if got := getEnv(key, "fallback"); got != "set-value" {
		t.Fatalf("set var: got %q, want %q", got, "set-value")
	}

	t.Setenv(key, "")
	if got := getEnv(key, "fallback"); got != "fallback" {
		t.Fatalf("empty var: got %q, want fallback", got)
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	t.Setenv("PORT", "")
	t.Setenv("LOG_LEVEL", "")
	t.Setenv("ENV", "")

	cfg := loadConfig()
	if cfg.Port != defaultPort {
		t.Errorf("Port: got %q, want %q", cfg.Port, defaultPort)
	}
	if cfg.LogLevel != slog.LevelInfo {
		t.Errorf("LogLevel: got %v, want Info", cfg.LogLevel)
	}
	if cfg.Env != defaultEnv {
		t.Errorf("Env: got %q, want %q", cfg.Env, defaultEnv)
	}
}
