package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/SatyaChamana/FinSight/internal/server"
	"github.com/SatyaChamana/FinSight/internal/store"
)

const (
	defaultPort     = "9090"
	defaultLogLevel = "info"
	defaultEnv      = "dev"
	shutdownTimeout = 10 * time.Second
)

type config struct {
	Port        string
	LogLevel    slog.Level
	Env         string
	DatabaseURL string
}

func loadConfig() config {
	return config{
		Port:        getEnv("PORT", defaultPort),
		LogLevel:    parseLogLevel(getEnv("LOG_LEVEL", defaultLogLevel)),
		Env:         getEnv("ENV", defaultEnv),
		DatabaseURL: getEnv("DATABASE_URL", ""),
	}
}

func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func parseLogLevel(s string) slog.Level {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

func newLogger(cfg config) *slog.Logger {
	opts := &slog.HandlerOptions{Level: cfg.LogLevel}
	var handler slog.Handler
	if cfg.Env == "dev" {
		handler = slog.NewTextHandler(os.Stdout, opts)
	} else {
		handler = slog.NewJSONHandler(os.Stdout, opts)
	}
	return slog.New(handler).With("service", "finsight", "env", cfg.Env)
}

// run owns the listener, optional db pool, and the gRPC server lifecycle.
// It returns when ctx is cancelled (graceful shutdown) or the server
// exits with an error.
func run(ctx context.Context, cfg config, logger *slog.Logger) error {
	var portfolios server.PortfolioReader
	if cfg.DatabaseURL != "" {
		connectCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		pool, err := store.NewPool(connectCtx, cfg.DatabaseURL)
		cancel()
		if err != nil {
			return err
		}
		defer pool.Close()
		portfolios = store.New(pool)
		logger.Info("database connected", "driver", "pgx")
	} else {
		logger.Warn("DATABASE_URL is unset; portfolio RPCs will return Unavailable")
	}

	addr := ":" + cfg.Port
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	srv := server.New(server.Options{
		Logger:           logger,
		Portfolios:       portfolios,
		EnableReflection: cfg.Env != "prod",
	})

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("grpc server starting",
			"addr", addr,
			"reflection", cfg.Env != "prod",
			"db", cfg.DatabaseURL != "",
		)
		if err := srv.Serve(lis); err != nil {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		logger.Info("shutdown signal received, draining connections", "timeout", shutdownTimeout)
		stopped := make(chan struct{})
		go func() {
			srv.GracefulStop()
			close(stopped)
		}()
		select {
		case <-stopped:
			logger.Info("grpc server stopped cleanly")
			return nil
		case <-time.After(shutdownTimeout):
			logger.Warn("graceful stop deadline exceeded, forcing stop")
			srv.Stop()
			return errors.New("graceful shutdown timed out")
		}
	}
}

func main() {
	os.Exit(mainExit())
}

func mainExit() int {
	cfg := loadConfig()
	logger := newLogger(cfg)
	slog.SetDefault(logger)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, cfg, logger); err != nil {
		logger.Error("server exited with error", "err", err)
		return 1
	}
	return 0
}
