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

	"github.com/SatyaChamana/FinSight/internal/inference"
	"github.com/SatyaChamana/FinSight/internal/model"
	"github.com/SatyaChamana/FinSight/internal/server"
	"github.com/SatyaChamana/FinSight/internal/service"
	"github.com/SatyaChamana/FinSight/internal/store"
)

const (
	defaultPort      = "9090"
	defaultLogLevel  = "info"
	defaultEnv       = "dev"
	defaultModelPath = "ml/models/artifacts/model.onnx"
	modelArch        = "lstm+attention"
	shutdownTimeout  = 10 * time.Second
)

type config struct {
	Port        string
	LogLevel    slog.Level
	Env         string
	DatabaseURL string
	ModelPath   string
}

func loadConfig() config {
	return config{
		Port:        getEnv("PORT", defaultPort),
		LogLevel:    parseLogLevel(getEnv("LOG_LEVEL", defaultLogLevel)),
		Env:         getEnv("ENV", defaultEnv),
		DatabaseURL: getEnv("DATABASE_URL", ""),
		ModelPath:   getEnv("MODEL_PATH", defaultModelPath),
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

	// Load the ONNX model and build the prediction service. If the model
	// (or its runtime) cannot be loaded, log a warning and continue: the
	// server still serves with dummy PredictRisk values so local dev and
	// machines without the ONNX Runtime shared library are not blocked.
	var predictor server.Predictor
	modelVer := ""
	engine, err := model.NewEngine(ctx, model.Config{ModelPath: cfg.ModelPath})
	if err != nil {
		logger.Warn("model not loaded; PredictRisk will return dummy values",
			"model_path", cfg.ModelPath,
			"err", err,
		)
	} else {
		defer func() {
			if closeErr := engine.Close(); closeErr != nil {
				logger.Warn("model engine close failed", "err", closeErr)
			}
		}()
		modelVer = engine.Version()
		// Real predictions need a portfolio store to read composition
		// from. Without a DB, fall back to dummy PredictRisk but still
		// report the loaded model version via GetModelInfo.
		if portfolios != nil {
			predictor = service.NewRiskService(service.Options{
				Portfolios: portfolios,
				Scorer:     inference.NewScorer(engine),
				Features:   inference.NewFeatureBuilder(),
				Logger:     logger,
			})
			logger.Info("model loaded; serving real predictions",
				"version", modelVer, "model_path", cfg.ModelPath)
		} else {
			logger.Warn("model loaded but DATABASE_URL is unset; PredictRisk returns dummy values until a portfolio store is configured",
				"version", modelVer, "model_path", cfg.ModelPath)
		}
	}

	addr := ":" + cfg.Port
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	srv := server.New(server.Options{
		Logger:            logger,
		Portfolios:        portfolios,
		Predictor:         predictor,
		ModelVersion:      modelVer,
		ModelArchitecture: modelArch,
		EnableReflection:  cfg.Env != "prod",
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
