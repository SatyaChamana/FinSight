// Package server wires together the gRPC server. It registers the
// RiskPredictionService implementation, the standard gRPC health
// check service and (in non-prod environments) gRPC reflection so
// grpcurl can introspect the schema.
//
// Phase 2 note: handlers return dummy values. Real business logic
// moves to internal/service in Phase 3 and the model layer in Phase 5.
package server

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	finsightv1 "github.com/SatyaChamana/FinSight/gen/go/finsight/v1"
	"github.com/SatyaChamana/FinSight/internal/service"
	"github.com/SatyaChamana/FinSight/internal/store"
	"github.com/SatyaChamana/FinSight/internal/tenant"
)

const (
	defaultLookbackDays    = 63
	defaultConfidenceLevel = 0.95
	dummyModelVersion      = "0.0.1-dev"
	dummyModelArchitecture = "lstm+attention"

	finsightMethodPrefix = "/finsight.v1."
)

// PortfolioReader is the read-side of the portfolios repository the
// gRPC handlers depend on. Defined here (where it is consumed) so the
// store package satisfies it via duck typing; the server package does
// not pin a concrete pgx-backed type.
type PortfolioReader interface {
	GetPortfolioWithAssets(ctx context.Context, portfolioID string) (*store.Portfolio, error)
}

// Predictor is the prediction orchestration the PredictRisk handler
// depends on. Defined here (where it is consumed); *service.RiskService
// satisfies it. When nil, PredictRisk falls back to dummy dev values so
// the server still runs without a loaded model.
type Predictor interface {
	PredictRisk(ctx context.Context, portfolioID string, lookbackDays int, confidenceLevel float64) (*service.Prediction, error)
}

// RiskService implements finsightv1.RiskPredictionServiceServer.
// It embeds the unimplemented base so forward-compatible RPCs added
// to the proto do not break the build until they are implemented.
type RiskService struct {
	finsightv1.UnimplementedRiskPredictionServiceServer
	logger     *slog.Logger
	portfolios PortfolioReader
	predictor  Predictor
	modelVer   string
	modelArch  string
	now        func() time.Time
}

// NewRiskService constructs a RiskService. The portfolio reader may be
// nil in unit tests that exercise PredictRisk only; GetPortfolio will
// then return Unavailable. The predictor may be nil; PredictRisk then
// returns dummy dev values. Empty modelVer/modelArch fall back to dummy
// metadata in GetModelInfo.
func NewRiskService(logger *slog.Logger, portfolios PortfolioReader, predictor Predictor, modelVer, modelArch string) *RiskService {
	if modelVer == "" {
		modelVer = dummyModelVersion
	}
	if modelArch == "" {
		modelArch = dummyModelArchitecture
	}
	return &RiskService{
		logger:     logger,
		portfolios: portfolios,
		predictor:  predictor,
		modelVer:   modelVer,
		modelArch:  modelArch,
		now:        time.Now,
	}
}

// PredictRisk serves a risk prediction. When a predictor is wired it
// runs real ONNX inference through the service layer; otherwise it falls
// back to dummy dev values so the server is usable without a model.
// Tenant id is enforced by the interceptor and read here for logging.
func (s *RiskService) PredictRisk(ctx context.Context, req *finsightv1.PredictRiskRequest) (*finsightv1.PredictRiskResponse, error) {
	tenantID, err := tenant.FromContext(ctx)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}

	if s.predictor == nil {
		return s.predictRiskDummy(ctx, tenantID, req)
	}

	pred, err := s.predictor.PredictRisk(
		ctx,
		req.GetPortfolioId(),
		int(req.GetLookbackDays()),
		float64(req.GetConfidenceLevel()),
	)
	if err != nil {
		return nil, s.mapPredictError(ctx, err)
	}

	contribs := make(map[string]float32, len(pred.AssetContributions))
	for sym, v := range pred.AssetContributions {
		contribs[sym] = float32(v)
	}

	return &finsightv1.PredictRiskResponse{
		Var_95:             float32(pred.VaR95),
		Cvar_95:            float32(pred.CVaR95),
		Volatility:         float32(pred.Volatility),
		ModelVersion:       pred.ModelVersion,
		PredictedAt:        timestamppb.New(pred.PredictedAt),
		AssetContributions: contribs,
	}, nil
}

// mapPredictError translates service-layer sentinel errors into gRPC
// status codes. gRPC code mapping lives only in the server package.
func (s *RiskService) mapPredictError(ctx context.Context, err error) error {
	switch {
	case errors.Is(err, service.ErrInvalidPortfolioID):
		return status.Error(codes.InvalidArgument, "portfolio_id is required")
	case errors.Is(err, service.ErrInvalidConfidence):
		return status.Error(codes.InvalidArgument, err.Error())
	case errors.Is(err, service.ErrPortfolioNotFound):
		return status.Error(codes.NotFound, "portfolio not found")
	case errors.Is(err, tenant.ErrMissing):
		return status.Error(codes.Unauthenticated, err.Error())
	default:
		s.logger.ErrorContext(ctx, "PredictRisk failed", "err", err)
		return status.Error(codes.Internal, "prediction failed")
	}
}

// predictRiskDummy is the no-model dev fallback. It validates inputs the
// same way the service layer would and returns fixed values.
func (s *RiskService) predictRiskDummy(ctx context.Context, tenantID string, req *finsightv1.PredictRiskRequest) (*finsightv1.PredictRiskResponse, error) {
	if req.GetPortfolioId() == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}

	lookback := req.GetLookbackDays()
	if lookback <= 0 {
		lookback = defaultLookbackDays
	}
	confidence := req.GetConfidenceLevel()
	if confidence <= 0 {
		confidence = defaultConfidenceLevel
	}
	if confidence >= 1 {
		return nil, status.Errorf(codes.InvalidArgument, "confidence_level must be < 1, got %f", confidence)
	}

	s.logger.DebugContext(ctx, "PredictRisk (dummy, no model loaded)",
		"tenant_id", tenantID,
		"portfolio_id", req.GetPortfolioId(),
		"lookback_days", lookback,
		"confidence_level", confidence,
	)

	return &finsightv1.PredictRiskResponse{
		Var_95:       0.0250,
		Cvar_95:      0.0420,
		Volatility:   0.1850,
		ModelVersion: s.modelVer,
		PredictedAt:  timestamppb.New(s.now()),
		AssetContributions: map[string]float32{
			"AAPL": 0.40,
			"MSFT": 0.35,
			"GOOG": 0.25,
		},
	}, nil
}

// GetPortfolio loads a portfolio (and its assets) from the store,
// scoped to the tenant on the incoming context. Cross-tenant lookups
// are filtered by Postgres RLS and surface as NotFound.
func (s *RiskService) GetPortfolio(ctx context.Context, req *finsightv1.GetPortfolioRequest) (*finsightv1.GetPortfolioResponse, error) {
	if req.GetPortfolioId() == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}
	if s.portfolios == nil {
		return nil, status.Error(codes.Unavailable, "portfolio store not configured")
	}

	p, err := s.portfolios.GetPortfolioWithAssets(ctx, req.GetPortfolioId())
	if err != nil {
		switch {
		case errors.Is(err, tenant.ErrMissing):
			return nil, status.Error(codes.Unauthenticated, err.Error())
		case errors.Is(err, store.ErrNotFound):
			return nil, status.Error(codes.NotFound, "portfolio not found")
		default:
			s.logger.ErrorContext(ctx, "GetPortfolio store call failed", "err", err)
			return nil, status.Error(codes.Internal, "lookup failed")
		}
	}

	assets := make([]*finsightv1.Asset, 0, len(p.Assets))
	for _, a := range p.Assets {
		assets = append(assets, &finsightv1.Asset{
			Symbol: a.Symbol,
			Weight: float32(a.Weight),
		})
	}

	return &finsightv1.GetPortfolioResponse{
		PortfolioId: p.ID,
		Name:        p.Name,
		Assets:      assets,
		UpdatedAt:   timestamppb.New(p.UpdatedAt),
	}, nil
}

// GetModelInfo returns dummy model metadata. No tenant scoping; the
// model is shared across all tenants.
func (s *RiskService) GetModelInfo(_ context.Context, _ *finsightv1.GetModelInfoRequest) (*finsightv1.GetModelInfoResponse, error) {
	return &finsightv1.GetModelInfoResponse{
		Version:      s.modelVer,
		Architecture: s.modelArch,
		LoadedAt:     timestamppb.New(s.now()),
		Sha256:       "0000000000000000000000000000000000000000000000000000000000000000",
	}, nil
}

// Options configures the gRPC server.
type Options struct {
	Logger            *slog.Logger
	Portfolios        PortfolioReader
	Predictor         Predictor
	ModelVersion      string
	ModelArchitecture string
	EnableReflection  bool
}

// requiresTenant returns true for RPCs that must carry an
// x-tenant-id metadata header. Health and reflection are exempt.
func requiresTenant(fullMethod string) bool {
	return strings.HasPrefix(fullMethod, finsightMethodPrefix)
}

// selectiveTenantUnary applies tenant.UnaryServerInterceptor only to
// finsight.v1.* methods, so health checks and reflection do not need
// tenant metadata.
func selectiveTenantUnary() grpc.UnaryServerInterceptor {
	base := tenant.UnaryServerInterceptor()
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if requiresTenant(info.FullMethod) {
			return base(ctx, req, info, handler)
		}
		return handler(ctx, req)
	}
}

// selectiveTenantStream mirrors selectiveTenantUnary for streams.
func selectiveTenantStream() grpc.StreamServerInterceptor {
	base := tenant.StreamServerInterceptor()
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if requiresTenant(info.FullMethod) {
			return base(srv, ss, info, handler)
		}
		return handler(srv, ss)
	}
}

// New constructs a *grpc.Server with the RiskPredictionService,
// gRPC health check and (optionally) reflection registered. The
// tenant interceptor is chained for finsight.v1.* methods.
func New(opts Options) *grpc.Server {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	srv := grpc.NewServer(
		grpc.UnaryInterceptor(selectiveTenantUnary()),
		grpc.StreamInterceptor(selectiveTenantStream()),
	)
	finsightv1.RegisterRiskPredictionServiceServer(srv, NewRiskService(logger, opts.Portfolios, opts.Predictor, opts.ModelVersion, opts.ModelArchitecture))

	hc := health.NewServer()
	hc.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	hc.SetServingStatus("finsight.v1.RiskPredictionService", healthpb.HealthCheckResponse_SERVING)
	healthpb.RegisterHealthServer(srv, hc)

	if opts.EnableReflection {
		reflection.Register(srv)
	}

	return srv
}
