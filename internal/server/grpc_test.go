package server_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	finsightv1 "github.com/SatyaChamana/FinSight/gen/go/finsight/v1"
	"github.com/SatyaChamana/FinSight/internal/server"
	"github.com/SatyaChamana/FinSight/internal/service"
	"github.com/SatyaChamana/FinSight/internal/store"
)

const (
	bufSize        = 1024 * 1024
	testTenantID   = "tenant-test"
	testTenantHdr  = "x-tenant-id"
	demoPortfolio  = "p-1"
	demoTenantName = "Demo Tech"
)

// fakePortfolioReader is a hand-rolled in-memory implementation of
// server.PortfolioReader for unit tests.
type fakePortfolioReader struct {
	portfolios map[string]*store.Portfolio
}

func (f *fakePortfolioReader) GetPortfolioWithAssets(_ context.Context, portfolioID string) (*store.Portfolio, error) {
	p, ok := f.portfolios[portfolioID]
	if !ok {
		return nil, store.ErrNotFound
	}
	return p, nil
}

// fakePredictor is a hand-rolled server.Predictor for testing the real
// (non-dummy) PredictRisk delegation path and error mapping.
type fakePredictor struct {
	pred *service.Prediction
	err  error
}

func (f *fakePredictor) PredictRisk(_ context.Context, _ string, _ int, _ float64) (*service.Prediction, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.pred, nil
}

// newTestServer spins up the gRPC server over an in-memory bufconn
// listener and returns a client connection. The server and connection
// are torn down by t.Cleanup.
func newTestServer(t *testing.T, reader server.PortfolioReader) *grpc.ClientConn {
	t.Helper()
	return newTestServerWith(t, server.Options{
		Portfolios:       reader,
		EnableReflection: false,
	})
}

// newTestServerWith spins up the server with caller-supplied Options,
// filling in a discard logger and a bufconn listener.
func newTestServerWith(t *testing.T, opts server.Options) *grpc.ClientConn {
	t.Helper()

	lis := bufconn.Listen(bufSize)
	opts.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	srv := server.New(opts)

	go func() {
		if err := srv.Serve(lis); err != nil {
			t.Logf("server.Serve returned: %v", err)
		}
	}()

	dialer := func(context.Context, string) (net.Conn, error) {
		return lis.DialContext(context.Background())
	}

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(dialer),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("grpc.NewClient: %v", err)
	}

	t.Cleanup(func() {
		_ = conn.Close()
		stopped := make(chan struct{})
		go func() {
			srv.GracefulStop()
			close(stopped)
		}()
		select {
		case <-stopped:
		case <-time.After(2 * time.Second):
			srv.Stop()
		}
	})

	return conn
}

// withTenant attaches the x-tenant-id outgoing metadata header.
func withTenant(ctx context.Context, id string) context.Context {
	return metadata.AppendToOutgoingContext(ctx, testTenantHdr, id)
}

func TestPredictRisk_ReturnsDummyValues(t *testing.T) {
	conn := newTestServer(t, nil)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	resp, err := client.PredictRisk(ctx, &finsightv1.PredictRiskRequest{
		PortfolioId:     "demo-1",
		LookbackDays:    63,
		ConfidenceLevel: 0.95,
	})
	if err != nil {
		t.Fatalf("PredictRisk: %v", err)
	}

	if resp.GetVar_95() <= 0 {
		t.Errorf("Var_95: got %f, want > 0", resp.GetVar_95())
	}
	if resp.GetCvar_95() < resp.GetVar_95() {
		t.Errorf("Cvar_95 (%f) should be >= Var_95 (%f)", resp.GetCvar_95(), resp.GetVar_95())
	}
	if resp.GetVolatility() <= 0 {
		t.Errorf("Volatility: got %f, want > 0", resp.GetVolatility())
	}
	if resp.GetModelVersion() == "" {
		t.Error("ModelVersion is empty")
	}
	if resp.GetPredictedAt() == nil {
		t.Error("PredictedAt is nil")
	}
	if len(resp.GetAssetContributions()) == 0 {
		t.Error("AssetContributions is empty")
	}
}

func TestPredictRisk_DelegatesToPredictor(t *testing.T) {
	want := &service.Prediction{
		VaR95:        0.031,
		CVaR95:       0.052,
		Volatility:   0.21,
		ModelVersion: "0.1.0",
		PredictedAt:  time.Unix(1_700_000_000, 0).UTC(),
		AssetContributions: map[string]float64{
			"AAPL": 0.018,
			"MSFT": 0.013,
		},
	}
	conn := newTestServerWith(t, server.Options{Predictor: &fakePredictor{pred: want}})
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	resp, err := client.PredictRisk(ctx, &finsightv1.PredictRiskRequest{PortfolioId: "p-1"})
	if err != nil {
		t.Fatalf("PredictRisk: %v", err)
	}
	if resp.GetVar_95() != float32(want.VaR95) {
		t.Errorf("Var_95: got %f, want %f", resp.GetVar_95(), float32(want.VaR95))
	}
	if resp.GetCvar_95() != float32(want.CVaR95) {
		t.Errorf("Cvar_95: got %f, want %f", resp.GetCvar_95(), float32(want.CVaR95))
	}
	if resp.GetVolatility() != float32(want.Volatility) {
		t.Errorf("Volatility: got %f, want %f", resp.GetVolatility(), float32(want.Volatility))
	}
	if resp.GetModelVersion() != want.ModelVersion {
		t.Errorf("ModelVersion: got %q, want %q", resp.GetModelVersion(), want.ModelVersion)
	}
	if got := resp.GetAssetContributions(); len(got) != 2 || got["AAPL"] != 0.018 {
		t.Errorf("AssetContributions: got %v", got)
	}
	if !resp.GetPredictedAt().AsTime().Equal(want.PredictedAt) {
		t.Errorf("PredictedAt: got %v, want %v", resp.GetPredictedAt().AsTime(), want.PredictedAt)
	}
}

func TestPredictRisk_MapsServiceErrors(t *testing.T) {
	cases := []struct {
		name    string
		err     error
		wantErr codes.Code
	}{
		{"invalid portfolio id", service.ErrInvalidPortfolioID, codes.InvalidArgument},
		{"invalid confidence", service.ErrInvalidConfidence, codes.InvalidArgument},
		{"portfolio not found", service.ErrPortfolioNotFound, codes.NotFound},
		{"unexpected error", errors.New("boom"), codes.Internal},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn := newTestServerWith(t, server.Options{Predictor: &fakePredictor{err: tc.err}})
			client := finsightv1.NewRiskPredictionServiceClient(conn)

			ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
			defer cancel()

			_, err := client.PredictRisk(ctx, &finsightv1.PredictRiskRequest{PortfolioId: "p-1"})
			st, ok := status.FromError(err)
			if !ok {
				t.Fatalf("error is not a gRPC status: %v", err)
			}
			if st.Code() != tc.wantErr {
				t.Errorf("code: got %v, want %v", st.Code(), tc.wantErr)
			}
		})
	}
}

func TestPredictRisk_ValidatesInput(t *testing.T) {
	conn := newTestServer(t, nil)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	cases := []struct {
		name    string
		req     *finsightv1.PredictRiskRequest
		wantErr codes.Code
	}{
		{
			name:    "missing portfolio_id",
			req:     &finsightv1.PredictRiskRequest{},
			wantErr: codes.InvalidArgument,
		},
		{
			name:    "confidence_level >= 1",
			req:     &finsightv1.PredictRiskRequest{PortfolioId: "p", ConfidenceLevel: 1.5},
			wantErr: codes.InvalidArgument,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
			defer cancel()
			_, err := client.PredictRisk(ctx, tc.req)
			if err == nil {
				t.Fatalf("want error %v, got nil", tc.wantErr)
			}
			st, ok := status.FromError(err)
			if !ok {
				t.Fatalf("error is not a gRPC status: %v", err)
			}
			if st.Code() != tc.wantErr {
				t.Errorf("code: got %v, want %v (msg: %s)", st.Code(), tc.wantErr, st.Message())
			}
		})
	}
}

func TestPredictRisk_RequiresTenantMetadata(t *testing.T) {
	conn := newTestServer(t, nil)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	// No tenant metadata attached on the outgoing context.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := client.PredictRisk(ctx, &finsightv1.PredictRiskRequest{PortfolioId: "p"})
	if err == nil {
		t.Fatal("want error, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Errorf("code: got %v, want Unauthenticated", st.Code())
	}
}

func TestGetPortfolio_FromStore(t *testing.T) {
	reader := &fakePortfolioReader{
		portfolios: map[string]*store.Portfolio{
			demoPortfolio: {
				ID:       demoPortfolio,
				TenantID: testTenantID,
				Name:     demoTenantName,
				Assets: []store.Asset{
					{Symbol: "AAPL", Weight: 0.40},
					{Symbol: "MSFT", Weight: 0.35},
					{Symbol: "GOOG", Weight: 0.25},
				},
				UpdatedAt: time.Now(),
			},
		},
	}
	conn := newTestServer(t, reader)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	resp, err := client.GetPortfolio(ctx, &finsightv1.GetPortfolioRequest{PortfolioId: demoPortfolio})
	if err != nil {
		t.Fatalf("GetPortfolio: %v", err)
	}
	if resp.GetPortfolioId() != demoPortfolio {
		t.Errorf("PortfolioId: got %q, want %q", resp.GetPortfolioId(), demoPortfolio)
	}
	if len(resp.GetAssets()) != 3 {
		t.Fatalf("Assets: got %d, want 3", len(resp.GetAssets()))
	}
	var sum float32
	for _, a := range resp.GetAssets() {
		sum += a.GetWeight()
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("weights sum to %f, want ~1.0", sum)
	}
}

func TestGetPortfolio_NotFound(t *testing.T) {
	reader := &fakePortfolioReader{portfolios: map[string]*store.Portfolio{}}
	conn := newTestServer(t, reader)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	_, err := client.GetPortfolio(ctx, &finsightv1.GetPortfolioRequest{PortfolioId: "missing"})
	st, _ := status.FromError(err)
	if st.Code() != codes.NotFound {
		t.Errorf("code: got %v, want NotFound", st.Code())
	}
}

func TestGetPortfolio_StoreNotConfigured(t *testing.T) {
	conn := newTestServer(t, nil)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	_, err := client.GetPortfolio(ctx, &finsightv1.GetPortfolioRequest{PortfolioId: demoPortfolio})
	st, _ := status.FromError(err)
	if st.Code() != codes.Unavailable {
		t.Errorf("code: got %v, want Unavailable", st.Code())
	}
}

func TestGetModelInfo(t *testing.T) {
	conn := newTestServer(t, nil)
	client := finsightv1.NewRiskPredictionServiceClient(conn)

	ctx, cancel := context.WithTimeout(withTenant(context.Background(), testTenantID), 2*time.Second)
	defer cancel()

	resp, err := client.GetModelInfo(ctx, &finsightv1.GetModelInfoRequest{})
	if err != nil {
		t.Fatalf("GetModelInfo: %v", err)
	}
	if resp.GetVersion() == "" {
		t.Error("Version is empty")
	}
	if resp.GetArchitecture() == "" {
		t.Error("Architecture is empty")
	}
}

func TestHealthCheck_NoTenantRequired(t *testing.T) {
	conn := newTestServer(t, nil)
	client := healthpb.NewHealthClient(conn)

	// Health check must NOT require tenant metadata.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	resp, err := client.Check(ctx, &healthpb.HealthCheckRequest{Service: ""})
	if err != nil {
		t.Fatalf("Health.Check: %v", err)
	}
	if resp.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("overall status: got %v, want SERVING", resp.GetStatus())
	}
}
