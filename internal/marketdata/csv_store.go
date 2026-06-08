// Package marketdata provides access to historical OHLCV bars that feed the
// feature pipeline. The model needs enough daily history per symbol (roughly
// 130+ bars) to produce a finite 63-day feature window after warm-up.
//
// This is the price-history data source the feature builder reads. The CSV
// implementation here is a simple, dependency-free store for development and
// the committed sample dataset; a Postgres-backed BarReader can replace it
// later without touching the feature builder (it depends on the interface).
package marketdata

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"

	"github.com/SatyaChamana/FinSight/internal/model"
)

// BarReader returns the daily OHLCV history for a symbol, oldest bar first.
type BarReader interface {
	Bars(ctx context.Context, symbol string) ([]model.OHLCV, error)
}

// CSVBarStore reads per-symbol OHLCV from {dir}/{SYMBOL}.csv files whose header
// is exactly: date,open,high,low,close,adj_close,volume. Parsed bars are cached
// in memory after the first read, guarded by a mutex for concurrent use.
type CSVBarStore struct {
	dir   string
	mu    sync.Mutex
	cache map[string][]model.OHLCV
}

// compile-time check that *CSVBarStore satisfies BarReader.
var _ BarReader = (*CSVBarStore)(nil)

// NewCSVBarStore constructs a store reading from dir.
func NewCSVBarStore(dir string) *CSVBarStore {
	return &CSVBarStore{dir: dir, cache: make(map[string][]model.OHLCV)}
}

// Bars returns the cached bars for symbol, parsing the CSV on first access.
func (s *CSVBarStore) Bars(_ context.Context, symbol string) ([]model.OHLCV, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if bars, ok := s.cache[symbol]; ok {
		return bars, nil
	}

	path := filepath.Join(s.dir, symbol+".csv")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open bars for %q: %w", symbol, err)
	}
	defer func() { _ = f.Close() }()

	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read bars for %q: %w", symbol, err)
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("bars for %q: empty or header-only file", symbol)
	}

	bars := make([]model.OHLCV, 0, len(records)-1)
	for i, rec := range records[1:] {
		if len(rec) < 7 {
			return nil, fmt.Errorf("bars for %q line %d: want 7 columns, got %d", symbol, i+2, len(rec))
		}
		bar, err := parseBar(rec)
		if err != nil {
			return nil, fmt.Errorf("bars for %q line %d: %w", symbol, i+2, err)
		}
		bars = append(bars, bar)
	}

	s.cache[symbol] = bars
	return bars, nil
}

// parseBar parses one CSV record. Columns: date,open,high,low,close,adj_close,
// volume. The date is not needed (bars are positionally ordered oldest-first),
// so it is not parsed.
func parseBar(rec []string) (model.OHLCV, error) {
	val := func(idx int, name string) (float64, error) {
		v, err := strconv.ParseFloat(rec[idx], 64)
		if err != nil {
			return 0, fmt.Errorf("parse %s %q: %w", name, rec[idx], err)
		}
		return v, nil
	}
	open, err := val(1, "open")
	if err != nil {
		return model.OHLCV{}, err
	}
	high, err := val(2, "high")
	if err != nil {
		return model.OHLCV{}, err
	}
	low, err := val(3, "low")
	if err != nil {
		return model.OHLCV{}, err
	}
	closePx, err := val(4, "close")
	if err != nil {
		return model.OHLCV{}, err
	}
	adj, err := val(5, "adj_close")
	if err != nil {
		return model.OHLCV{}, err
	}
	vol, err := val(6, "volume")
	if err != nil {
		return model.OHLCV{}, err
	}
	return model.OHLCV{Open: open, High: high, Low: low, Close: closePx, AdjClose: adj, Volume: vol}, nil
}
