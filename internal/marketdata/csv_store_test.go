package marketdata

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func writeCSV(t *testing.T, dir, symbol, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, symbol+".csv"), []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}

func TestCSVBarStore_Bars(t *testing.T) {
	dir := t.TempDir()
	writeCSV(t, dir, "TEST",
		"date,open,high,low,close,adj_close,volume\n"+
			"2020-01-02,10.0,11.0,9.5,10.5,10.4,1000\n"+
			"2020-01-03,10.5,12.0,10.2,11.8,11.7,2000\n",
	)

	s := NewCSVBarStore(dir)
	bars, err := s.Bars(context.Background(), "TEST")
	if err != nil {
		t.Fatalf("Bars: %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("got %d bars, want 2", len(bars))
	}
	got := bars[0]
	if got.Open != 10.0 || got.High != 11.0 || got.Low != 9.5 || got.Close != 10.5 || got.AdjClose != 10.4 || got.Volume != 1000 {
		t.Errorf("bar[0] mismatch: %+v", got)
	}
	if bars[1].Close != 11.8 {
		t.Errorf("bar[1].Close: got %v, want 11.8", bars[1].Close)
	}
}

func TestCSVBarStore_Caches(t *testing.T) {
	dir := t.TempDir()
	writeCSV(t, dir, "TEST", "date,open,high,low,close,adj_close,volume\n2020-01-02,1,1,1,1,1,1\n")

	s := NewCSVBarStore(dir)
	if _, err := s.Bars(context.Background(), "TEST"); err != nil {
		t.Fatalf("first read: %v", err)
	}
	// Remove the file; a cached read must still succeed.
	if err := os.Remove(filepath.Join(dir, "TEST.csv")); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, err := s.Bars(context.Background(), "TEST"); err != nil {
		t.Errorf("cached read failed after file removal: %v", err)
	}
}

func TestCSVBarStore_Errors(t *testing.T) {
	dir := t.TempDir()

	s := NewCSVBarStore(dir)
	if _, err := s.Bars(context.Background(), "MISSING"); err == nil {
		t.Error("want error for missing file, got nil")
	}

	writeCSV(t, dir, "HEADERONLY", "date,open,high,low,close,adj_close,volume\n")
	if _, err := s.Bars(context.Background(), "HEADERONLY"); err == nil {
		t.Error("want error for header-only file, got nil")
	}

	writeCSV(t, dir, "BADNUM", "date,open,high,low,close,adj_close,volume\n2020-01-02,x,1,1,1,1,1\n")
	if _, err := s.Bars(context.Background(), "BADNUM"); err == nil {
		t.Error("want error for non-numeric field, got nil")
	}
}
