package model

import (
	"fmt"
	"math"
	"sort"
)

// This file mirrors the Python feature pipeline so that Go inference sees the
// exact same numbers the model was trained on. Each helper names the Python
// source function it mirrors. There is no scaling step (the manifest's
// scaler_path is null), so features are fed to the model RAW.
//
// Python references:
//   - ml/data/features.py            (per-symbol features)
//   - ml/data/portfolio_features.py  (portfolio features)
//   - ml/data/dataset.py             (column layout: per-symbol blocks then portfolio)

// annualization is sqrt(252), matching ANNUALIZATION in the Python modules.
var annualization = math.Sqrt(252.0)

// Per-symbol feature counts and the portfolio feature count, fixed by the model.
const (
	perSymbolFeatures   = 9 // log_return, rolling_vol_5/21/63, ewma_vol_21, garman_klass_vol, rsi_14, atr_14, bb_width_20
	portfolioFeatures   = 4 // portfolio_return, portfolio_vol_21, portfolio_concentration, portfolio_avg_correlation
	expectedSymbolCount = 5 // 5 symbols * 9 + 4 = 49 == featureCols

	rsiPeriod         = 14
	atrPeriod         = 14
	bbWindow          = 20
	ewmaHalflife      = 21
	correlationWindow = 63
	portfolioVolWin   = 21
)

// OHLCV is a single daily bar. AdjClose is the dividend/split-adjusted close used
// for return calculations (Python uses adj_close for log_return). The non-adjusted
// Close/Open/High/Low are used by Garman-Klass, RSI, ATR, and Bollinger width,
// mirroring features.py exactly.
type OHLCV struct {
	Open     float64
	High     float64
	Low      float64
	Close    float64
	AdjClose float64
	Volume   float64
}

// symbolFeatures holds the 9 per-symbol feature series for one symbol, each
// aligned to the symbol's bar dates. NaN marks warm-up rows that the Python
// pipeline drops.
type symbolFeatures struct {
	logReturn    []float64
	rollingVol5  []float64
	rollingVol21 []float64
	rollingVol63 []float64
	ewmaVol21    []float64
	garmanKlass  []float64
	rsi14        []float64
	atr14        []float64
	bbWidth20    []float64
}

// BuildFeatureMatrix produces the [63][49] feature matrix the model expects, in
// the exact column order:
//
//	[s0_f0..s0_f8, s1_f0..s1_f8, ..., s4_f0..s4_f8, p0, p1, p2, p3]
//
// where the symbol order is the sorted order of the keys in symbolBars (so the
// layout is deterministic regardless of map iteration order) and the per-symbol
// feature order matches features.FEATURE_COLUMNS. The portfolio features p0..p3
// match portfolio_features.PORTFOLIO_FEATURE_COLUMNS.
//
// It computes features over the full input series, drops warm-up rows where any
// feature is NaN (mirroring the Python dropna), aligns symbols and portfolio
// features on common dates by position, and returns the last 63 aligned rows.
//
// weights maps symbol -> portfolio weight and must contain exactly the symbols
// in symbolBars and sum to ~1.0 (matching _validate_weights in
// portfolio_features.py).
//
// NOTE: parity. The Python pipeline aligns by calendar date. Here we assume the
// caller passes each symbol's bars covering the same trading days in the same
// order (the standard case for a portfolio fetched together). We align by
// position after each symbol's own warm-up rows are dropped, which matches the
// Python result when all symbols share the same date axis.
func BuildFeatureMatrix(symbolBars map[string][]OHLCV, weights map[string]float64) ([][]float32, error) {
	return BuildFeatureMatrixOrdered(sortedKeys(symbolBars), symbolBars, weights)
}

// BuildFeatureMatrixOrdered is BuildFeatureMatrix with an explicit symbol
// column order. The per-symbol blocks are laid out in `symbols` order, which
// MUST match the order the model was trained with (the Python pipeline uses
// list(weights.keys()) insertion order, not alphabetical). Use this when the
// caller knows the training order; BuildFeatureMatrix sorts and is only safe
// when the model is order-insensitive.
func BuildFeatureMatrixOrdered(symbols []string, symbolBars map[string][]OHLCV, weights map[string]float64) ([][]float32, error) {
	if len(symbols) != expectedSymbolCount {
		return nil, fmt.Errorf("BuildFeatureMatrix expects %d symbols, got %d", expectedSymbolCount, len(symbols))
	}
	if len(symbolBars) != len(symbols) {
		return nil, fmt.Errorf("symbolBars has %d symbols, order list has %d", len(symbolBars), len(symbols))
	}
	for _, s := range symbols {
		if _, ok := symbolBars[s]; !ok {
			return nil, fmt.Errorf("order list references symbol %q not in symbolBars", s)
		}
	}
	if err := validateWeights(symbols, weights); err != nil {
		return nil, err
	}

	// Compute per-symbol features and find each symbol's first fully-valid row.
	perSymbol := make([]symbolFeatures, len(symbols))
	firstValid := make([]int, len(symbols))
	barCount := make([]int, len(symbols))
	for i, sym := range symbols {
		bars := symbolBars[sym]
		if len(bars) == 0 {
			return nil, fmt.Errorf("symbol %q has no bars", sym)
		}
		sf := computeSymbolFeatures(bars)
		perSymbol[i] = sf
		firstValid[i] = firstFullyValidIndex(sf)
		barCount[i] = len(bars)
		if firstValid[i] < 0 {
			return nil, fmt.Errorf("symbol %q has no fully-valid feature rows after warm-up", sym)
		}
	}

	// All symbols are assumed to share the same date axis (see NOTE: parity).
	// Verify equal lengths so positional alignment is sound.
	for i := 1; i < len(symbols); i++ {
		if barCount[i] != barCount[0] {
			return nil, fmt.Errorf(
				"symbol %q has %d bars but symbol %q has %d; all symbols must share the same date axis",
				symbols[i], barCount[i], symbols[0], barCount[0],
			)
		}
	}

	// Portfolio features need aligned per-symbol log returns. The Python pipeline
	// drops dates where any symbol return is NaN; positionally that is the max of
	// the per-symbol first-valid indices for log_return (index 1 is the first
	// finite log_return, since log_return needs 1 prior bar).
	totalRows := barCount[0]
	pf := computePortfolioFeatures(perSymbol, symbols, weights, totalRows)

	// The model's effective warm-up start is the latest point where every
	// per-symbol feature AND every portfolio feature is finite.
	start := 0
	for i := range symbols {
		if firstValid[i] > start {
			start = firstValid[i]
		}
	}
	if pf.firstValid > start {
		start = pf.firstValid
	}

	validRows := totalRows - start
	if validRows < lookbackDays {
		return nil, fmt.Errorf(
			"not enough valid rows after warm-up: have %d, need %d (provide more history)",
			validRows, lookbackDays,
		)
	}

	// Take the last 63 aligned rows.
	windowStart := totalRows - lookbackDays
	matrix := make([][]float32, lookbackDays)
	for r := 0; r < lookbackDays; r++ {
		row := make([]float32, featureCols)
		src := windowStart + r
		col := 0
		for i := range symbols {
			sf := perSymbol[i]
			row[col+0] = float32(sf.logReturn[src])
			row[col+1] = float32(sf.rollingVol5[src])
			row[col+2] = float32(sf.rollingVol21[src])
			row[col+3] = float32(sf.rollingVol63[src])
			row[col+4] = float32(sf.ewmaVol21[src])
			row[col+5] = float32(sf.garmanKlass[src])
			row[col+6] = float32(sf.rsi14[src])
			row[col+7] = float32(sf.atr14[src])
			row[col+8] = float32(sf.bbWidth20[src])
			col += perSymbolFeatures
		}
		row[col+0] = float32(pf.portfolioReturn[src])
		row[col+1] = float32(pf.portfolioVol21[src])
		row[col+2] = float32(pf.concentration)
		row[col+3] = float32(pf.avgCorrelation[src])
		matrix[r] = row
	}
	return matrix, nil
}

// sortedKeys returns the map keys in deterministic ascending order.
func sortedKeys(m map[string][]OHLCV) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// validateWeights mirrors _validate_weights in ml/data/portfolio_features.py:
// weights must be non-empty, cover every symbol, and sum to 1.0 within 1e-6.
func validateWeights(symbols []string, weights map[string]float64) error {
	if len(weights) == 0 {
		return fmt.Errorf("weights must be a non-empty map")
	}
	var sum float64
	for _, w := range weights {
		sum += w
	}
	const tol = 1e-6
	if math.Abs(sum-1.0) > tol {
		return fmt.Errorf("weights must sum to 1.0 within %g; got %g", tol, sum)
	}
	for _, sym := range symbols {
		if _, ok := weights[sym]; !ok {
			return fmt.Errorf("weights missing symbol %q", sym)
		}
	}
	return nil
}

// computeSymbolFeatures computes all 9 per-symbol features. Mirrors
// _features_for_symbol in ml/data/features.py.
func computeSymbolFeatures(bars []OHLCV) symbolFeatures {
	n := len(bars)
	open := make([]float64, n)
	high := make([]float64, n)
	low := make([]float64, n)
	close := make([]float64, n)
	adj := make([]float64, n)
	for i, b := range bars {
		open[i] = b.Open
		high[i] = b.High
		low[i] = b.Low
		close[i] = b.Close
		adj[i] = b.AdjClose
	}

	logRet := logReturn(adj)
	return symbolFeatures{
		logReturn:    logRet,
		rollingVol5:  rollingVol(logRet, 5),
		rollingVol21: rollingVol(logRet, 21),
		rollingVol63: rollingVol(logRet, 63),
		ewmaVol21:    ewmaVol(logRet, ewmaHalflife),
		garmanKlass:  garmanKlass(open, high, low, close),
		rsi14:        rsi(close, rsiPeriod),
		atr14:        atr(high, low, close, atrPeriod),
		bbWidth20:    bbWidth(close, bbWindow),
	}
}

// firstFullyValidIndex returns the first row index where every per-symbol
// feature is finite (mirrors the per-symbol dropna in features.py), or -1.
func firstFullyValidIndex(sf symbolFeatures) int {
	n := len(sf.logReturn)
	for i := 0; i < n; i++ {
		if isFinite(sf.logReturn[i]) &&
			isFinite(sf.rollingVol5[i]) &&
			isFinite(sf.rollingVol21[i]) &&
			isFinite(sf.rollingVol63[i]) &&
			isFinite(sf.ewmaVol21[i]) &&
			isFinite(sf.garmanKlass[i]) &&
			isFinite(sf.rsi14[i]) &&
			isFinite(sf.atr14[i]) &&
			isFinite(sf.bbWidth20[i]) {
			return i
		}
	}
	return -1
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// nan is a convenience for an unset / warm-up value.
func nan() float64 { return math.NaN() }

// logReturn mirrors features.py _log_return: ln(adj_close_t / adj_close_{t-1}).
// Index 0 is NaN (no prior bar).
func logReturn(adj []float64) []float64 {
	n := len(adj)
	out := make([]float64, n)
	out[0] = nan()
	for i := 1; i < n; i++ {
		out[i] = math.Log(adj[i] / adj[i-1])
	}
	return out
}

// rollingVol mirrors features.py _rolling_vol: rolling sample std (ddof=1) of
// log returns over `window`, requiring a full window, times sqrt(252).
//
// The first valid log return is at index 1, so the first valid rolling-vol row
// is at index `window` (a window of `window` returns ending there).
func rollingVol(logRet []float64, window int) []float64 {
	n := len(logRet)
	out := make([]float64, n)
	for i := range out {
		out[i] = nan()
	}
	// Window of returns is logRet[i-window+1 .. i], all must be finite.
	for i := 0; i < n; i++ {
		start := i - window + 1
		if start < 1 { // index 0 of logRet is NaN, so a full finite window starts at 1
			continue
		}
		std, ok := sampleStd(logRet[start : i+1])
		if !ok {
			continue
		}
		out[i] = std * annualization
	}
	return out
}

// sampleStd computes the sample standard deviation (ddof=1, matching pandas
// default). Returns ok=false if any value is non-finite or the window is too
// short to debias.
func sampleStd(vals []float64) (float64, bool) {
	n := len(vals)
	if n < 2 {
		return 0, false
	}
	var sum float64
	for _, v := range vals {
		if !isFinite(v) {
			return 0, false
		}
		sum += v
	}
	mean := sum / float64(n)
	var ss float64
	for _, v := range vals {
		d := v - mean
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1)), true
}

// ewmaVol mirrors features.py _ewma_vol: ewm(halflife).std() with adjust=False,
// times sqrt(252). The std uses pandas' debiased exponentially-weighted sample
// variance (West's incremental algorithm with bias correction), verified to
// match pandas exactly. Index 0 is NaN.
func ewmaVol(logRet []float64, halflife int) []float64 {
	n := len(logRet)
	out := make([]float64, n)
	for i := range out {
		out[i] = nan()
	}

	// alpha for a given halflife: 1 - exp(ln(0.5)/halflife).
	alpha := 1.0 - math.Exp(math.Log(0.5)/float64(halflife))

	// Find the first finite log return to seed the recurrence (index 1).
	seed := -1
	for i := 0; i < n; i++ {
		if isFinite(logRet[i]) {
			seed = i
			break
		}
	}
	if seed < 0 {
		return out
	}

	mean := logRet[seed]
	variance := 0.0
	sumW := 1.0
	sumW2 := 1.0
	// First observation has no debiased variance (NaN), matching pandas.
	for i := seed + 1; i < n; i++ {
		x := logRet[i]
		if !isFinite(x) {
			// A gap in returns; pandas would carry state forward, but our inputs
			// are contiguous post-warm-up, so treat as warm-up.
			continue
		}
		// Decay existing weights by (1-alpha).
		sumW *= (1.0 - alpha)
		sumW2 *= (1.0 - alpha) * (1.0 - alpha)
		// Incorporate the new observation with weight alpha (West's update).
		wNew := alpha
		newSumW := sumW + wNew
		delta := x - mean
		mean += (wNew / newSumW) * delta
		variance = (sumW*variance + wNew*delta*(x-mean)) / newSumW
		sumW = newSumW
		sumW2 += wNew * wNew
		// Debias: divide by (1 - sumW2/sumW^2).
		denom := 1.0 - (sumW2 / (sumW * sumW))
		if denom > 0 {
			out[i] = math.Sqrt(variance/denom) * annualization
		}
		// Renormalize so sumW stays 1 (numerical stability, mirrors pandas).
		sumW2 /= (sumW * sumW)
		sumW = 1.0
	}
	return out
}

// garmanKlass mirrors features.py _garman_klass. Per-day variance estimate
// 0.5*(ln(H/L))^2 - (2*ln2 - 1)*(ln(C/O))^2, clipped at 0, sqrt'd, * sqrt(252).
func garmanKlass(open, high, low, close []float64) []float64 {
	n := len(open)
	out := make([]float64, n)
	c := 2.0*math.Log(2.0) - 1.0
	for i := 0; i < n; i++ {
		hl := math.Log(high[i] / low[i])
		co := math.Log(close[i] / open[i])
		dailyVar := 0.5*hl*hl - c*co*co
		if dailyVar < 0 {
			dailyVar = 0
		}
		out[i] = math.Sqrt(dailyVar) * annualization
	}
	return out
}

// rsi mirrors features.py _rsi: Wilder's RSI using ewm(alpha=1/period,
// adjust=False, min_periods=period) on gains and losses. When avg_loss is zero,
// RSI saturates at 100.
func rsi(close []float64, period int) []float64 {
	n := len(close)
	out := make([]float64, n)
	for i := range out {
		out[i] = nan()
	}
	if n < 2 {
		return out
	}

	gain := make([]float64, n)
	loss := make([]float64, n)
	gain[0] = nan()
	loss[0] = nan()
	for i := 1; i < n; i++ {
		delta := close[i] - close[i-1]
		if delta > 0 {
			gain[i] = delta
			loss[i] = 0
		} else {
			gain[i] = 0
			loss[i] = -delta
		}
	}

	alpha := 1.0 / float64(period)
	// Wilder smoothing = ewm mean with adjust=False, min_periods=period. The
	// first valid output is at index `period` (period deltas starting at index 1).
	avgGain := wilderEMA(gain, alpha, 1, period)
	avgLoss := wilderEMA(loss, alpha, 1, period)

	for i := 0; i < n; i++ {
		ag := avgGain[i]
		al := avgLoss[i]
		if !isFinite(ag) || !isFinite(al) {
			continue
		}
		if al == 0 {
			out[i] = 100.0
			continue
		}
		rs := ag / al
		out[i] = 100.0 - (100.0 / (1.0 + rs))
	}
	return out
}

// atr mirrors features.py _atr: true range smoothed by ewm(alpha=1/period,
// adjust=False, min_periods=period). True range uses prev close; index 0 has no
// prev close so its TR is NaN.
func atr(high, low, close []float64, period int) []float64 {
	n := len(high)
	tr := make([]float64, n)
	tr[0] = nan()
	for i := 1; i < n; i++ {
		hl := math.Abs(high[i] - low[i])
		hc := math.Abs(high[i] - close[i-1])
		lc := math.Abs(low[i] - close[i-1])
		tr[i] = math.Max(hl, math.Max(hc, lc))
	}
	alpha := 1.0 / float64(period)
	return wilderEMA(tr, alpha, 1, period)
}

// wilderEMA computes ewm(alpha, adjust=False).mean() over vals, where the input
// becomes valid starting at firstValid (earlier entries are NaN warm-up), and
// the output is NaN until at least minPeriods observations have been seen
// (mirroring pandas min_periods). The recurrence is m_t = (1-alpha)*m_{t-1} +
// alpha*x_t, seeded with the first observation.
func wilderEMA(vals []float64, alpha float64, firstValid, minPeriods int) []float64 {
	n := len(vals)
	out := make([]float64, n)
	for i := range out {
		out[i] = nan()
	}
	if firstValid >= n {
		return out
	}

	m := vals[firstValid]
	count := 1
	if count >= minPeriods {
		out[firstValid] = m
	}
	for i := firstValid + 1; i < n; i++ {
		x := vals[i]
		if !isFinite(x) {
			continue
		}
		m = (1.0-alpha)*m + alpha*x
		count++
		if count >= minPeriods {
			out[i] = m
		}
	}
	return out
}

// bbWidth mirrors features.py _bb_width: (4 * rolling_std) / rolling_sma over
// `window`, requiring a full window. Uses sample std (ddof=1) on close prices.
func bbWidth(close []float64, window int) []float64 {
	n := len(close)
	out := make([]float64, n)
	for i := range out {
		out[i] = nan()
	}
	for i := window - 1; i < n; i++ {
		seg := close[i-window+1 : i+1]
		var sum float64
		for _, v := range seg {
			sum += v
		}
		mean := sum / float64(window)
		std, ok := sampleStd(seg)
		if !ok || mean == 0 {
			continue
		}
		out[i] = (4.0 * std) / mean
	}
	return out
}

// portfolioFeatureSet bundles the four portfolio feature series plus the scalar
// concentration and the first index where all four are finite.
type portfolioFeatureSet struct {
	portfolioReturn []float64
	portfolioVol21  []float64
	concentration   float64
	avgCorrelation  []float64
	firstValid      int
}

// computePortfolioFeatures mirrors ml/data/portfolio_features.py. It assumes the
// per-symbol log returns are already positionally aligned (same date axis,
// totalRows long). portfolio_return is the weighted sum of log returns;
// portfolio_vol_21 is its 21-day rolling std * sqrt(252); concentration is the
// Herfindahl index of the weights; portfolio_avg_correlation is the mean of the
// strict upper triangle of the 63-day rolling correlation matrix.
func computePortfolioFeatures(
	perSymbol []symbolFeatures,
	symbols []string,
	weights map[string]float64,
	totalRows int,
) portfolioFeatureSet {
	p := len(symbols)
	w := make([]float64, p)
	for i, sym := range symbols {
		w[i] = weights[sym]
	}

	// portfolio_return at each row: dot(returns_row, weights). NaN if any symbol
	// return is NaN (warm-up). Mirrors the dropna(how="any") alignment.
	portRet := make([]float64, totalRows)
	for t := 0; t < totalRows; t++ {
		valid := true
		var acc float64
		for i := 0; i < p; i++ {
			r := perSymbol[i].logReturn[t]
			if !isFinite(r) {
				valid = false
				break
			}
			acc += r * w[i]
		}
		if valid {
			portRet[t] = acc
		} else {
			portRet[t] = nan()
		}
	}

	// portfolio_vol_21: rolling sample std of portRet over 21 rows * sqrt(252).
	portVol := make([]float64, totalRows)
	for i := range portVol {
		portVol[i] = nan()
	}
	for i := portfolioVolWin - 1; i < totalRows; i++ {
		seg := portRet[i-portfolioVolWin+1 : i+1]
		std, ok := sampleStd(seg)
		if !ok {
			continue
		}
		portVol[i] = std * annualization
	}

	// concentration: Herfindahl index = sum(w_i^2). Constant across rows.
	var concentration float64
	for _, wi := range w {
		concentration += wi * wi
	}

	// portfolio_avg_correlation: at row i, the mean of the strict upper triangle
	// of corrcoef over the trailing 63 rows of aligned returns. Requires i+1 >=
	// correlationWindow valid rows. Mirrors the loop in portfolio_features.py.
	avgCorr := make([]float64, totalRows)
	for i := range avgCorr {
		avgCorr[i] = nan()
	}
	if p >= 2 {
		for i := 0; i < totalRows; i++ {
			if i+1 < correlationWindow {
				continue
			}
			windowStart := i + 1 - correlationWindow
			// Collect the 63-row window; skip if any return is NaN.
			ok := true
			cols := make([][]float64, p)
			for c := 0; c < p; c++ {
				cols[c] = make([]float64, correlationWindow)
			}
			for r := 0; r < correlationWindow; r++ {
				for c := 0; c < p; c++ {
					v := perSymbol[c].logReturn[windowStart+r]
					if !isFinite(v) {
						ok = false
						break
					}
					cols[c][r] = v
				}
				if !ok {
					break
				}
			}
			if !ok {
				continue
			}
			avgCorr[i] = upperTriMeanCorr(cols)
		}
	}

	// firstValid: first row where all four features are finite. concentration is
	// always finite. The binding constraints are portVol (>=21 valid returns) and
	// avgCorr (>=63 valid returns), so avgCorr typically dominates.
	firstValid := totalRows
	for i := 0; i < totalRows; i++ {
		if isFinite(portRet[i]) && isFinite(portVol[i]) && isFinite(avgCorr[i]) {
			firstValid = i
			break
		}
	}

	return portfolioFeatureSet{
		portfolioReturn: portRet,
		portfolioVol21:  portVol,
		concentration:   concentration,
		avgCorrelation:  avgCorr,
		firstValid:      firstValid,
	}
}

// upperTriMeanCorr computes the Pearson correlation matrix of the given columns
// (each a series of equal length) and returns the mean of the strict upper
// triangle, mirroring _upper_tri_mean and np.corrcoef in
// ml/data/portfolio_features.py. Non-finite pairwise correlations (constant
// column) are skipped from the mean.
func upperTriMeanCorr(cols [][]float64) float64 {
	p := len(cols)
	if p < 2 {
		return nan()
	}
	means := make([]float64, p)
	stds := make([]float64, p)
	n := len(cols[0])
	for c := 0; c < p; c++ {
		var sum float64
		for _, v := range cols[c] {
			sum += v
		}
		means[c] = sum / float64(n)
	}
	for c := 0; c < p; c++ {
		var ss float64
		for _, v := range cols[c] {
			d := v - means[c]
			ss += d * d
		}
		// np.corrcoef uses population covariance (the normalization cancels in the
		// correlation ratio, so ddof choice does not matter). Use sqrt of sum of
		// squares directly.
		stds[c] = math.Sqrt(ss)
	}

	var sum float64
	var count int
	for a := 0; a < p; a++ {
		for b := a + 1; b < p; b++ {
			if stds[a] == 0 || stds[b] == 0 {
				continue // constant column -> undefined correlation, skip
			}
			var cov float64
			for k := 0; k < n; k++ {
				cov += (cols[a][k] - means[a]) * (cols[b][k] - means[b])
			}
			corr := cov / (stds[a] * stds[b])
			if isFinite(corr) {
				sum += corr
				count++
			}
		}
	}
	if count == 0 {
		return nan()
	}
	return sum / float64(count)
}
