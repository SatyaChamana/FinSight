# Phase 0B: ML Deep Dive -- From Raw Data to ONNX Model

## Part 1: Data -- What We Download and Why

### Source: yfinance

```python
import yfinance as yf

# Download daily OHLCV for Apple
aapl = yf.download("AAPL", start="2015-01-01", end="2025-01-01")
```

What you get per day:

| Column | Meaning | Example |
|---|---|---|
| **Open** | Price at market open (9:30 AM ET) | $148.50 |
| **High** | Highest price during the day | $152.30 |
| **Low** | Lowest price during the day | $147.80 |
| **Close** | Price at market close (4:00 PM ET) | $151.20 |
| **Volume** | Number of shares traded | 58,200,000 |

That's **OHLCV**: Open, High, Low, Close, Volume. The raw material for everything.

**How much data?**
- 10 years of daily data = ~2,520 trading days per asset
- 50 assets in a portfolio = 50 x 2,520 = 126,000 rows
- This is SMALL by ML standards. NLP models train on billions of tokens. We have thousands of rows. This is why full Transformer overfits and LSTM + attention is the right call.

**What about weekends and holidays?**
- Markets are closed. No data. yfinance skips these.
- If different assets have different trading calendars (US stocks vs international), you align on the intersection of trading days.

---

## Part 2: Feature Engineering -- Turning Raw Prices Into Model Inputs

Raw OHLCV is useless to a model. The model needs **features** that capture patterns. Each feature is a number computed from raw data.

### Feature Category 1: Price-Derived

**Log returns (primary input):**
```python
import numpy as np

# Daily log return
log_return = np.log(close_today / close_yesterday)

# Example: close went from 150 to 153
# ln(153/150) = 0.0198 = +1.98%
```

Why log returns, not raw prices? (Recall from 0A: stationarity, additivity, approximate normality.)

**Intraday range:**
```python
# How much the price moved within the day
intraday_range = (high - low) / close

# Big range = volatile day, small range = quiet day
```

### Feature Category 2: Volatility Features

These directly relate to what we're predicting. Think of them as "hints" to the model.

**Rolling standard deviation:**
```python
# Volatility over different windows
vol_5d  = returns.rolling(5).std()   # 1 week
vol_21d = returns.rolling(21).std()  # 1 month
vol_63d = returns.rolling(63).std()  # 3 months (1 quarter)
```

Why multiple windows? They capture different regimes:
- 5d vol spikes during flash crashes (short-term shock)
- 63d vol rises during prolonged bear markets (regime change)
- When 5d vol >> 63d vol, a sudden event just happened
- When 5d vol << 63d vol, markets are calming down after a stressed period

**EWMA volatility (Exponentially Weighted Moving Average):**
```python
# Recent days matter more than old days
# decay factor (lambda) typically 0.94 for daily data (RiskMetrics standard)
ewma_vol = returns.ewm(span=21).std()
```

Why EWMA on top of rolling std?
- Rolling window treats day 1 and day 21 equally
- EWMA says "yesterday's return matters more than 21 days ago"
- Markets have **volatility clustering**: volatile days tend to follow volatile days. EWMA captures this better.
- Interview: "We used EWMA with lambda 0.94 following the RiskMetrics methodology because financial volatility clusters, and exponential weighting captures the recency effect."

**Garman-Klass volatility:**
```python
# Uses OHLC (not just close-to-close)
# Captures intraday price movement that close-to-close misses
gk_vol = 0.5 * np.log(high/low)**2 - (2*np.log(2) - 1) * np.log(close/open)**2
```

Why? Close-to-close volatility misses what happened during the day. A stock that opens at $100, crashes to $80, then recovers to $100 has zero close-to-close return but massive intraday risk. Garman-Klass captures this.

Interview: "We included Garman-Klass volatility because close-to-close measures miss intraday risk. A stock can round-trip and show zero return but have experienced extreme intraday stress."

### Feature Category 3: Technical Indicators

**RSI (Relative Strength Index):**
```python
# Measures momentum: 0-100 scale
# RSI > 70 = "overbought" (might drop)
# RSI < 30 = "oversold" (might bounce)

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

Why RSI for risk prediction? Extreme RSI values precede reversals. An RSI of 85 suggests the asset is stretched, increasing the probability of a pullback (higher short-term VaR).

**ATR (Average True Range):**
```python
# Measures market volatility using OHLC
true_range = np.maximum(
    high - low,
    np.maximum(abs(high - prev_close), abs(low - prev_close))
)
atr = true_range.rolling(14).mean()
```

ATR is similar to Garman-Klass but expressed in dollar terms, not log terms. Traders use it for position sizing. For our model, it's another volatility signal using different information (gaps between close and next open).

**Bollinger Band Width:**
```python
sma_20 = close.rolling(20).mean()
std_20 = close.rolling(20).std()
upper = sma_20 + 2 * std_20
lower = sma_20 - 2 * std_20
bb_width = (upper - lower) / sma_20
```

Why? Width contracts before explosive moves ("the squeeze") and expands during volatile periods. Gives the model a leading indicator of volatility regime changes.

### Feature Category 4: Portfolio-Level Features

Individual asset features aren't enough. Risk is a portfolio property.

**Weighted portfolio return:**
```python
# weights = [0.3, 0.25, 0.2, 0.15, 0.1]
# asset_returns = DataFrame of per-asset returns
portfolio_return = (asset_returns * weights).sum(axis=1)
```

**Rolling correlation:**
```python
# Pairwise correlation between assets over trailing 63 days
# When correlations spike toward 1.0, diversification fails
# (all assets drop together, like March 2020)
rolling_corr = asset_returns.rolling(63).corr()
avg_correlation = rolling_corr.mean()  # average pairwise correlation
```

This is critical. During crises, correlations spike to 1.0 (everything drops together). The model needs to see this signal to predict higher VaR during stress.

**Herfindahl concentration index:**
```python
# Measures how concentrated the portfolio is
# HHI = sum of squared weights
# HHI = 1.0 means one asset (max concentration)
# HHI = 1/N means equal weight (max diversification)
hhi = np.sum(weights ** 2)

# Our portfolio: 0.3^2 + 0.25^2 + 0.2^2 + 0.15^2 + 0.1^2
# = 0.09 + 0.0625 + 0.04 + 0.0225 + 0.01 = 0.225
# Min possible with 5 assets = 0.2 (equal weight)
# So we're slightly concentrated (AAPL at 30% dominates)
```

### Feature Category 5: Macro Features

**VIX (the "fear index"):**
```python
vix = yf.download("^VIX", start="2015-01-01")
# VIX < 15: calm markets
# VIX 15-25: normal
# VIX 25-35: stressed
# VIX > 35: panic (COVID crash hit 82)
```

VIX measures the market's EXPECTATION of future volatility (derived from S&P 500 options prices). It's forward-looking, unlike our other volatility features which are backward-looking. Gives the model the market's own risk estimate.

### Total Feature Count

Per asset per day:
```
Log return              1
Intraday range          1
Vol 5d, 21d, 63d        3
EWMA vol                1
Garman-Klass vol        1
RSI                     1
ATR                     1
Bollinger width         1
                       ---
Per asset:             10 features
```

Portfolio-level per day:
```
Weighted return         1
Avg pairwise corr       1
HHI concentration       1
VIX                     1
                       ---
Portfolio:              4 features
```

Total for 50 assets: `50 * 10 + 4 = 504 features per day`

Input tensor shape: `[batch_size, 63 days, 504 features]`

---

## Part 3: Train/Test Split -- Walk-Forward (Never Peek Into the Future)

**Why random split is WRONG for time series:**

```
WRONG (random split):
  Day 1: train
  Day 2: TEST    <- model saw Day 3 during training!
  Day 3: train
  Day 4: TEST

  Model "learned" from future data. Test accuracy is fake.
  This is called DATA LEAKAGE. Guaranteed to fail in production.
```

**Walk-forward split (correct):**

```
|-------- Train --------|--- Val ---|--- Test ---|
2015                    2022       2023         2025

Rules:
- Train: everything before validation start
- Validation: used for early stopping and hyperparameter tuning
- Test: NEVER touched until final evaluation. Simulates production.
- Data only flows forward in time. Never backward.
```

**Walk-forward cross-validation (even better):**

```
Fold 1: |--Train--|--Val--|
Fold 2:    |--Train--|--Val--|
Fold 3:       |--Train--|--Val--|

Each fold slides the window forward.
Tests whether the model generalizes across different market regimes.
```

Interview: "We used walk-forward validation because random splits cause data leakage in time series. The model must only train on past data and validate on future data, exactly like production. We also used multiple walk-forward folds to verify the model works across different market regimes."

---

## Part 4: Model Architecture -- LSTM + Attention Deep Dive

### What is an LSTM and Why?

Regular neural networks see all inputs at once. But financial data is sequential: day 1 matters for day 2, which matters for day 3. **Order matters.**

**RNN (Recurrent Neural Network)** is the basic solution: process one day at a time, pass a "hidden state" from each day to the next. The hidden state is the network's "memory" of what it's seen so far.

Problem: vanilla RNNs have **vanishing gradients**. After ~20 steps, the gradient signal from early days becomes negligibly small. The network "forgets" what happened 30+ days ago.

**LSTM (Long Short-Term Memory)** solves this with **gates**: mechanisms that control what to remember, what to forget, and what to output.

```
LSTM Cell (one per timestep):

       forget gate    input gate    output gate
          |              |              |
  h(t-1) --> [sigma] --> [sigma] --> [sigma] --> h(t)
              |            |            |
              v            v            v
  c(t-1) --> [x forget] + [x new] ------------> c(t)

  c = cell state (long-term memory, the key innovation)
  h = hidden state (short-term output)
  sigma = sigmoid function (outputs 0-1, acts as a "gate")
```

**Three gates in plain English:**

| Gate | Question it answers | Example |
|---|---|---|
| **Forget gate** | "Should I forget old information?" | "Volatility regime changed. Forget the calm period stats." |
| **Input gate** | "Should I store this new information?" | "Big VIX spike today. This is important, store it." |
| **Output gate** | "What should I output for this timestep?" | "Based on everything I remember, output this hidden state." |

The **cell state** (c) is the magic. It flows through time with minimal modification (just multiply by forget gate, add new input). Gradients flow through the cell state cleanly, solving the vanishing gradient problem. The network can remember patterns from 63 days ago.

**Our configuration: 2-layer stacked LSTM**

```
Day 1 features --> LSTM Layer 1 --> LSTM Layer 2 --> hidden state 1
Day 2 features --> LSTM Layer 1 --> LSTM Layer 2 --> hidden state 2
...
Day 63 features --> LSTM Layer 1 --> LSTM Layer 2 --> hidden state 63

Layer 1 captures low-level patterns (daily volatility spikes, returns)
Layer 2 captures higher-level patterns (weekly trends, regime shifts)
Dropout 0.3 between layers prevents overfitting
```

Why 2 layers? One layer underfits (can't capture complex patterns). Three layers overfits on our small dataset. Two is the empirical sweet spot for financial time series.

### What is Self-Attention and Why Add It?

LSTM processes sequentially: day 1, day 2, ..., day 63. By day 63, the memory of day 1 is faded (better than vanilla RNN, but still degraded).

**Self-attention** lets the model directly compare ANY day to ANY other day:

```
Without attention (LSTM only):
  Day 63 knows about Day 1 only through a chain of 62 hidden states.
  Signal degrades.

With attention:
  Day 63 directly asks: "Which past days are most relevant to me?"
  Answer: "Day 5 (earnings shock) and Day 42 (similar volatility spike)"
  The model ATTENDS to those specific days.
```

**How attention works (simplified):**

```
Inputs: all 63 hidden states from LSTM [h1, h2, ..., h63]

Step 1: For each pair of days, compute a "relevance score"
        score(i, j) = how relevant is day i to day j?
        Computed as: dot_product(h_i * W_query, h_j * W_key) / sqrt(d)

Step 2: Normalize scores with softmax (make them sum to 1)
        attention_weights = softmax(scores)

Step 3: Weighted sum of all hidden states
        context = sum(attention_weights * h_values)
```

**Query, Key, Value -- the interview explanation:**

Think of it like a search engine:
- **Query:** "What am I looking for?" (each day asks this)
- **Key:** "What do I contain?" (each day advertises this)
- **Value:** "What information do I provide?" (the actual content)

Score = how well Query matches Key. High score = "this day is relevant to my query." Output = weighted average of Values, where weights are the scores.

**Multi-head attention (4 heads):**

Each head learns to look for different patterns:
```
Head 1: might focus on recent volatility spikes (short-term)
Head 2: might focus on earnings-related days (quarterly pattern)
Head 3: might focus on macro regime shifts (correlation spikes)
Head 4: might focus on volume anomalies (liquidity events)
```

We don't manually assign these roles. The model learns what each head should attend to through training. But the result is that 4 heads give 4 different "perspectives" on which past days matter.

**Why not full Transformer?**

Full Transformer = stack of 6-12 attention layers + feedforward layers + positional encoding. Designed for billions of tokens in NLP.

We have 63 timesteps. Full Transformer:
- Overfits (too many parameters for too little data)
- Positional encoding for time series is an open research problem (sinusoidal encoding designed for text doesn't capture calendar effects)
- Overkill complexity for our problem size

LSTM + one attention layer = best of both worlds. LSTM handles the sequential nature, attention handles the "look back at important days" capability.

### Putting It Together: Full Architecture

```python
import torch
import torch.nn as nn

class RiskPredictor(nn.Module):
    def __init__(
        self,
        input_dim=504,      # features per day
        hidden_dim=128,      # LSTM hidden size
        num_layers=2,        # stacked LSTM layers
        num_heads=4,         # attention heads
        dropout=0.3
    ):
        super().__init__()

        # Layer 1 & 2: Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Three output heads
        self.volatility_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
        self.cvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch, 63, 504]

        # LSTM processes sequence
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch, 63, 128]

        # Self-attention over LSTM outputs
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        # attn_out shape: [batch, 63, 128]

        # Take the last timestep's attended representation
        context = attn_out[:, -1, :]
        # context shape: [batch, 128]

        # Three prediction heads
        volatility = self.volatility_head(context)
        var = self.var_head(context)
        cvar = self.cvar_head(context)

        return volatility, var, cvar
```

**Walk through a single forward pass:**

```
Input: 63 days of 504 features for a batch of 32 portfolios
       Shape: [32, 63, 504]
         |
         v
LSTM (2 layers, 128 hidden):
  Processes day-by-day. Each day updates hidden state.
  Output: hidden state per day.
  Shape: [32, 63, 128]
         |
         v
Self-Attention (4 heads):
  Each day attends to all other days.
  "Day 63, which past days matter most for predicting risk?"
  Output: attention-weighted hidden states.
  Shape: [32, 63, 128]
         |
         v
Take last timestep (day 63):
  This contains the full summary: LSTM memory + attention context.
  Shape: [32, 128]
         |
         v
Three linear heads:
  volatility = Linear(128 -> 1)  -> [32, 1]
  var        = Linear(128 -> 1)  -> [32, 1]
  cvar       = Linear(128 -> 1)  -> [32, 1]
```

**Parameter count:**
```
LSTM:      ~530,000 parameters (2 layers, 504 input, 128 hidden)
Attention:  ~66,000 parameters (4 heads, 128 dim)
Heads:         ~390 parameters (3 x (128 + 1))
Total:     ~596,000 parameters

For comparison:
  GPT-3: 175 BILLION parameters
  Our model: 600K parameters

This is intentionally small. Financial data is scarce. Small model = less overfitting.
```

---

## Part 5: Loss Functions -- Teaching the Model What "Wrong" Means

### Huber Loss (for volatility)

```
Huber(y, pred):
  if |y - pred| <= delta:
    return 0.5 * (y - pred)^2          # MSE when error is small
  else:
    return delta * |y - pred| - 0.5 * delta^2  # MAE when error is large
```

```python
# PyTorch
volatility_loss = nn.HuberLoss(delta=1.0)
```

Why not plain MSE? MSE squares the error. On flash crash days where volatility is 10x normal, the squared error dominates the entire loss, pulling the model toward overfitting to rare extreme days. Huber transitions to linear (MAE) for large errors, so outlier days influence the model but don't dominate it.

Interview: "Huber loss gives us the best of both worlds: MSE's sensitivity for normal days and MAE's robustness for extreme outlier days like flash crashes."

### Quantile Loss / Pinball Loss (for VaR)

This is the most important loss to understand deeply.

```
QuantileLoss(y_true, y_pred, alpha=0.95):
  error = y_true - y_pred

  if error > 0:   # underestimated the loss (dangerous!)
    return alpha * error        = 0.95 * error
  else:            # overestimated the loss (conservative)
    return (1 - alpha) * |error| = 0.05 * |error|
```

**The asymmetry is the point:**
```
Underestimating VaR (saying loss will be $1k when it's $5k):
  Penalty = 0.95 * $4k = $3,800     <- HEAVY penalty

Overestimating VaR (saying loss will be $5k when it's $1k):
  Penalty = 0.05 * $4k = $200       <- light penalty
```

The model learns: "It's 19x more expensive to underestimate risk than overestimate it." This pushes the prediction to the 95th percentile of the loss distribution, which is exactly the definition of VaR(95%).

```python
def quantile_loss(y_pred, y_true, alpha=0.95):
    error = y_true - y_pred
    return torch.mean(
        torch.max(alpha * error, (alpha - 1) * error)
    )
```

Interview: "We use quantile loss because VaR is a percentile, not a mean. MSE would predict the average loss, which is useless for VaR. Pinball loss at alpha=0.95 penalizes underestimation 19x more than overestimation, which mathematically forces the prediction to converge on the 95th percentile."

### Truncated MSE (for CVaR)

CVaR is the average loss beyond VaR. We only care about accuracy in the tail.

```python
def truncated_mse(cvar_pred, loss_true, var_pred):
    # Only compute loss where actual loss exceeds predicted VaR
    tail_mask = loss_true > var_pred
    if tail_mask.sum() == 0:
        return torch.tensor(0.0)
    tail_errors = (cvar_pred[tail_mask] - loss_true[tail_mask]) ** 2
    return tail_errors.mean()
```

Why truncated? Full MSE would optimize CVaR accuracy across ALL days. But CVaR is only defined for tail days (beyond VaR). Computing MSE on non-tail days adds noise and confuses the model.

### Combined Loss

```python
def combined_loss(vol_pred, var_pred, cvar_pred, vol_true, var_true, cvar_true,
                  w1=1.0, w2=1.0, w3=1.0):
    l_vol = nn.HuberLoss()(vol_pred, vol_true)
    l_var = quantile_loss(var_pred, var_true, alpha=0.95)
    l_cvar = truncated_mse(cvar_pred, cvar_true, var_pred)

    return w1 * l_vol + w2 * l_var + w3 * l_cvar
```

Start with equal weights (1.0, 1.0, 1.0). Tune later based on which target has worse performance on validation set.

---

## Part 6: Training Loop -- How the Model Learns

### Core concepts

**Epoch:** One complete pass through all training data.
**Batch:** A chunk of training samples processed together (e.g., 32 portfolios).
**Learning rate:** How big a step the optimizer takes. Too big = overshoots, too small = never converges.

```python
model = RiskPredictor(input_dim=504)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    for batch_x, batch_targets in train_loader:
        # Forward pass
        vol_pred, var_pred, cvar_pred = model(batch_x)

        # Compute loss
        loss = combined_loss(
            vol_pred, var_pred, cvar_pred,
            batch_targets['vol'], batch_targets['var'], batch_targets['cvar']
        )

        # Backward pass (compute gradients)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients in LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

    # Validate after each epoch
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

### Key training concepts explained

**Adam optimizer:** Adaptive learning rate per parameter. Combines momentum (keeps moving in the same direction) with RMSprop (adapts to each parameter's gradient magnitude). Standard choice for most deep learning. Learning rate of 1e-3 is a good default.

**Gradient clipping:** LSTMs can have **exploding gradients** (gradient becomes astronomically large, weights update wildly). `clip_grad_norm_` caps the total gradient magnitude at 1.0. If gradient is 500.0, it gets scaled down to 1.0. Prevents training from blowing up.

Interview: "We clip gradients at max_norm=1.0 because LSTMs are susceptible to exploding gradients, especially with volatile financial data. Without clipping, a single batch containing a market crash day could destabilize the entire model."

**Early stopping:**
```python
patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(100):
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pt")  # save checkpoint
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model (not the last one!)
model.load_state_dict(torch.load("best_model.pt"))
```

Why? Training loss always decreases. Validation loss decreases, then starts INCREASING (overfitting). Early stopping saves the model at the sweet spot before overfitting begins.

```
Loss
  |    Training loss keeps dropping
  |  \  ______________________________
  |   \/
  |    \   Validation loss
  |     \    /
  |      \  /   <- STOP HERE (best model)
  |       \/
  |
  +---------------------------------------- Epochs
```

**Overfitting signs to watch:**
- Train loss drops, val loss rises -> classic overfitting
- Model memorizes March 2020 crash -> predicts crash-level VaR on calm days
- Training accuracy is 98%, validation accuracy is 60% -> memorized, not generalized

---

## Part 7: Computing Ground Truth Labels

The model predicts VaR, CVaR, and volatility. But what are the "correct answers" we train against?

```python
def compute_labels(portfolio_returns, window=63, alpha=0.95):
    """
    For each day t, compute labels using the NEXT window of returns.
    This is what the model should have predicted.
    """
    labels = []
    for t in range(len(portfolio_returns) - window):
        future_returns = portfolio_returns[t+1 : t+1+window]

        # Volatility: std of future returns
        vol = future_returns.std()

        # VaR(95%): 5th percentile of future returns (losses are negative)
        var_95 = -np.percentile(future_returns, 100 - 95)

        # CVaR: mean of returns worse than VaR
        losses = -future_returns
        cvar_95 = losses[losses >= var_95].mean()

        labels.append({'vol': vol, 'var': var_95, 'cvar': cvar_95})

    return labels
```

**Important:** Labels are computed from FUTURE returns (the next 63 days). The model sees PAST features (the previous 63 days) and predicts future risk. This is why walk-forward splitting is critical. If future data leaks into training, the model "sees" the answers.

---

## Part 8: ONNX Export -- Bridging Python and Go

### What is ONNX?

Open Neural Network Exchange. A standardized format for neural network models. Think of it as "PDF for ML models": any framework can export to ONNX, any runtime can load it.

```
PyTorch model (.pt) --> ONNX export --> ONNX model (.onnx) --> Go loads via ONNX Runtime
```

Why not just run PyTorch in production?
- PyTorch is Python. Our service is Go.
- ONNX Runtime is optimized C++ with Go bindings. Faster inference.
- Decouples training (Python, data scientists) from serving (Go, production).
- Interview: "ONNX gives us language-agnostic model serving. Data scientists iterate in PyTorch, production serves via ONNX Runtime in Go. Neither team blocks the other."

**Export code:**
```python
import torch.onnx

dummy_input = torch.randn(1, 63, 504)  # batch=1, 63 days, 504 features

torch.onnx.export(
    model,
    dummy_input,
    "risk_model.onnx",
    input_names=["features"],
    output_names=["volatility", "var_95", "cvar_95"],
    dynamic_axes={
        "features": {0: "batch_size"},    # batch can vary
        "volatility": {0: "batch_size"},
        "var_95": {0: "batch_size"},
        "cvar_95": {0: "batch_size"},
    },
    opset_version=17,
)
```

**Validation (critical step):**
```python
import onnxruntime as ort

# Load exported model
session = ort.InferenceSession("risk_model.onnx")

# Run same input through both
pytorch_out = model(dummy_input)
onnx_out = session.run(None, {"features": dummy_input.numpy()})

# Compare outputs (must match within tolerance)
for name, pt, ox in zip(["vol", "var", "cvar"], pytorch_out, onnx_out):
    diff = abs(pt.item() - ox[0].item())
    assert diff < 1e-5, f"{name} mismatch: PyTorch={pt.item()}, ONNX={ox[0].item()}"
    print(f"{name}: match (diff={diff:.8f})")
```

If these don't match within tolerance, something went wrong in export. Common causes: unsupported operations, numerical precision differences, dynamic shapes mishandled.

**Model metadata:**
```python
import onnx

model_onnx = onnx.load("risk_model.onnx")
meta = model_onnx.metadata_props.add()
meta.key = "model_version"
meta.value = "v1.0.0"
meta = model_onnx.metadata_props.add()
meta.key = "input_features"
meta.value = "504"
meta = model_onnx.metadata_props.add()
meta.key = "lookback_days"
meta.value = "63"
meta = model_onnx.metadata_props.add()
meta.key = "trained_on"
meta.value = "2015-01-01 to 2024-12-31"
onnx.save(model_onnx, "risk_model.onnx")
```

This metadata travels with the model file. Go service reads it on load. Logged with every prediction. If something goes wrong, you know exactly which model version, what data it was trained on, and what inputs it expects.

---

## Part 8B: Synthetic Data for Stress Testing

Real data has limited extreme events (2008, 2020, maybe 3-4 crashes in 10 years). The model needs more tail samples.

```python
def generate_crash_scenario(n_assets=50, n_days=63):
    """Simulate a market crash: correlated drawdown across all assets."""
    # High correlation during crash (all assets drop together)
    correlation = 0.85

    # Negative drift with high volatility
    daily_returns = np.random.multivariate_normal(
        mean=[-0.02] * n_assets,          # -2% daily average
        cov=make_corr_matrix(n_assets, correlation) * 0.04,  # 4% daily vol
        size=n_days
    )
    return daily_returns

def generate_calm_scenario(n_assets=50, n_days=63):
    """Simulate low-volatility regime."""
    correlation = 0.3
    daily_returns = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,         # +0.05% daily
        cov=make_corr_matrix(n_assets, correlation) * 0.0001,
        size=n_days
    )
    return daily_returns
```

Mix real data (~80%) with synthetic stress scenarios (~20%) during training. This teaches the model that crashes exist and what they look like, even if the real training data only has 2-3 examples.

Interview: "We augmented real market data with synthetic stress scenarios to improve tail prediction. Real data has few crash samples, so the model would underestimate extreme risk without augmentation."

---

## Interview Flashcard: 0B

| Question | Your answer |
|---|---|
| "Why LSTM over Transformer?" | "Financial time series have strong local dependencies and small datasets (<3k days). LSTM handles sequential nature well. Full Transformer overfits. We add one attention layer for long-range lookback." |
| "What's quantile loss?" | "Asymmetric loss that penalizes underestimation of risk 19x more than overestimation at alpha=0.95. Forces prediction to the 95th percentile. Correct loss for VaR, which is a quantile, not a mean." |
| "Why not random train/test split?" | "Data leakage. Time series must split chronologically. Random split lets the model see future data during training. Walk-forward validation mimics production." |
| "Why ONNX?" | "Decouples training (Python) from serving (Go). ONNX Runtime is optimized C++. Data scientists iterate in PyTorch, production serves via Go with ONNX Runtime. Neither team blocks the other." |
| "How do you handle few crash samples?" | "Synthetic stress data augmentation. Generate correlated drawdown scenarios, mix 80/20 with real data. Model learns tail behavior from both real crashes and synthetic ones." |
| "What are the three gates in LSTM?" | "Forget (drop old info), Input (store new info), Output (what to emit). Cell state flows through time with minimal modification, solving vanishing gradients." |
| "Why Garman-Klass?" | "Close-to-close volatility misses intraday risk. A stock that round-trips shows zero return but experienced extreme stress. Garman-Klass uses full OHLC." |
| "How many parameters?" | "~600K. Intentionally small. Financial data is scarce compared to NLP. Small model = less overfitting." |

---

**Interview questions you can now answer:**
- "Walk me through your model architecture."
- "Why did you choose this loss function?"
- "How do you handle the small dataset problem in finance?"
- "Explain the attention mechanism in your model."
- "How do you prevent data leakage?"
- "What's the difference between your three output heads?"
