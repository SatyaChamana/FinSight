# Phase 0A: Domain Understanding -- Portfolio Risk

## What is a Portfolio?

A collection of financial assets (stocks, bonds, ETFs) held by an investor. Each asset has a **weight** (percentage of total value).

Example portfolio:
```
AAPL   30%   ($30,000)
GOOGL  25%   ($25,000)
MSFT   20%   ($20,000)
BND    15%   ($15,000)  <- bond ETF
GLD    10%   ($10,000)  <- gold ETF
Total: 100%  ($100,000)
```

Weights must sum to 1.0 (100%). When prices change, weights drift. Rebalancing = adjusting back to target weights.

## Why Measure Risk?

You have $100k in that portfolio. Three questions keep you up at night:

1. **"How bad could tomorrow be?"** -> VaR answers this
2. **"If it IS bad, HOW bad?"** -> CVaR answers this
3. **"How bumpy is the ride?"** -> Volatility answers this

Every hedge fund, bank, and asset manager measures these. Regulators (Basel III) require it. Without risk measurement, you're gambling blind.

---

## Returns: The Foundation

Risk is measured on **returns**, not prices. Why?

```
AAPL price: $150 -> $153
  Simple return:  (153 - 150) / 150 = 0.02 = +2%
  Log return:     ln(153/150) = 0.0198 = +1.98%
```

**Why returns, not prices?**
- $AAPL at $150 moving $3 is very different from $AAPL at $10 moving $3
- Returns normalize this. 2% is 2% regardless of price level.
- Returns are (approximately) **stationary**: their statistical properties don't change over time. Prices trend upward forever. Models need stationary data.

**Why log returns specifically?**
- Log returns are **additive** across time: `log_return(day1) + log_return(day2) = log_return(2_days)`. Simple returns aren't.
- Log returns are approximately **normally distributed** (bell curve). This matters for VaR calculation.
- For small values (<5%), log return and simple return are nearly identical.

Your model will train on log returns.

---

## Volatility

**Definition:** Standard deviation of returns over a period.

```
Daily returns for AAPL over 5 days:
  +1.2%, -0.8%, +2.1%, -1.5%, +0.3%

Mean = 0.26%
Volatility = std_dev = 1.31%
```

That's **daily** volatility. To annualize (industry standard):

```
Annual volatility = daily volatility * sqrt(252)

Why 252? Trading days per year.
Why sqrt? Volatility scales with square root of time (assuming returns are independent).

1.31% * sqrt(252) = 1.31% * 15.87 = 20.8% annualized
```

**Interpretation:** "AAPL's returns fluctuate roughly 20.8% per year around its mean." Higher volatility = riskier (more unpredictable).

**Portfolio volatility is NOT the weighted average of individual volatilities.** This is the key insight:

```
If AAPL vol = 25% and BND vol = 5%:
  50/50 portfolio vol != (25% + 5%) / 2 = 15%

Actual portfolio vol depends on CORRELATION between assets.
  If AAPL and BND are negatively correlated (when stocks drop, bonds rise),
  portfolio vol could be as low as 10%.

This is DIVERSIFICATION. The whole point of having a portfolio.
```

Portfolio volatility formula:
```
sigma_portfolio = sqrt(w^T * Sigma * w)

w = weight vector [0.3, 0.25, 0.2, 0.15, 0.1]
Sigma = covariance matrix (NxN matrix of how assets move together)
```

Our model predicts this portfolio-level volatility, accounting for correlations.

---

## Value at Risk (VaR)

**Definition:** "The maximum loss I expect over a given time horizon at a given confidence level."

```
VaR(95%, 1-day) = $2,300

Meaning: "I am 95% confident that my portfolio will NOT lose more than
$2,300 tomorrow."

Or equivalently: "On 95 out of 100 trading days, my loss will be less
than $2,300. On 5 days, it could be worse."
```

**How is VaR traditionally calculated?** Three methods:

1. **Historical simulation:** Sort last 1000 days of returns. The 50th worst day (5th percentile) is your VaR. Simple but assumes past = future.

2. **Parametric (variance-covariance):** Assume returns are normally distributed. VaR = mean - z_score * volatility. Fast but normality assumption is wrong (fat tails).

3. **Monte Carlo:** Simulate thousands of random scenarios. Take the 5th percentile. Flexible but slow.

**Our approach:** None of the above. We use LSTM + attention to **predict** VaR directly. The model learns the relationship between recent market conditions and future VaR. This captures non-linear patterns (regime changes, volatility clustering) that traditional methods miss.

**VaR's fatal flaw:**
```
Two portfolios, both with VaR(95%) = $2,300:

Portfolio A: On the worst 5% of days, loses $2,400 (slightly over VaR)
Portfolio B: On the worst 5% of days, loses $50,000 (catastrophic)

VaR says they're equally risky. They're not.
VaR tells you WHERE the cliff is, not HOW FAR the fall.
```

This is why VaR alone is considered a flawed metric. Regulators and quants know this. Interviewers will test if YOU know it.

---

## Conditional VaR (CVaR / Expected Shortfall)

**Definition:** "If I DO breach VaR, what's my average loss?"

```
CVaR(95%, 1-day) = $4,100

Meaning: "On the worst 5% of days (when VaR is breached), my AVERAGE
loss is $4,100."
```

CVaR fixes VaR's blind spot:
```
Portfolio A: VaR = $2,300, CVaR = $2,500  <- tail is mild
Portfolio B: VaR = $2,300, CVaR = $15,000 <- tail is catastrophic

NOW you can tell them apart.
```

**Mathematical properties (interview gold):**
- CVaR is **coherent** (satisfies subadditivity: combining portfolios can't increase risk). VaR is NOT coherent.
- CVaR >= VaR always. CVaR is the average of losses beyond VaR.
- Basel III shifted toward CVaR (Expected Shortfall) for regulatory capital requirements precisely because VaR is not subadditive.

**Interview answer:** "We predict both VaR and CVaR because VaR is the industry standard but has a known flaw: it ignores tail severity. CVaR complements it by measuring average loss in the tail. Together they give a complete picture: where the danger zone starts (VaR) and how bad it gets inside that zone (CVaR). This aligns with Basel III's shift toward Expected Shortfall."

---

## How the Three Metrics Work Together

```
                          Volatility = 20.8% annualized
                          "The ride is bumpy"
                                |
        Daily return distribution for your portfolio:

    Extreme loss        VaR boundary      Mean        Gain
        |                   |               |           |
  ------|---------|---------|----- ... -----|-----------|----
        |   CVaR  |         |                           |
        | region  |  VaR    |                           |
        |<------->|  region |                           |
        |         |<------->|                           |

  CVaR = $4,100    VaR = $2,300     (95% of days are to the right)
  "Average loss     "Worst loss
   when things       on a normal
   go really bad"    bad day"
```

**Volatility** tells you how spread out the distribution is (overall bumpiness).
**VaR** marks the boundary of "normal bad" (95th percentile of losses).
**CVaR** measures the average of "really bad" (the left tail beyond VaR).

---

## Real-World Usage

| Who | Uses it for | Example |
|---|---|---|
| **Hedge funds** | Position sizing. If VaR exceeds risk budget, reduce positions. | "Our daily VaR limit is $500k. Current VaR is $480k. No new positions." |
| **Banks** | Regulatory capital. Basel III requires holding capital proportional to risk. | "Our CVaR requires $2M in reserve capital." |
| **Asset managers** | Client reporting. Show clients their portfolio risk profile. | "Your portfolio has 18% annualized volatility, 95% VaR of $12k." |
| **Risk desks** | Monitoring. Real-time alerts when risk exceeds thresholds. | "Alert: Portfolio VaR breached $1M limit at 2:15pm." |

This is exactly what FinSight does. Per-tenant risk predictions served via gRPC.

---

## Key Terms Summary (Interview Flashcard)

| Term | One-liner |
|---|---|
| **Return** | Percentage change in price. Log returns are additive and approximately normal. |
| **Volatility** | Std dev of returns. Annualize by multiplying daily vol by sqrt(252). |
| **VaR(95%, 1d)** | Max expected loss in 1 day, 95% confidence. 5th percentile of loss distribution. |
| **CVaR(95%, 1d)** | Average loss on the worst 5% of days. Always >= VaR. |
| **Subadditivity** | Diversification should reduce risk. CVaR has this property, VaR doesn't. |
| **Stationarity** | Statistical properties don't change over time. Returns are stationary, prices aren't. |
| **Covariance matrix** | Captures how assets move together. Drives portfolio-level risk calculation. |
| **Basel III** | Banking regulation. Shifted from VaR to CVaR (Expected Shortfall) for capital requirements. |

---

**Interview questions you can now answer:**
- "What is VaR and why is it flawed?"
- "How does CVaR fix VaR's weakness?"
- "Why log returns instead of simple returns?"
- "Why is portfolio volatility not the weighted average of individual volatilities?"
- "What regulatory framework requires these metrics?"
