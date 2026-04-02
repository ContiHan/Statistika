# Brainstorming: Forecast Horizon vs. Statistical Significance

**Date:** January 31, 2026
**Topic:** Relationship between Test Set Size ($N$), Forecast Horizon ($h$), and the Diebold-Mariano Test.

## 1. The Core Mathematical Constraint
When performing the **Diebold-Mariano (DM) test** with the **Harvey-Leybourne-Newbold (HLN) correction** (standard for small samples), a hard mathematical constraint applies to ensure stability:

$$ N \ge 2h $$

Where:
*   **$N$**: Total size of the Test Set (number of ground truth points available for comparison).
*   **$h$**: Forecast Horizon (how many steps ahead the model predicts).

If this condition is not met (i.e., if $N < 2h$), the correction formula involves the square root of a negative number, causing the test to fail.

## 2. The "Hold-out" Dilemma (Current Project Strategy)
In standard forecasting projects (especially with limited data like Macroeconomics), we often split data using a simple **Hold-out** strategy where the Test Set size equals the Forecast Horizon.

**Scenario:**
*   Test Set ($N$) = 6 years.
*   Desired Horizon ($h$) = 6 years.

**The Calculation:**
$$ 6 \ge 2 \times 6 \rightarrow 6 \ge 12 \rightarrow \mathbf{FALSE} $$

**Implication:**
We cannot statistically validate the model's performance on the *full* horizon because we only have *one* independent sample of that full horizon.

**The Solution ($h=1$):**
We must restrict the statistical test to **one-step-ahead ($h=1$)** accuracy.
$$ 6 \ge 2 \times 1 \rightarrow 6 \ge 2 \rightarrow \mathbf{TRUE} $$
This tells us: *"Is the model significantly better at predicting the next step?"* rather than *"Is the model significantly better at predicting the whole trajectory?"*

## 3. The "Ideal" Scenario (Rolling Test Set)
To statistically validate a longer horizon (e.g., $h=24$), we would need a Test Set much larger than the horizon ($N \gg h$).

**Example (BTC Hourly):**
*   **Desired Horizon ($h$):** 24 hours.
*   **Required Test Set:** At least $2 \times 24 = 48$ hours, ideally $5 \times 24 = 120+$ hours.
*   **Method:** Rolling/Sliding Window. The model predicts 24 hours, moves one hour forward, predicts again.
*   **Trade-off:** This requires "sacrificing" more recent data for testing, removing it from the training set.

## 4. Specific Dataset Strategy

### A. Macro Data (GDP, Investments, Forex) -> **Use h=1**
*   **Constraint:** Data is expensive (few rows). We cannot afford a large Rolling Test Set without ruining the training process.
*   **Status:** $N = h$.
*   **Action:** Set `h=1` for DM tests.

### B. BTC (High Frequency) -> **Use h=1 (for consistency)**
*   **Constraint:** We technically *could* afford a large Rolling Test Set, but to keep methodology consistent with Macro datasets, we use a Hold-out split.
*   **Status:** $N = h$.
*   **Action:** Set `h=1` for DM tests.

### C. Walmart (Multi-series) -> **Use h=Full Horizon**
*   **Constraint:** Test Set is short (28 days), but we have **5 parallel series**.
*   **Logic:** The statistical test pools all series together.
    *   $N = 28 \text{ days} \times 5 \text{ series} = 140 \text{ points}$.
    *   $h = 28 \text{ days}$.
*   **Calculation:** $140 \ge 56 \rightarrow \mathbf{TRUE}$.
*   **Action:** Safe to set `h=test_periods` (28).

## 5. Summary
*   **Training Horizon:** Always train the model to predict the full target length (e.g., 6 years, 24 hours).
*   **Statistical Horizon ($h$):
    *   **Single Series:** Must be **1**.
    *   **Multi-Series (Pooled):** Can be **Full Horizon** (if $Series \times Length \ge 2 \times Horizon$).
