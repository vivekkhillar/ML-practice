# Extracting Insights from Data with `describe()` and Beyond

A practical guide for understanding and interpreting statistical summaries from any dataset.

---

## Table of Contents
1. [What `describe()` Returns](#what-describe-returns)
2. [How to Get Insights](#how-to-get-insights)
3. [Titanic Dataset Example](#titanic-dataset-example)
4. [Universal Logic for Any Dataset](#universal-logic-for-any-dataset)
5. [Mental Model](#mental-model)
6. [Universal Formulas](#universal-formulas)
7. [Universal Checklist](#universal-checklist)
8. [Going Deeper](#going-deeper)

---

## What `describe()` Returns

`describe()` gives you summary statistics for numeric columns:

| Statistic | Meaning |
|-----------|---------|
| **count** | Number of non-null values (helps spot missing data) |
| **mean** | Average value |
| **std** | Standard deviation (spread of values) |
| **min** | Smallest value |
| **25%** | 25th percentile (Q1) |
| **50%** | Median (middle value) |
| **75%** | 75th percentile (Q3) |
| **max** | Largest value |

---

## How to Get Insights

### 1. Missing Data
Compare **count** across columns. If one column has a lower count than others, you have missing values.

### 2. Skewness & Outliers
- **Mean vs Median**: If mean >> median → right-skewed (high outliers). If mean << median → left-skewed.
- **IQR Rule**: `IQR = 75% - 25%`. Values beyond `Q1 - 1.5×IQR` or `Q3 + 1.5×IQR` are often outliers.

### 3. Scale & Spread
- **std**: Larger = more spread.
- **min–max**: Shows full range.

### 4. Target Variable (for ML)
- **mean** of binary target = proportion of positive class (e.g., survival rate).

---

## Titanic Dataset Example

### Sample `describe()` Output

|       | survived | pclass | age    | sibsp | parch | fare    |
|-------|----------|--------|--------|-------|-------|---------|
| count | 891      | 891    | 714    | 891   | 891   | 891     |
| mean  | 0.38     | 2.31   | 29.70  | 0.52  | 0.38  | 32.20   |
| std   | 0.49     | 0.84   | 14.53  | 1.10  | 0.81  | 49.69   |
| min   | 0        | 1      | 0.42   | 0     | 0     | 0       |
| 25%   | 0        | 2      | 20.13  | 0     | 0     | 7.91    |
| 50%   | 0        | 3      | 28     | 0     | 0     | 14.45   |
| 75%   | 1        | 3      | 38     | 1     | 0     | 31.00   |
| max   | 1        | 3      | 80     | 8     | 6     | 512.33  |

### Column-by-Column Insights

**survived**: mean 0.38 → ~38% survived. Slight class imbalance.

**pclass**: median 3 → most passengers in 3rd class. Strong predictor candidate.

**age**: count 714 vs 891 → ~20% missing. Need imputation. Mean ≈ median → roughly symmetric.

**sibsp / parch**: median 0 → most travel alone or with few family. Max 8 and 6 → outliers.

**fare**: mean 32 vs median 14.5 → right-skewed. Max 512 → high outliers. Min 0 → investigate.

### Summary of Actions
- Impute age (~20% missing)
- Consider log transform for fare (skewed)
- Investigate fare = 0
- Use pclass as key feature
- Consider "alone" or "family size" features

---

## Universal Logic for Any Dataset

### The Core Question: "What Could Go Wrong?"

| Question | Why It Matters |
|----------|----------------|
| **Are values missing?** | Models can't use them; need imputation or dropping |
| **Are values plausible?** | min/max can reveal data entry errors |
| **Is the distribution skewed?** | Skewed features may need log transform or binning |
| **Are there outliers?** | Can distort models or need special handling |
| **Is the scale weird?** | Some models need normalization |
| **Is the target imbalanced?** | Affects which metrics to use and model behavior |

### Universal Rules from `describe()`

**Rule 1: Count vs Total Rows**
```
If count < total_rows for any column → Missing data
```
- < 5% missing: Often fine to drop or simple imputation
- 5–20% missing: Need imputation strategy
- > 20% missing: Consider if column is worth keeping

**Rule 2: Mean vs Median**
```
If mean >> median  → Right-skewed (long right tail, high outliers)
If mean << median  → Left-skewed (long left tail, low outliers)
If mean ≈ median   → Roughly symmetric
```
- Skewed → consider log, sqrt, or binning
- Symmetric → mean/median imputation is usually fine

**Rule 3: Standard Deviation**
```
If std ≈ 0        → Almost constant (little information)
If std very large → High variance (outliers or wide range)
```
- Constant columns: usually drop
- High std: check for outliers and scaling

**Rule 4: Min/Max Sanity Check**
```
min < 0 when it shouldn't?  → Data error
max = 999 or 9999?         → Possible placeholder for missing
min = max?                 → Constant column
```

**Rule 5: Percentiles vs Mean**
```
If 75% - 25% (IQR) is small but range is large → Outliers at extremes
```
- Use IQR to define "typical" range
- Values outside Q1 − 1.5×IQR or Q3 + 1.5×IQR are often outliers

---

## Mental Model

Think in three layers:

```
Layer 1: DATA QUALITY
├── Missing values (count)
├── Impossible values (min, max)
└── Placeholders (e.g., 999, -1)

Layer 2: DISTRIBUTION
├── Shape (mean vs median → skewness)
├── Spread (std, IQR)
└── Outliers (IQR rule, extreme percentiles)

Layer 3: RELATIONSHIPS (beyond describe)
├── Correlations with target
├── Correlations between features (multicollinearity)
└── Group differences (by categorical variables)
```

---

## Universal Formulas

| Insight | Formula / Logic |
|---------|-----------------|
| Missing % | `(total - count) / total × 100` |
| Skewness | `mean - median` (positive → right skew) |
| Outlier bounds | `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` where `IQR = Q3 - Q1` |
| Coefficient of variation | `std / mean` (relative spread; use when mean ≠ 0) |
| Zero variance | `std == 0` → drop column |

---

## Universal Checklist

```
□ For each numeric column:
  □ Compare count to total rows → missing?
  □ Compare mean vs median → skew?
  □ Check min/max → plausible?
  □ Compute IQR → outliers?

□ For target (if present):
  □ Check balance (mean for 0/1, or value_counts for multiclass)
  □ If imbalanced → adjust metrics (precision, recall, F1, AUC)

□ For scaling:
  □ Compare std across columns → need normalization?

□ For feature engineering:
  □ Skewed → log/sqrt?
  □ High cardinality → binning?
  □ Interactions → product/ratio of related features?
```

---

## Going Deeper

1. **Always compare groups** — Use `groupby` on categorical columns and compare `describe()` or `mean()` of numeric/target columns.

2. **Use percentiles** — They're robust to outliers. Median = 50th percentile. IQR = 75th − 25th. Use percentiles for binning.

3. **Combine stats** — High std + low mean = high relative variability. Low std + high mean = stable, high values.

4. **Think downstream** — Missing → imputation. Skewed → transform. Outliers → cap, clip, or robust methods. Imbalanced → sampling or class weights.

5. **Cross-check with `info()`** — `df.info()` shows dtypes and non-null counts. Use both `info()` and `describe()` together.

### Quick Code Snippets

```python
# Include categorical columns in describe
df.describe(include='all')

# Compare survivors vs non-survivors
df.groupby('survived').describe()

# Correlation with target
df[numeric_cols].corr()['target']

# Survival by group
df.groupby('pclass')['survived'].mean()
```

---

## One-Line Summary

**For any dataset:** Use `count` for missingness, `mean` vs `median` for skew, `std` and `min`/`max` for spread and outliers, and percentiles for robust summaries. Then ask: *"What could break my model or analysis?"* and fix those first.
