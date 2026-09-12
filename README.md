## Project Overview

This project analyzes two years (2023–2024) of transaction-level retail data across 8 stores and 40 products to answer real business questions: data quality issues, category performance by store type, the relationship between marketing spend and sales, margin analysis, sales volatility, and seasonal trends.

## Dataset

| File | Description |
|---|---|
| `transactions.csv` | Transaction-level sales data — `transaction_id`, `date`, `store_id`, `product_id`, `quantity`, `unit_price`, `discount_pct`, `sales_value` |
| `stores.csv` | Store reference data — `store_id`, `city`, `store_type` (Flagship / Mall / Standalone) |
| `products.csv` | Product reference data — `product_id`, `category`, `unit_cost`, `unit_price` |
| `marketing_spend.csv` | Weekly marketing spend — `week_start`, `marketing_spend` |

The raw transaction data intentionally includes duplicate rows, missing values, and a few data-integrity mismatches to practice real-world cleaning.

## Tech Stack

- Python 3
- pandas
- NumPy

## Setup

```bash
pip install pandas numpy
```

```python
import pandas as pd
import numpy as np

transactions = pd.read_csv('transactions.csv')
products = pd.read_csv('products.csv')
stores = pd.read_csv('stores.csv')
marketing_spend = pd.read_csv('marketing_spend.csv')

transactions['date'] = pd.to_datetime(transactions['date'])
marketing_spend['week_start'] = pd.to_datetime(marketing_spend['week_start'])
```

## Analysis

### A. Data Cleaning

**1. Find and remove duplicate transactions**

Identified duplicate rows using a business-key subset rather than `transaction_id`, since two rows can share a distinct ID and still represent the same double-logged sale.

```
transactions = transactions.drop_duplicates(
    subset=['date','store_id','product_id','quantity','unit_price','discount_pct'],
    keep='first'
).reset_index(drop=True)
transactions
```

**2. Handle missing quantity**

Checked the proportion of missing `quantity` values; rows dropped rather than imputed, since the missing rate was under 1% with no evident pattern by store or product, so dropping loses negligible signal versus guessing a value that would distort `sales_value`.

```
transactions = transactions.dropna(subset=['quantity']).reset_index(drop=True)

transactions
```

**3. Recompute sales_value and flag mismatches**

Recalculated `sales_value` as `quantity × unit_price × (1 − discount_pct)` and compared against the stored value, treating the recomputed value as the source of truth.

```
transactions['sales_value_check'] = (
    transactions['quantity'] * transactions['unit_price'] * (1 - transactions['discount_pct'])
).round(2)

mismatches = transactions[
    (transactions['sales_value'] - transactions['sales_value_check']).abs() > 0.01
]

transactions['sales_value'] = transactions['sales_value_check']
transactions = transactions.drop(columns='sales_value_check')

transactions

```

### B. Joins

**4. Best-selling category by store type**

Merged `transactions` with `products` (for `category`) and `stores` (for `store_type`), grouped by both, and identified the top category by revenue within each store type.

```python
tx_full = transactions.merge(products[['product_id','category']], on='product_id', how='left') \
                       .merge(stores[['store_id','store_type']], on='store_id', how='left')

category_by_type = tx_full.groupby(['store_type','category'], as_index=False)['sales_value'].sum()

best_per_type = category_by_type.loc[
    category_by_type.groupby('store_type')['sales_value'].idxmax()
]
```

**5. Marketing spend vs. weekly sales correlation**

Bucketed each transaction date to its week's Monday, aggregated to weekly revenue, joined against `marketing_spend` on that shared weekly key, and computed the correlation.

```python
transactions['week_start'] = transactions['date'] - pd.to_timedelta(transactions['date'].dt.weekday, unit='D')

weekly_sales = transactions.groupby('week_start', as_index=False)['sales_value'].sum()
weekly = weekly_sales.merge(marketing_spend, on='week_start', how='inner')

correlation = weekly['sales_value'].corr(weekly['marketing_spend'])

```

`how='inner'` is used here rather than `left` — if a week exists in transactions but `marketing_spend` doesn't have that exact Monday (edge weeks at the start/end of the dataset), NaNs would distort the correlation calculation.

### C. Groupby & Aggregation

**6. Revenue and margin by category**

Merged in `unit_cost`, computed margin, and aggregated total revenue, total margin, and margin percentage per category.

```python
tx_cost = transactions.merge(products[['product_id','category','unit_cost']], on='product_id', how='left')
tx_cost['margin'] = tx_cost['sales_value'] - (tx_cost['quantity'] * tx_cost['unit_cost'])

category_summary = tx_cost.groupby('category', as_index=False).agg(
    total_revenue=('sales_value','sum'),
    total_margin=('margin','sum')
)
category_summary['margin_pct'] = (category_summary['total_margin'] / category_summary['total_revenue'] * 100).round(1)
category_summary.sort_values('total_revenue', ascending=False
```

**7. Most volatile store (coefficient of variation)**

Aggregated to daily revenue per store, then compared stores using coefficient of variation (`std ÷ mean`) rather than raw standard deviation, to normalize for stores with naturally larger average sales.

```python
daily_store_sales = transactions.groupby(['store_id','date'], as_index=False)['sales_value'].sum()

volatility = daily_store_sales.groupby('store_id')['sales_value'].agg(['mean','std'])
volatility['cv'] = volatility['std'] / volatility['mean']
```

**8. Top 3 products by revenue, per store**

```python
product_store_rev = transactions.groupby(['store_id','product_id'], as_index=False)['sales_value'].sum()

top3_per_store = (
    product_store_rev.sort_values(['store_id','sales_value'], ascending=[True, False])
    .groupby('store_id')
    .head(3)
)
top3_per_store
```

### D. Time Series

**9. Monthly revenue trend, MoM and YoY change**

```python
monthly = transactions.set_index('date').resample('ME')['sales_value'].sum().reset_index()
monthly['mom_pct_change'] = monthly['sales_value'].pct_change() * 100
monthly['yoy_pct_change'] = monthly['sales_value'].pct_change(12) * 100
monthly
```

**10. Seasonal spike validation**

Compared average daily revenue in November–December against the rest of the year to quantify the size of the holiday season lift.

```python
transactions['month'] = transactions['date'].dt.month
daily_totals = transactions.groupby('date', as_index=False)['sales_value'].sum()
daily_totals['is_holiday_season'] = daily_totals['date'].dt.month.isin([11,12])

comparison = daily_totals.groupby('is_holiday_season')['sales_value'].mean()
print(comparison)

lift_pct = (comparison[True] / comparison[False] - 1) * 100

```

**11. 7-day rolling average and outlier days**

Computed a 7-day rolling average of daily revenue to smooth day-of-week noise, then identified the single best and worst day relative to that rolling baseline.

```python
daily_totals = daily_totals.sort_values('date').set_index('date')
daily_totals['rolling_7d'] = daily_totals['sales_value'].rolling(7).mean()
daily_totals['diff_from_rolling'] = daily_totals['sales_value'] - daily_totals['rolling_7d']

best_day = daily_totals['diff_from_rolling'].idxmax()
worst_day = daily_totals['diff_from_rolling'].idxmin()

```

### E. Capstone

**12. Diagnosing the 2 underperforming stores (YoY revenue decline)**

Combined groupby, merges, and time-series comparison across three candidate explanations — category mix shift, marketing spend, and discount usage — to diagnose why the two lowest year-over-year growth stores are underperforming.

*Stage 1: Identify the 2 underperforming stores by YoY revenue growth*

```python
transactions['year'] = transactions['date'].dt.year

store_year_rev = transactions.groupby(['store_id','year'], as_index=False)['sales_value'].sum()

# pivot so 2023 and 2024 sit side by side
rev_pivot = store_year_rev.pivot(index='store_id', columns='year', values='sales_value')
rev_pivot['yoy_growth_pct'] = (rev_pivot[2024] - rev_pivot[2023]) / rev_pivot[2023] * 100

worst_2 = rev_pivot.sort_values('yoy_growth_pct').head(2)
worst_2
```

*Stage 2: Check for a category mix problem*

```python
tx_products = transactions.merge(products[['product_id','category']], on='product_id', how='left')

category_mix = tx_products.groupby(['store_id','year','category'], as_index=False)['sales_value'].sum()

worst_ids = worst_2.index.tolist()
category_mix_worst = category_mix[category_mix['store_id'].isin(worst_ids)]

# turn each category into a % share of that store-year's total, to compare shift over time
category_mix_worst['pct_of_store_year'] = category_mix_worst.groupby(['store_id','year'])['sales_value'].transform(lambda x: x / x.sum() * 100)
category_mix_worst.sort_values(['store_id','year','pct_of_store_year'], ascending=[True,True,False]
```

*Stage 3: Check for a marketing spend problem*

`marketing_spend` has no `store_id` — it's tracked at the whole-business level, so store-level marketing attribution isn't possible with this data. As a proxy, the overall company-wide marketing-to-revenue correlation (see Q5) can be checked for a weak or flat relationship, which would weaken (without fully ruling out) marketing as the explanation for these specific stores.

*Stage 4: Check for a discount-usage problem*

```python
discount_behavior = transactions[transactions['store_id'].isin(worst_ids)].groupby(['store_id','year'], as_index=False).agg(
    avg_discount=('discount_pct','mean'),
    pct_transactions_discounted=('discount_pct', lambda x: (x > 0).mean() * 100)
)
discount_behavior
```

A sharp rise in `avg_discount` or `pct_transactions_discounted` from 2023→2024 for these stores would indicate stable unit sales but eroded price/margin — a real driver of weak revenue growth even without a volume drop. Comparing these figures against the company-wide average distinguishes a store-specific issue from a general trend.

*Stage 5: Conclusion*

> Fill in after running Stages 1–4 — state which of category mix, marketing, or discount usage showed the clearest signal for each of the two underperforming stores, and note that store-level marketing attribution isn't possible due to the data gap in `marketing_spend`.

## Key Findings

> Fill in after running the analysis — e.g. top-performing category by store type, the marketing-spend correlation coefficient, the most volatile store, and the measured size of the Nov/Dec seasonal lift.

## Notes / Limitations

- `marketing_spend` is tracked at the whole-business level, not per store, so store-level marketing attribution isn't possible with this dataset.
- The Nov/Dec marketing spend increase overlaps with the Nov/Dec seasonal sales lift, which can inflate the marketing-to-sales correlation if seasonality isn't accounted for separately.
