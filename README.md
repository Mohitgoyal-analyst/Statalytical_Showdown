# Conversion Intelligence: E-Commerce Purchase Intent Prediction

> **Statalytical Showdown — Finalist Project**
> Binary classification case study on an e-commerce platform's session data

---

## Business Problem

Only **15.5% of website sessions result in a purchase**. The remaining 84.5% leave without buying.

This project answers three questions:

1. **Who converts?** — When and which visitors are most likely to buy?
2. **What signals intent?** — Which session behaviours predict a purchase?
3. **Can we predict it?** — Build a session-level conversion model

---

## Dataset

- **Source:** Statalytical Showdown competition dataset
- **Size:** 12,330 sessions × 18 features
- **Target:** `Revenue` — whether the session ended in a purchase
- **Class imbalance:** 84.5% No Purchase / 15.5% Purchase (5.5:1)
- **Missing values:** None
- **Duplicate rows:** 125 (not removed in this version)

---

## Project Structure

```
├── purchase_intention_model.ipynb           # Main notebook
├── statalytical showdown(in).csv            # Dataset
├── purchase_intention_model.pkl             # Saved Random Forest model
├── scaler.pkl                               # Saved RobustScaler
├── Statalytical_Showdown_Presentation.pptx  # Competition slides
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis

**Key findings:**

- November (25.4%) and October (20.9%) have the highest conversion, about 1.6x the 15.5% average. February is lowest at 1.6%.
- New visitors convert at about 25%, returning visitors at about 14% (returning visitors are about 86% of sessions).
- Better-supported traffic channels: Affiliate 27.7% (343 sessions), Community 25.3% (198), Organic Search 21.6% (3,913).
- Retargeting shows 33%, but this is only 3 sessions (1 purchase), so it is **insufficient data**, not a finding.
- Four traffic types show 0% conversion, but with only 1–38 sessions each, so they are small-sample artifacts.
- Sessions with PageValues > 0 convert at about 56%, versus about 4% when PageValues = 0.

### 2. Feature Engineering

7 features derived from session data:

| Feature | Description | Correlation with Revenue |
|---|---|---|
| `high_page_value` | Flag: PageValues > 0 | 0.60 (raw PageValues: 0.49) |
| `is_festive_month` | Oct or Nov session | 0.16 |
| `total_pages_visited` | Sum of all page type visits | 0.16 |
| `total_duration` | Total session time | 0.16 |
| `is_returning_visitor` | Encoded from VisitorType | -0.10 |
| `bounce_exit_ratio` | BounceRates / (ExitRates + 0.001) | -0.13 |
| `product_focus_ratio` | ProductRelated / (total_pages + 1) | 0.09 |

> `high_page_value` is a re-encoding of `PageValues`, so its higher correlation shows that a simple threshold works better than the skewed raw value.

### 3. Preprocessing

- Label encoding for Month, OS, Browser, Region, TrafficType
- Stratified 80/20 train-test split
- RobustScaler, fit on training data only, to avoid leakage

### 4. Class Imbalance Handling

- **SMOTE** (training data only) and **class weights** were compared.
- The test set keeps its original distribution (84.5 / 15.5).

### 5. Modelling

| Model | F1 (Purchase) | Precision | Recall |
|---|---|---|---|
| **Random Forest (SMOTE)** | **0.658** | **0.626** | **0.694** |
| LR (Class Weight) | 0.639 | 0.531 | 0.801 |
| LR (SMOTE) | 0.633 | 0.521 | 0.806 |
| XGBoost (Class Weight) | 0.632 | 0.598 | 0.670 |
| XGBoost (SMOTE) | 0.629 | 0.662 | 0.599 |
| RF (Class Weight) | 0.625 | 0.748 | 0.537 |

**Why Random Forest (SMOTE)?** It has the best F1 on this test split. Logistic Regression variants reach about 0.80 recall, but their precision is about 0.52, so nearly half of the flagged sessions would be false alarms. Random Forest catches 69.4% of real buyers while wrongly flagging 7.6% of non-buyers.

### 6. Model Explainability (SHAP)

Top features by mean |SHAP|:

1. `PageValues`
2. `high_page_value`
3. `is_festive_month`
4. `Administrative_Duration`
5. `ExitRates`
6. `is_returning_visitor`

---

## Business Recommendations

| # | Recommendation | Evidence |
|---|---|---|
| 01 | Concentrate marketing effort in Oct and Nov | Nov 25.4%, Oct 20.9% (about 1.6x average); Feb 1.6% |
| 02 | Prioritise well-supported channels (Affiliate, Community) and test retargeting properly | Affiliate 27.7% (n=343); retargeting has only 3 sessions |
| 03 | Test PageValues as an intent trigger | #1 SHAP feature; likely a late-session signal, so timing needs testing |
| 04 | Re-engage returning visitors differently | About 86% of sessions are returning visitors, converting at about 14% vs about 25% for new |
| 05 | Reduce exit rates on product pages | ExitRates is the #5 SHAP feature |

**Rough impact:** each +1 percentage point in conversion rate adds about 100 purchases per 10,000 sessions.

---

## Results Summary

- **Final model:** Random Forest Classifier with SMOTE
- **F1 (Purchase class):** 0.658
- **Precision:** 0.626
- **Recall:** 0.694
- **Accuracy:** 88.9%
- **Buyers correctly identified:** 265 of 382 (69.4%)
- **Non-buyers incorrectly flagged:** 158 of 2,084 (7.6%)

---

## Limitations

- Some categories have very few sessions (e.g. Retargeting: 3), so their rates are not reliable.
- The model was chosen on a single train/test split, without cross-validation; the top models are within about 0.02 F1.
- `PageValues` likely rises late in a session, so it may be a late-stage signal rather than an early trigger.
- Categorical features were label-encoded; one-hot encoding would be more appropriate.
- 125 duplicate rows were not removed.
- The notebook trains and evaluates a model; it does not include a deployed real-time system.

---

## Tech Stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, imbalanced-learn, SHAP, pickle

---

## Competition Note

This project was originally built for the **Statalytical Showdown** data science competition, where it reached the **finalist stage**. This repository is a rebuilt and extended version of that work using the same dataset, as the original submission files were not retained.

---

## Author

**Mohit**
B.Tech Mechanical Engineering, MNNIT Allahabad (2027)
GitHub: [Mohitgoyal-analyst](https://github.com/Mohitgoyal-analyst)
