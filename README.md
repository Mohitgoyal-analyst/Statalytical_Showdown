# Conversion Intelligence: E-Commerce Purchase Intent Prediction

> **Statalytical Showdown — Finalist Project**  
> Binary classification case study on an Indian e-commerce platform's session data

---

## Business Problem

An Indian e-commerce platform sees only **15.5% of website sessions result in a purchase**. The remaining 84.5% leave without buying — a massive revenue opportunity sitting untapped.

This project answers three business questions:

1. **Who converts?** — When and which visitors are most likely to buy?
2. **What signals intent?** — Which session behaviours predict a purchase?
3. **Can we predict it?** — Build a real-time session-level conversion predictor to trigger targeted interventions

---

## Dataset

- **Source:** Statalytical Showdown competition dataset (Indian e-commerce platform)
- **Size:** 12,330 sessions × 18 features
- **Target:** `Revenue` — whether the session resulted in a purchase (True/False)
- **Class Imbalance:** 84.5% No Purchase / 15.5% Purchase (5.5:1 ratio)
- **Missing Values:** None

**Features include:**
- Session behaviour: pages visited, time spent, bounce rates, exit rates, page values
- Session context: month, traffic type, visitor type, OS, browser, region, weekend flag
- Proximity: special day indicator

---

## Project Structure

```
├── statalytical_showdown.ipynb   # Main notebook
├── statalytical_showdown_in_.csv # Dataset
├── purchase_intention_model.pkl  # Saved Random Forest model
├── scaler.pkl                    # Saved RobustScaler
├── charts/                       # All EDA and model visualisations
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis
- Distribution analysis across all numeric and categorical features
- Buyer vs non-buyer behavioural comparison (KDE plots)
- Conversion rate breakdown by month, traffic type, visitor type, region
- Correlation heatmap identifying multicollinearity and feature-target relationships

**Key EDA Findings:**
- November (25.4%) and October (21.0%) are peak conversion months — 2.5x the annual average
- Retargeting traffic converts at 33% — double the platform average
- New visitors convert at 25% vs returning visitors at only 14%
- PageValues is near-zero for almost all non-buyers — strongest raw signal in the dataset
- 4 traffic types showed 0% conversion due to insufficient sample size (1–38 sessions), not genuine zero-conversion channels

### 2. Feature Engineering

7 new features derived from business logic:

| Feature | Description | Correlation with Revenue |
|---|---|---|
| `high_page_value` | Binary flag: PageValues > 0 | **0.60** (stronger than raw PageValues at 0.49) |
| `is_festive_month` | Oct or Nov session | 0.16 |
| `total_pages_visited` | Sum of all page type visits | 0.16 |
| `total_duration` | Total session time across all page types | 0.16 |
| `is_returning_visitor` | Encoded from VisitorType | -0.10 |
| `bounce_exit_ratio` | BounceRates / (ExitRates + 0.001) | -0.13 |
| `product_focus_ratio` | ProductRelated / (total_pages + 1) | 0.09 |

> **Notable:** `high_page_value` (engineered binary flag) achieved 0.60 correlation with Revenue — stronger than the raw `PageValues` feature (0.49). Two of the top three SHAP predictors are engineered features.

### 3. Preprocessing

- **Label encoding** for categorical features (Month, OS, Browser, Region, TrafficType)
- **Train-test split** (80/20, stratified to preserve class ratio)
- **RobustScaler** applied — chosen over StandardScaler due to heavy right skew and outliers in session duration features
- **Data leakage prevention:** scaler fit only on training data, then applied to test set

### 4. Class Imbalance Handling

Two strategies compared:
- **SMOTE** — generates synthetic minority class samples by interpolating between existing ones
- **Class weights** (`class_weight='balanced'`) — penalises minority class misclassification during training

> SMOTE applied only to training data. Test set kept at original distribution (84.5/15.5) to ensure honest evaluation.

### 5. Modelling

Six models evaluated across both imbalance strategies:

| Model | F1 (Purchase) | Precision | Recall |
|---|---|---|---|
| **Random Forest (SMOTE) ✓** | **0.658** | **0.626** | **0.694** |
| LR (Class Weight) | 0.639 | 0.531 | 0.801 |
| LR (SMOTE) | 0.633 | 0.521 | 0.806 |
| XGBoost (Class Weight) | 0.632 | 0.598 | 0.670 |
| XGBoost (SMOTE) | 0.629 | 0.662 | 0.599 |
| RF (Class Weight) | 0.625 | 0.748 | 0.537 |

**Why Random Forest (SMOTE)?**
Logistic Regression variants achieve higher recall (0.80+) but precision drops to ~0.52 — meaning nearly half of flagged sessions are false positives. For a real-time discount/intervention trigger, this level of false positives has a direct budget cost. Random Forest (SMOTE) offers the best F1 balance: catching 69.4% of real buyers while incorrectly flagging only 7.6% of non-buyers.

### 6. Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) used to explain individual predictions and identify global feature importance.

**Top predictors (SHAP):**
1. `PageValues` — dominant signal; sessions with PageValues > 0 are overwhelmingly buyers
2. `high_page_value` — engineered binary flag; second strongest predictor overall
3. `is_festive_month` — engineered; festive months significantly boost conversion probability
4. `ExitRates` — strong negative signal; high exit rate pushes strongly away from purchase
5. `is_returning_visitor` — returning visitors are slightly less likely to convert (counterintuitive but confirmed)

---

## Business Recommendations

| # | Recommendation | Evidence |
|---|---|---|
| 01 | **Front-load marketing spend in Oct & Nov** | Nov: 25.4%, Oct: 21.0% — 2.5x platform average. Feb: 1.6%. |
| 02 | **Prioritise retargeting traffic channel** | Retargeting converts at 33% vs platform average of 15.5% |
| 03 | **Use PageValues as a real-time intent trigger** | #1 SHAP predictor — fire discount popup or live chat when PageValue > 0 |
| 04 | **Re-engage returning visitors differently** | 86% of sessions are returning visitors, but they convert at only 14% vs 25% for new |
| 05 | **Reduce exit rates on product pages** | ExitRates = #4 SHAP predictor with strong negative impact on conversion |

**Expected Impact:** A 3% improvement in conversion rate adds ~370 additional purchases per 10,000 sessions.

---

## Results Summary

- **Final Model:** Random Forest Classifier with SMOTE
- **F1 Score (Purchase class):** 0.658
- **Precision:** 0.626
- **Recall:** 0.694
- **Overall Accuracy:** 88.9%
- **Buyers correctly identified:** 265 out of 382 (69.4%)
- **Non-buyers incorrectly flagged:** 158 out of 2,084 (7.6%)

---

## Tech Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **Scikit-learn** — preprocessing, modelling, evaluation
- **XGBoost** — gradient boosting classifier
- **imbalanced-learn** — SMOTE implementation
- **SHAP** — model explainability
- **Pickle** — model serialisation

---

## Competition Note

This project was originally built for the **Statalytical Showdown** data science competition, where it reached the **finalist stage**. This repository is a rebuilt and extended version of that work using the same dataset, as the original submission files were not retained. The analysis and modelling approach have been revisited and improved using current skills.

---

## Author

**Mohit**  
B.Tech Mechanical Engineering, MNNIT Allahabad (2027)  
GitHub: [Mohitgoyal-analyst](https://github.com/Mohitgoyal-analyst)
