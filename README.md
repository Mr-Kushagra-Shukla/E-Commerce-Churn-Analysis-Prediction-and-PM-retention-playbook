# E-Commerce-Churn-Analysis-Prediction-and-PM-retention-playbook
Predicts customer churn in e-commerce using Random Forest &amp; Decision Tree. Features SHAP-based model interpretability, churn risk tier segmentation, and a PM retention playbook translating ML outputs into actionable business strategy.
# E-Commerce Customer Churn Prediction
### ML model with SHAP-based interpretability and a PM-focused retention playbook

---

## Project Goal

Predict which e-commerce customers are likely to churn — and more importantly, **explain why** and **what the business should do about it**. This project goes beyond model accuracy to translate ML outputs into actionable product strategy, structured around a 3-layer interpretation framework used in product management.

---

## Dataset Overview

| Property | Detail |
|---|---|
| Source | E-Commerce customer dataset (`E_comm.xlsx`) |
| Total records | 5,630 customers |
| Features | 19 (after removing CustomerID) |
| Target variable | `Churn` (1 = churned, 0 = retained) |
| Class distribution | Imbalanced — addressed via `class_weight='balanced'` |

**Feature categories:**
- **Behavioral** — Tenure, OrderCount, DaySinceLastOrder, HourSpendOnApp, CouponUsed
- **Transactional** — CashbackAmount, OrderAmountHikeFromlastYear, WarehouseToHome
- **Categorical** — PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus
- **Engagement** — SatisfactionScore, Complain, NumberOfDeviceRegistered, NumberOfAddress

---

## Process Walkthrough

### 1. Exploratory Data Analysis (EDA)

**Target variable check:**
A count plot of `Churn` confirmed class imbalance — significantly more non-churners than churners. This informed the use of `class_weight='balanced'` in all models.

**Key EDA finding (counterintuitive):**
Churned customers had a *shorter* median `DaySinceLastOrder` (~2 days) compared to non-churned customers (~4 days). This went against the standard e-commerce churn assumption that inactivity drives churn — signalling that **engagement quality matters more than recency alone** in this dataset.

**Univariate analysis:**
- Bar plots and density histograms across 6 categorical variables vs Churn
- Pie charts for CityTier, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore distributions
- Box plots and violin plots for all 12 numerical features, split by Churn class

**Bivariate analysis:**
- Scatter plots of all numerical features vs CashbackAmount, colored by Churn
- Correlation heatmap across all numerical features

**Scatter plot insight:** Behavioral variables (purchase frequency, engagement) showed a relationship with churn. Demographic and monetary variables showed little association. Churn in this dataset is primarily driven by **declining engagement**, not customer characteristics or spending power.

---

### 2. Data Preprocessing

**Missing value treatment** (imputed with training set median — no data leakage):

| Feature | Missing % |
|---|---|
| DaySinceLastOrder | 5.45% |
| OrderAmountHikeFromlastYear | 4.71% |
| Tenure | 4.69% |
| CouponUsed | 4.55% |
| OrderCount | 4.58% |
| HourSpendOnApp | 4.53% |
| WarehouseToHome | 4.46% |

**Categorical encoding:**
Label encoding applied to 5 object-type columns: PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus.

**Train/Test split:** 80% train / 20% test (`random_state=42`)

**Outlier treatment (applied to training set only):**
- IQR removal for `WarehouseToHome` and `DaySinceLastOrder` (extreme outliers removed)
- Percentile capping (1st–90th) for `CouponUsed`, `OrderCount`, `CashbackAmount` (long-tail variables)

After treatment — zero missing values confirmed in both train and test sets.

---

### 3. Feature Engineering

Three new features were engineered to capture behavioral ratios not present in raw data:

| Feature | Formula | Business meaning |
|---|---|---|
| `engagement_score` | `OrderCount / (Tenure + 1)` | How frequently a customer orders relative to how long they've been with us |
| `complaint_rate` | `Complain / (OrderCount + 1)` | Complaints per order — a quality signal |
| `recency_ratio` | `DaySinceLastOrder / (Tenure + 1)` | How recently they ordered relative to their total tenure |

These engineered features became some of the strongest predictors in the final model.

---

### 4. Model Building

Two models were built and compared, both with `class_weight='balanced'` to handle class imbalance, and hyperparameters tuned via 5-fold GridSearchCV.

#### Decision Tree
**Best hyperparameters:** `max_depth=8`, `min_samples_leaf=1`, `min_samples_split=2`, `random_state=42`

| Metric | Score |
|---|---|
| Accuracy | 87.66% |
| Precision | 58.27% |
| Recall | 87.57% |
| F1 Score | 69.98% |
| Jaccard Score | 53.82% |
| Log Loss | 4.449 |

#### Random Forest ✅ (Final Model)
**Best hyperparameters:** `n_estimators=200`, `max_depth=None`, `max_features='sqrt'`, `random_state=42`
*(Model trained with n_estimators=100 for efficiency)*

| Metric | Score |
|---|---|
| Accuracy | **97.87%** |
| Precision | **99.39%** |
| Recall | **87.57%** |
| F1 Score | **93.10%** |
| Jaccard Score | **87.10%** |
| AUC Score | **0.988** |
| Log Loss | 0.768 |

**Why Random Forest over Decision Tree:**
Random Forest improved accuracy by ~10 percentage points and precision by ~41 points. The ensemble of 100 trees reduces variance and avoids overfitting to noise — making it far more reliable for real-world deployment.

---

### 5. Model Interpretation (3-Layer Framework)

#### Layer 1 — What does the model globally care about? (Feature Importance)

Top 5 churn drivers identified by Random Forest feature importance:

| Rank | Feature | Business meaning |
|---|---|---|
| 1 | `Tenure` | How long the customer has been with us |
| 2 | `engagement_score` | Order frequency relative to tenure *(engineered)* |
| 3 | `CashbackAmount` | Cashback received — a loyalty proxy |
| 4 | `recency_ratio` | Recency relative to tenure *(engineered)* |
| 5 | `WarehouseToHome` | Delivery distance |

Two of the top 5 features were **engineered features** — validating the feature engineering step.

#### Layer 2 — Why is this specific customer at risk? (SHAP Analysis)

SHAP (SHapley Additive exPlanations) was used for individual-level explainability. Computed on a 300-customer sample of the test set.

**Highest-risk customer identified: 99.0% churn probability**

Risk factors pushing this customer toward churn:
- `Tenure = 1.0` — Very new customer (low switching cost)
- `engagement_score = 3.5` — Unusually high engagement suddenly — could signal final binge before leaving
- `Complain = 1.0` — Active complaint on record

Protective factors reducing churn risk:
- `DaySinceLastOrder = 8.0` — Relatively recent purchase
- `HourSpendOnApp = 4.0` — Still spending time on the app

SHAP base value (average churn probability across training data): **0.5002**

#### Layer 3 — What does the business do? (Risk Tier Segmentation)

Customers scored and bucketed into 3 actionable tiers:

| Risk Tier | Customers | Avg Churn Probability | Actual Churn Rate |
|---|---|---|---|
| 🔴 High Risk (p ≥ 0.70) | 122 (10.8%) | 83.7% | **100%** |
| 🟡 Medium Risk (0.40 ≤ p < 0.70) | 54 (4.8%) | 58.3% | 87.0% |
| 🟢 Low Risk (p < 0.40) | 950 (84.4%) | 4.8% | 1.7% |

The tier boundaries proved highly accurate — every High Risk customer in the test set actually churned (100% actual churn rate).

---

### 6. PM Retention Playbook

| Tier | Actions |
|---|---|
| 🔴 High Risk | Proactive support outreach within 24hrs · Personalized win-back discount · Reduce checkout friction · Escalate high-LTV accounts |
| 🟡 Medium Risk | Personalized push notifications · Enroll in loyalty/rewards program · 1-question satisfaction pulse survey · Re-engagement email campaign |
| 🟢 Low Risk | Deepen engagement via reviews and referrals · Surface upsell/premium tier · Monitor for behavioral drop-off · Use as control group in A/B tests |

---

## Final Results Summary

| | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | 87.66% | **97.87%** |
| Precision | 58.27% | **99.39%** |
| F1 Score | 69.98% | **93.10%** |
| AUC | — | **0.988** |

**Business impact (30% retention save rate assumption):**
- 1,126 customers scored in test set
- 122 high-risk customers identified (10.8% of base)
- ~36 customers retained per cycle at 30% save rate

**Core PM thesis:**
> Tenure and complaint behavior are the strongest early-warning signals. A low-tenure customer with an active complaint is 3–4x more likely to churn. Intervene before day 30 of their lifecycle. Use p ≥ 0.70 for high-cost interventions (personal outreach, discounts) and p ≥ 0.40 for low-cost nudges (push notifications, emails). Adjust thresholds based on your LTV vs CAC ratio.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models, preprocessing, evaluation |
| matplotlib / seaborn | Visualization |
| SHAP 0.49.1 | Model interpretability |
| Jupyter Notebook | Development environment |

---

## Repository Structure

```
ecommerce-churn-prediction-ml/
│
├── EcommerceCustomerChurnRatePredictionAndAnalysis.ipynb   # Main notebook
├── E_comm.xlsx                                              # Dataset
└── README.md                                               # This file
```

---

## Key Takeaways

1. **Engineered features outperformed raw features** — `engagement_score` and `recency_ratio` ranked in the top 5, above most original features
2. **Random Forest dramatically outperforms Decision Tree** — 10% accuracy gain, 41% precision gain
3. **Model is highly reliable for high-risk identification** — 100% actual churn rate in the High Risk tier on unseen test data
4. **Churn is engagement-driven, not demographics-driven** — intervention strategy should focus on behavioral signals, not customer profiles
5. **SHAP enables personalized retention** — each at-risk customer has a unique combination of risk factors, enabling targeted rather than blanket intervention

---

*Built as part of a product analytics portfolio. Structured with a PM interpretation framework — translating ML outputs into business strategy.*
