# Technical Report: Shipment ETA Prediction & Delay Propagation System

**Safiri AI — AI Internship Take-Home Assignment**

---

## 1. Introduction

### 1.1 Problem Statement

In global logistics, delays at one stage of a shipment — late vessel departure, port congestion, or customs bottleneck — rarely remain isolated. They propagate downstream, compounding at each successive stage and ultimately amplifying the total delivery delay far beyond the initial disruption.

This project addresses three interconnected challenges:

1. **Predict the final delivery delay** of a shipment given upstream stage information
2. **Classify whether a shipment will experience a significant delay** (> 1 day)
3. **Explain which factors drive the predicted delay** — both globally and for individual shipments

### 1.2 Approach Summary

We adopt a **domain-driven machine learning approach** that explicitly models delay propagation dynamics. Rather than treating the final delay as an independent target, our feature engineering captures how delays cascade through the 4-stage shipment pipeline (Departure → Port Arrival → Customs Clearance → Final Delivery).

---

## 2. Data Description

### 2.1 Dataset Overview

We use a synthetic dataset of **250 shipment journeys** spanning 10 major trade lanes:

| Attribute           | Value                               |
| ------------------- | ----------------------------------- |
| Total shipments     | 250                                 |
| Trade lanes         | 10 (8 sea, 2 air)                   |
| Stages per shipment | 4 (Departure, Port, Customs, Final) |
| Missing data        | 9.2% of customs clearance records   |
| Delay rate (>1 day) | 66.4% of shipments                  |

### 2.2 Data Generation Assumptions

The synthetic data was generated with **explicit delay propagation**:

- Departure delay follows an exponential distribution (random operational events)
- Port delay depends on: departure delay × 0.30 + congestion × route_factor
- Customs delay depends on: port delay × 0.40 + weather impact
- Final delay depends on: customs delay × 1.00 + 0.2 × port + 0.1 × departure + noise

This cascading structure simulates the real-world phenomenon where upstream disruptions amplify through the supply chain.

### 2.3 Missing Data Strategy

Missing customs clearance data (~9%) was imputed using **route-corridor median delay** — a strategy motivated by the observation that customs processing times vary significantly between trade lanes (e.g., US CBP vs. Singapore Customs). A `customs_data_missing` indicator flag was created to preserve the information signal from missingness itself.

---

## 3. Feature Engineering

### 3.1 Design Philosophy

Every feature was designed to reflect **how logistics professionals reason about delays**, not arbitrary statistical transformations. We engineered 35 features across 7 categories:

### 3.2 Feature Categories

**Delay Propagation Features** (most important category):

- Cumulative delay at each stage (determines if downstream schedules are recoverable)
- Delay acceleration (is the disruption growing or being absorbed?)
- Propagation ratios (how much does each stage amplify the upstream delay?)

**Route Risk Features**:

- Sea routes receive a higher risk factor (1.5x) than air (0.5x), reflecting their greater exposure to congestion, weather, and canal bottlenecks
- Congestion × route_factor interaction captures the amplified impact on sea lanes

**Temporal Features**:

- Peak season indicator (Oct-Dec for pre-holiday surge; Jan-Feb for Chinese New Year disruption)
- Day of week (weekend departures may indicate scheduling irregularities)

**External Interaction Features**:

- Congestion × weather interaction models the non-linear compounding effect when multiple disruptions occur simultaneously (e.g., typhoon season + port congestion)

---

## 4. Modeling Strategy

### 4.0 Evaluation Setup

To ensure robust evaluation on our 250-record dataset, all models were evaluated using **5-fold cross-validation**. 
For the classification task, we used **Stratified 5-fold CV** to maintain the target class distribution (66.4% delayed vs. 33.6% on-time). Additionally, `class_weight="balanced"` was applied to our baseline Logistic Regression model to prevent the class imbalance from skewing predictions toward the majority class, ensuring reliable precision and recall metrics.

### 4.1 Regression: Predicting Delay Magnitude

**Target**: `delay_final_days` (continuous, in days)

We trained three models to compare linear vs. non-linear approaches:

| Model            | MAE (days) | RMSE (days) | R²    |
| ---------------- | ---------- | ----------- | ------ |
| Ridge Regression | 0.2658     | 0.3288      | 0.7483 |
| XGBoost          | 0.2789     | 0.3521      | 0.7113 |
| LightGBM         | 0.2916     | 0.3682      | 0.6843 |

**Observation**: Ridge Regression performs comparably to tree-based models. This is primarily because with a limited sample size of only **250 records**, highly parameterized tree-based models (XGBoost/LightGBM) are more prone to overfitting and struggle to fully leverage non-linear interactions without more data. Additionally, the underlying data generation process has a strong linear component (weighted sums of upstream delays), which Ridge captures effectively.

### 4.2 Classification: Delay Risk Prediction

**Target**: `is_delayed` (binary, delay > 1 day)

| Model               | Accuracy | F1 Score | AUC-ROC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 0.812    | 0.8498   | 0.8675  |
| LightGBM Classifier | 0.788    | 0.8418   | 0.8613  |

Both models achieve strong performance with AUC > 0.86, indicating good discrimination ability between on-time and delayed shipments.

---

## 5. Explainability Analysis

### 5.1 Global Feature Importance (SHAP)

The SHAP analysis reveals that **delay propagation features dominate** the prediction:

1. `delay_customs_days` — The strongest single predictor globally (highest mean |SHAP| value across the dataset)
2. `cumulative_delay_customs` — Total accumulated delay is the next strongest signal
3. `delay_port_days` — Port-level delays carry through to final delivery
4. `congestion_index` — The primary external driver of delays
5. `propagation_port_to_customs` — The rate of delay amplification between stages

### 5.2 Individual Shipment Explanation (Example)

For the most-delayed shipment (#65, final delay = 3.89 days):

| Factor                             | SHAP Impact | Interpretation                      |
| ---------------------------------- | ----------- | ----------------------------------- |
| Customs delay = 3.12 days          | +0.89 days  | Severe customs bottleneck           |
| Cumulative delay = 5.81 days       | +0.56 days  | Pipeline completely behind schedule |
| Port-to-customs propagation = 2.40 | +0.17 days  | Delay amplifying (not recovering)   |

### 5.3 Delay Propagation Analysis

*Note on Definition:* The "propagation ratio" here is defined as the **average observed delay amplification** (`median(delay_stage_n) / median(delay_stage_n-1)`). This observed ratio is often much larger than the baseline generation coefficients (e.g., 0.30) because it captures the *compounded reality* of the supply chain, including external shocks (port congestion, weather) that exponentially inflate the final observed delays between stages.

Stage-to-stage propagation ratios by transport mode:

| Transition        | Sea Routes | Air Routes |
| ----------------- | ---------- | ---------- |
| Departure → Port | 3.24x      | 2.71x      |
| Port → Customs   | 0.76x      | 1.03x      |
| Customs → Final  | 1.25x      | 1.23x      |

**Key insight**: Sea routes show higher initial delay amplification (3.24x at Departure → Port), heavily driven by external congestion effects compounding the initial departure delay. However, sea routes show partial recovery at the Port → Customs transition (0.76x), suggesting that port buffer times partially absorb delays. Both modes show final-stage amplification (~1.25x), indicating that last-mile delivery consistently adds to accumulated delays.

---

## 6. Discussion

### 6.1 Real-World Relevance

Our system reflects several real-world logistics behaviors:

1. **Cascading delays**: The model successfully captures how upstream disruptions propagate and amplify — the central challenge in supply chain ETA prediction
2. **Congestion as primary risk**: Port congestion is identified as the strongest external predictor, consistent with industry experience (e.g., the 2021 LA/Long Beach port crisis)
3. **Route-dependent vulnerability**: Sea routes are more delay-prone, which aligns with their longer transit times and exposure to weather, canal bottlenecks (Suez, Panama), and port capacity constraints
4. **Data quality as signal**: Missing customs data carries predictive value, reflecting the real-world correlation between poor tracking visibility and operational issues

### 6.2 Limitations

1. **Synthetic data**: While the propagation structure is realistic, real-world delays exhibit higher variance and more complex dependencies (multi-leg routings, transshipment, carrier-specific behavior)
2. **Static features**: The current model does not incorporate real-time signals (AIS vessel tracking, port queue APIs) that would improve mid-journey predictions
3. **Small sample size**: 250 records is a small dataset. This limits the model's ability to learn rare but impactful disruption patterns (e.g., canal closures) and explains why advanced tree-based models struggled to decisively outperform linear baselines without overfitting.

### 6.3 Future Work

- Integrate AIS vessel tracking data for real-time ETA updates
- Model multi-leg shipments with graph neural networks (GNNs)
- Add carrier-specific delay profiles
- Implement anomaly detection for "black swan" disruption events
- Build a real-time dashboard for supply chain control tower integration

---

## 7. Conclusion

This project demonstrates that ETA prediction is fundamentally a **systems-level problem** that requires understanding how delays emerge and propagate across multiple stages. By explicitly modeling delay cascading, leveraging domain-specific feature engineering, and providing SHAP-based explanations, our system moves beyond isolated predictions to deliver actionable insights into the causes and dynamics of supply chain delays.

---

*Submitted for: Safiri AI — AI Internship*
