# Lab 4 — Company Bankruptcy Prediction (Decisions & Rationale)

## 1) Choosing the Initial Models
- **Benchmark:** Logistic Regression (LR) → calibrated, transparent baseline.
- **Add:** Random Forest (RF) → non-linear patterns, low prep.
- **Add:** XGBoost (XGB) → strong tabular lift; imbalance via `scale_pos_weight`.
- **Unsupervised:** not primary (labels available; weaker regulatory justification).

## 2) Data Pre-processing
- **Impute:** median (fit on **train** only) to avoid leakage.
- **Scale:** only LR (Standard/Robust) for stable optimization/calibration.
- **Trees/Boosting:** no scaling; split-based.
- **Keep valid tails;** clean only NaN/inf.

## 3) Handling Class Imbalance
- **Stratify** all splits/CV (preserves rare positive rate).
- **Weights:** LR/RF use `class_weight`; XGB uses `scale_pos_weight = neg/pos`.
- **SMOTE:** only if recall remains weak after weights (noise risk).
- **Threshold:** pick on validation (F1/Youden) using `predict_proba`.

## 4) Outlier Detection & Treatment
- **Keep plausible extremes** (distress signal).
- **RobustScaler** for LR; trees tolerate outliers.
- **Remove only impossibles** (NaN/inf/parse errors).
- **Light winsorize** only if calibration is distorted (check Brier).

## 5) Sampling Bias (Train vs Test)
- **PSI summary:** max PSI = **0.0209**; top drifting features = **Operating profit per person**, **Working Capital to Total Assets**, **Cash flow rate**.
- **Counts:** PSI ≥0.25 (action) = **0**; 0.10–0.25 (watch) = **0**.
- **Action:** split is **stable** (all features <0.10).
- **Why:** prevents inflated/fragile test metrics from distribution shift.

## 6) Data Normalization
- **LR only** (Standard/Robust) → stable coefficients & calibration.
- **RF/XGB:** no scaling required.
- **Scaling inside Pipeline** to prevent leakage.
- **Pick scaler** by Brier/calibration on validation.

## 7) Testing for Normality
- **Not required** for LR/RF/XGB predictive use.
- **Selective transforms** (log/yeo-johnson) only if LR calibration improves.
- **Decide by metrics** (Brier/Calibration), not p-values.
- **Preserve interpretability;** avoid blanket transforms.

## 8) PCA (Dimensionality Reduction)
- **Pros:** cuts multicollinearity/noise; can stabilize LR.
- **Cons:** loses feature meaning; may hide rare risk signals.
- **Decision:** no PCA for trees; optional small PCA for LR only if AUC/Brier improve.
- **Governance:** interpretability preferred unless gains are clear.

## 9) Feature Engineering
- **Minimal FE;** ratios already rich.
- **Optional signed-log** for a few heavy tails (LR only) if calibration helps.
- **No leakage** (no post-event info).
- **Consistency** across models for fair comparison.

## 10) Multicollinearity
- **Corr filter:** drop one of pairs with |ρ|>0.90 (train-based).
- **VIF>10 (LR):** remove/transform unstable features.
- **Trees tolerate** collinearity; prune only if importances unstable.
- **Re-check** AUC/Brier after pruning.

## 12) Feature Selection
- **Two-stage:** correlation filter → model-based (LR-L1 or XGB gain).
- **Balance:** too many ↑ variance; too few ↓ signal.
- **Keep** features that raise CV AUC or lower Brier.
- **Log** kept/dropped list for auditability.

## 13) Hyperparameter Tuning
- **RandomizedSearchCV**, StratifiedKFold=5, scoring=ROC-AUC.
- **Include imbalance params** (weights/`scale_pos_weight`) in search.
- **50–100 trials/model** (or smaller for demo; justify compute).
- **Fix seed & reuse folds** for apples-to-apples.

## 14) Cross-Validation Strategy
- **Stratified K=5**, shuffle, seed=42.
- **Same splitter** across models.
- **Report** mean±std ROC-AUC (variance matters with rare class).
- **Hold out** test untouched for final assessment.

## 15) Evaluation Metrics
- **Optimize ROC-AUC;** also report **PR-AUC** (imbalance sensitive).
- **Use `predict_proba`;** choose threshold on validation (F1/Youden).
- **Include Brier + calibration curve** for prob quality.
- **Show confusion** at chosen threshold (operational clarity).

## 16) Drift & Model Degradation (PSI)
- **Train vs Test PSI:** max **0.0209**; features ≥0.25 = **none**.
- **Thresholds:** <0.10 stable; 0.10–0.25 watch; >0.25 action.
- **Production plan:** periodic PSI on key features + score bands.
- **Action on drift:** retrain / resample / re-threshold if PSI rises.

## 17) Explainability
- **LR:** coefficients → odds ratios (global).
- **RF/XGB:** SHAP summary + example local explanations.
- **Sanity-check directions** (e.g., leverage ↑ → risk ↑).
- **Use explanations** to justify operating threshold to stakeholders.

---

### Appendix — PSI & Class Balance Artifacts
- **PSI table:** `reports/lab4/psi_table.csv`
- **PSI plot:** `reports/lab4/figures/psi_top.png`
- **Max PSI value:** **0.0209**
- **Top 3 PSI features:** **Operating profit per person**, **Working Capital to Total Assets**, **Cash flow rate**
- **Counts:** `# ≥0.25 = 0`, `# 0.10–0.25 = 0`, `# <0.10 = 95`
- **Class balance:** positives **220**, negatives **6599**, rate **3.23%**
- **Class plot:** `reports/lab4/figures/class_balance.png`
