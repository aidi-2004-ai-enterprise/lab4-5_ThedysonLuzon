# Lab 5 — Report (Company Bankruptcy)

## EDA (Why)
- Minority class confirmed; heavy-tailed ratios present.
- Missingness low; median imputation chosen.
- High correlations observed; prune via |ρ|>0.90.
- Figures saved under figures/ (class balance, missingness, heatmap).

## Preprocessing & Imbalance (Why)
- Stratified split, seed fixed for reproducibility.
- Median imputation; LR scaled; trees unscaled.
- Imbalance handled: class_weight (LR/RF), scale_pos_weight (XGB).
- PSI computed train vs test to validate split stability.

## Feature Selection (Why)
- Correlation pruning to reduce redundancy and stabilize LR.
- No PCA for trees; minimal transforms keep interpretability.
- Kept features logged to selected_features.txt.
- Consistent prep across models for fair comparison.

## Hyperparameter Tuning (Why)
- RandomizedSearchCV with StratifiedKFold=3 (fast demo).
- Tiny, sensible param spaces to finish quickly.
- AUC scoring reflects ranking performance under imbalance.
- Seeded CV for apples-to-apples comparison.

## Training (Why)
- Three models trained: LR (benchmark), RF, XGB.
- Thresholds chosen on train probabilities to maximize F1 (demo speed).
- Artifacts saved in artifacts/ (*.joblib, *_params.json).
- Reproducible seeds set globally.

## Evaluation & Comparison (Why)
- Report ROC-AUC, PR-AUC, Brier, F1@threshold on train/test.
- Overlay ROC + calibration curves to check generalization & prob quality.
- Compare train vs test gaps for over/underfitting signs.
- model_comparison.csv provides a summary table.

## SHAP (Why)
- SHAP computed for best model: XGBoost (small sample for speed).
- Global summary + bar plots saved (figures/).
- Use to align top drivers with business risk.
- Local explanations optional if needed.

## PSI (Why)
- PSI per feature saved (psi_table.csv) + top plot (psi_top.png).
- Thresholds: <0.1 stable; 0.1–0.25 watch; >0.25 action.
- Highlights drift candidates for monitoring.
- Supports deployment readiness discussion.

## Challenges & Reflections (Why)
- Imbalance vs precision–recall trade-offs.
- Calibration sensitivity for LR; trees more robust.
- Compute kept low via small CV & grids; SHAP sampled.
- Next: fuller tuning & monitoring in production.

## Metric Comparison Table

| model              | set   |   roc_auc |   pr_auc |   brier |   f1_at_thresh |   threshold |
|:-------------------|:------|----------:|---------:|--------:|---------------:|------------:|
| LogisticRegression | test  |    0.9194 |   0.3613 |  0.0952 |         0.4082 |      0.9174 |
| LogisticRegression | train |    0.9548 |   0.444  |  0.0918 |         0.5112 |      0.9174 |
| RandomForest       | test  |    0.9561 |   0.5398 |  0.0226 |         0.4706 |      0.5741 |
| RandomForest       | train |    0.9998 |   0.9942 |  0.0079 |         0.9805 |      0.5741 |
| XGBoost            | test  |    0.9501 |   0.5236 |  0.0243 |         0.087  |      0.9889 |
| XGBoost            | train |    1      |   1      |  0.0002 |         1      |      0.9889 |

## Figures
- cal_LogisticRegression.png
- cal_RandomForest.png
- cal_XGBoost.png
- class_balance.png
- corr_heatmap_subset.png
- example_feature_hist.png
- missingness.png
- psi_top.png
- roc_LogisticRegression.png
- roc_RandomForest.png
- roc_XGBoost.png
- shap_bar_XGBoost.png
- shap_summary_XGBoost.png
