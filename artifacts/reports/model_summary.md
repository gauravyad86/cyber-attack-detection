# Model Evaluation Summary

- Best supervised model: `random_forest`
- Labels: `benign, brute_force, dos, port_scan`

## Metrics

### logistic_regression
- precision_weighted: `0.9818`
- recall_weighted: `0.9780`
- f1_weighted: `0.9790`
- f1_macro: `0.9646`
- roc_auc_ovr_weighted: `0.9994`
- precision_attack_binary: `0.9363`
- recall_attack_binary: `0.9966`
- f1_attack_binary: `0.9655`

### random_forest
- precision_weighted: `1.0000`
- recall_weighted: `1.0000`
- f1_weighted: `1.0000`
- f1_macro: `1.0000`
- roc_auc_ovr_weighted: `1.0000`
- precision_attack_binary: `1.0000`
- recall_attack_binary: `1.0000`
- f1_attack_binary: `1.0000`

### isolation_forest
- precision_attack_binary: `0.6674`
- recall_attack_binary: `0.9932`
- f1_attack_binary: `0.7984`