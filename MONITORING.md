# Drift monitoring analysis

The verified stable scenario reported no drifting features (0 of 10; share 0.00).
The verified drifted scenario reported `monthly_charge`, `support_calls`, and
`contract` (3 of 10; share 0.30). The configurable alert threshold is 0.25,
so the stable scenario returns 0 and the shifted scenario returns 1.

## Why these features drifted
The simulation adds 60 to monthly charges, adds eight support calls, and changes
all production contracts to monthly. This represents pricing changes, a service
incident, and a changed acquisition mix. The other seven features are unchanged.
Reference and production rows come from disjoint, fixed splits of the same dataset.
Evidently uses its default feature-type/sample-size-specific drift tests; the
HTML report records the individual tests, scores, and distributions. Missing values
are retained so the monitor sees the input distribution before model imputation.

## Likely performance impact
All three shifted features help determine the synthetic target, so these changes
could affect predicted churn and calibration. Some shifted numeric values fall
outside the training range, where a random forest cannot extrapolate reliably.
Input drift alone does not measure performance or establish concept drift.
The simulation deliberately leaves labels unchanged; evaluating those labels after
changing their inputs would not be a valid estimate of real production accuracy.

## Recommended action
Investigate first: confirm prices and support counts are measured consistently,
check pipeline/schema changes, and verify whether contract mix changed for a valid
business reason. Compare missingness and feature ranges as separate quality checks.
Acquire representative new labels, evaluate ROC AUC, F1, accuracy and calibration,
then retrain and validate if degradation is confirmed. Gate promotion against fresh
holdout data and retain the previous model for rollback. Continue monitoring the
stable scenario; an alert does not by itself authorize automatic retraining.

Run `python -m src.monitor_drift --scenario stable` or `--scenario drifted`.
Reports are saved under `reports/drift_<scenario>.html` with JSON summaries.
