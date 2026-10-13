from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, X, y):
    prediction = model.predict(X)
    return {"roc_auc": float(roc_auc_score(y, model.predict_proba(X)[:, 1])),
            "accuracy": float(accuracy_score(y, prediction)),
            "f1": float(f1_score(y, prediction, zero_division=0))}

def check_thresholds(metrics, thresholds):
    failures = [f"{key}: {metrics[key]:.4f} < {minimum}" for key, minimum in thresholds.items()
                if metrics[key] < minimum]
    if failures:
        raise ValueError("Performance gate failed: " + "; ".join(failures))
