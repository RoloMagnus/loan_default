"""Experiment 4: Optuna tuning of LightGBM (5-fold CV accuracy)."""
import sys, os, time, warnings, json
import numpy as np, pandas as pd, optuna, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build

DATA = sys.argv[1]; N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def cv(params, rounds=4000, es=150, return_oof=False):
    oof = np.zeros(len(X)); iters = []
    for a, b in folds:
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), rounds, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); iters.append(m.best_iteration)
    acc = ((oof > 0.5) == y).mean()
    return (acc, oof, iters) if return_oof else acc


def objective(t):
    p = dict(objective="binary", verbose=-1, n_jobs=6, seed=42, bagging_freq=1,
             learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
             num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
             max_depth=t.suggest_int("max_depth", 4, 12),
             min_child_samples=t.suggest_int("min_child_samples", 10, 200, log=True),
             feature_fraction=t.suggest_float("feature_fraction", 0.4, 1.0),
             bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 1.0),
             lambda_l1=t.suggest_float("lambda_l1", 1e-3, 10, log=True),
             lambda_l2=t.suggest_float("lambda_l2", 1e-3, 10, log=True),
             min_gain_to_split=t.suggest_float("min_gain_to_split", 0, 1),
             cat_smooth=t.suggest_float("cat_smooth", 1, 50, log=True))
    acc = cv(p)
    print(f"trial {t.number}: acc={acc:.5f} {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in t.params.items()})}", flush=True)
    return acc


t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
study.enqueue_trial(dict(learning_rate=0.03, num_leaves=63, max_depth=10, min_child_samples=40, feature_fraction=0.8,
                         bagging_fraction=0.8, lambda_l1=0.001, lambda_l2=1.0, min_gain_to_split=0.0, cat_smooth=10))
study.optimize(objective, n_trials=N_TRIALS)
print("BEST", study.best_value, json.dumps(study.best_params), f"{time.time()-t0:.0f}s", flush=True)
best = dict(objective="binary", verbose=-1, n_jobs=6, seed=42, bagging_freq=1, **study.best_params)
acc, oof, iters = cv(best, return_oof=True)
print("refit CV acc", acc, "iters", iters)
json.dump({"params": study.best_params, "iters": iters, "cv_acc": acc}, open(os.path.join(HERE, "lgb_best.json"), "w"), indent=1)
np.save(os.path.join(HERE, "oof_lgb_tuned.npy"), oof)
