"""Experiment 19: Optuna tuning of CatBoost (5-fold CV accuracy, resumable study)."""
import sys, os, time, json, warnings
import numpy as np, pandas as pd, optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from features import build, CAT_COLS

DATA = sys.argv[1]; N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
tr = pd.read_csv(f"{DATA}/Train_set.csv"); te = pd.read_csv(f"{DATA}/Test_set.csv")
X, y, Xte = build(tr, te)
for c in CAT_COLS: X[c] = X[c].astype(str)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
pools = [(Pool(X.iloc[a], y[a], cat_features=CAT_COLS), Pool(X.iloc[b], y[b], cat_features=CAT_COLS), b) for a, b in folds]


def cv(params, return_oof=False):
    oof = np.zeros(len(X)); iters = []
    for ptr, pva, b in pools:
        m = CatBoostClassifier(iterations=3000, eval_metric="Logloss", od_type="Iter", od_wait=100,
                               random_seed=42, verbose=False, thread_count=12, **params)
        m.fit(ptr, eval_set=pva, use_best_model=True)
        oof[b] = m.predict_proba(pva)[:, 1]; iters.append(m.get_best_iteration())
    acc = ((oof > 0.5) == y).mean()
    return (acc, oof, iters) if return_oof else acc


def objective(t):
    p = dict(learning_rate=t.suggest_float("learning_rate", 0.03, 0.15, log=True),
             depth=t.suggest_int("depth", 5, 9),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 20, log=True),
             random_strength=t.suggest_float("random_strength", 0.1, 5, log=True),
             bagging_temperature=t.suggest_float("bagging_temperature", 0, 2),
             border_count=t.suggest_categorical("border_count", [128, 254]),
             min_data_in_leaf=t.suggest_int("min_data_in_leaf", 1, 100, log=True))
    acc = cv(p)
    print(f"trial {t.number}: acc={acc:.5f} {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in t.params.items()})}", flush=True)
    return acc


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=3),
                            study_name="cat_acc", storage=f"sqlite:///{os.path.join(HERE, 'optuna_cat.db')}", load_if_exists=True)
if len(study.trials) == 0:
    study.enqueue_trial(dict(learning_rate=0.08, depth=7, l2_leaf_reg=3, random_strength=1, bagging_temperature=1, border_count=254, min_data_in_leaf=1))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS)
print("BEST", study.best_value, json.dumps(study.best_params), f"{time.time()-t0:.0f}s", flush=True)
acc, oof, iters = cv(study.best_params, return_oof=True)
json.dump({"params": study.best_params, "iters": iters, "cv_acc": acc}, open(os.path.join(HERE, "cat_best.json"), "w"), indent=1)
np.save(os.path.join(HERE, "oof_cat_tuned.npy"), oof)
print("refit acc", acc, "iters", iters)
