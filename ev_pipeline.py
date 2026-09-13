"""
Predicting Electric Vehicle Interest - final pipeline.

Model : LightGBM (15 leaves, lr 0.05), 5-fold CV + 4 full-data seeds.
Score : OOF AUC 0.941977

Usage
-----
    python ev_pipeline.py              # full run  (~20 min, 8 threads)
    python ev_pipeline.py --smoke      # 100k-row sanity run (~1 min)
    python ev_pipeline.py --folds 0 1  # run a subset of folds, then resume later

Fold artifacts are cached to WORK_DIR, so an interrupted run resumes where it
stopped rather than retraining from scratch.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "/mnt/user-data/uploads"
WORK_DIR = "/home/claude/work"
OUT_PATH = "/mnt/user-data/outputs/submission.csv"

TARGET   = "Will_Buy_EV"
N_FOLDS  = 5
SEED     = 42
FULL_SEEDS = [101, 202, 303, 404]   # full-data models, averaged with the CV bag

# Tuned by search over num_leaves {7,15,31,63,127,255}: performance improves
# monotonically as capacity DROPS, then flattens. 15 leaves matches 7 leaves
# within seed noise and trains ~3x faster.
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 500,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "feature_pre_filter": False,
    "verbose": -1,
    "num_threads": -1,
}
MAX_ROUNDS = 4000
EARLY_STOP = 200

CAT_COLS = ["gender", "city", "car"]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def build_features(df):
    """Raw columns -> numeric matrix.

    Deliberately minimal. A 13-feature engineered set (ratios, explicit
    env x subsidy interactions, log-income terms) scored WORSE on OOF
    (0.94145 vs 0.94174): the trees already recover those interactions,
    and the extra columns only dilute the splits.

    The two clip flags are the exception and do earn their place --
    income is floor-clipped at exactly $30,000 (9.2% of rows) and commute
    at exactly 5.0 km (21.5%). The income floor is a strong negative
    marker: 4.4% buy rate vs 18.8% elsewhere.
    """
    d = df.copy()

    d["inc_clip"] = (d["Annual_Income_USD"] == 30000).astype(np.int8)
    d["com_clip"] = (d["Daily_Commute_km"] == 5.0).astype(np.int8)

    # Ordinal where the level has a natural order, plain codes otherwise.
    d["range_anx"] = d["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(np.int8)
    d["home_chg"]  = (d["Home_Charging_Possible"] == "Yes").astype(np.int8)
    d["subsidy"]   = (d["Subsidy_Available"] == "Yes").astype(np.int8)
    d["gender"]    = d["Gender"].map({"Male": 0, "Female": 1, "Other": 2}).astype(np.int8)
    d["city"]      = d["City_Type"].map({"Urban": 0, "Suburban": 1, "Rural": 2}).astype(np.int8)
    d["car"]       = d["Current_Car_Type"].map({"Sedan": 0, "SUV": 1, "Hatchback": 2, "Truck": 3}).astype(np.int8)
    d["env"]       = d["Environmental_Concern_Level"].astype(np.int8)

    d = d.drop(columns=[
        "Range_Anxiety_Level", "Home_Charging_Possible", "Subsidy_Available",
        "Gender", "City_Type", "Current_Car_Type", "Environmental_Concern_Level",
    ])
    return d


def load_data(smoke=False):
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test  = pd.read_csv(f"{DATA_DIR}/test.csv")

    if smoke:
        train = train.sample(100_000, random_state=SEED).reset_index(drop=True)
        test  = test.head(20_000).reset_index(drop=True)

    y = (train[TARGET] == "Yes").astype(np.int8).values
    test_id = test["id"].values

    X  = build_features(train.drop(columns=[TARGET, "id"]))
    Xt = build_features(test.drop(columns=["id"]))

    assert list(X.columns) == list(Xt.columns), "train/test column mismatch"
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

    return (X.values.astype(np.float32), Xt.values.astype(np.float32),
            y, test_id, list(X.columns), cat_idx)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run_cv(X, Xt, y, cat_idx, folds_to_run, cache=True):
    """5-fold CV. Returns OOF preds and the fold-averaged test preds.

    Each fold's output is cached so a long run can be resumed.
    """
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y))

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(Xt))
    best_iters = []

    for f in folds_to_run:
        tr_idx, va_idx = folds[f]
        cache_oof = f"{WORK_DIR}/fold{f}_oof.npy"

        if cache and os.path.exists(cache_oof):
            oof[va_idx] = np.load(cache_oof)
            test_pred += np.load(f"{WORK_DIR}/fold{f}_pt.npy") / N_FOLDS
            print(f"  fold {f}  auc={roc_auc_score(y[va_idx], oof[va_idx]):.6f}  (cached)")
            continue

        t0 = time.time()
        model = lgb.train(
            PARAMS,
            lgb.Dataset(X[tr_idx], y[tr_idx], categorical_feature=cat_idx),
            MAX_ROUNDS,
            valid_sets=[lgb.Dataset(X[va_idx], y[va_idx], categorical_feature=cat_idx)],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        best = model.best_iteration
        best_iters.append(best)

        p_va = model.predict(X[va_idx], num_iteration=best)
        p_te = model.predict(Xt, num_iteration=best)
        oof[va_idx] = p_va
        test_pred += p_te / N_FOLDS

        if cache:
            np.save(cache_oof, p_va)
            np.save(f"{WORK_DIR}/fold{f}_pt.npy", p_te)
            np.save(f"{WORK_DIR}/fold{f}_idx.npy", va_idx)

        print(f"  fold {f}  auc={roc_auc_score(y[va_idx], p_va):.6f}  "
              f"iters={best}  {time.time() - t0:.0f}s", flush=True)

    return oof, test_pred, best_iters, folds


def run_full_seeds(X, Xt, y, cat_idx, n_rounds, seeds=FULL_SEEDS, cache=True):
    """Train on 100% of the data at several seeds and average.

    Seed bagging is the one change that reliably helped: on a held-out fold,
    a single seed scored ~0.94079 and a 3-seed bag reached 0.940934.
    No early stopping is possible here (no validation set), so n_rounds is
    scaled up from the CV best_iteration to account for the larger training set.
    """
    preds = []
    for s in seeds:
        path = f"{WORK_DIR}/full_s{s}.npy"
        if cache and os.path.exists(path):
            preds.append(np.load(path))
            print(f"  seed {s}  (cached)")
            continue

        t0 = time.time()
        p = {**PARAMS, "seed": s, "bagging_seed": s + 1, "feature_fraction_seed": s + 2}
        model = lgb.train(p, lgb.Dataset(X, y, categorical_feature=cat_idx), n_rounds)
        pred = model.predict(Xt)
        preds.append(pred)
        if cache:
            np.save(path, pred)
        print(f"  seed {s}  rounds={n_rounds}  {time.time() - t0:.0f}s", flush=True)

    return np.mean(preds, axis=0)


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------
def write_submission(test_id, pred, path=OUT_PATH, validate_against_sample=True):
    sub = pd.DataFrame({"id": test_id, TARGET: np.clip(pred, 1e-6, 1 - 1e-6)})

    assert sub[TARGET].notna().all() and np.isfinite(sub[TARGET]).all(), "non-finite predictions"
    assert sub[TARGET].between(0, 1).all(), "predictions outside [0, 1]"
    assert sub["id"].is_unique, "duplicate ids"

    if validate_against_sample:
        sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
        assert len(sub) == len(sample), f"row count {len(sub)} != {len(sample)}"
        assert (sub["id"].values == sample["id"].values).all(), "id order differs from sample"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"\nwrote {path}  rows={len(sub)}  mean={sub[TARGET].mean():.5f}")
    return sub


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast run on a 100k subsample")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds to run")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cache = not args.no_cache and not args.smoke
    folds_to_run = args.folds if args.folds is not None else list(range(N_FOLDS))

    print("Loading data...")
    X, Xt, y, test_id, cols, cat_idx = load_data(smoke=args.smoke)
    print(f"  train={X.shape}  test={Xt.shape}  positive rate={y.mean():.5f}")
    print(f"  features: {cols}\n")

    print(f"Cross-validation ({len(folds_to_run)} folds)...")
    oof, pt_cv, best_iters, folds = run_cv(X, Xt, y, cat_idx, folds_to_run, cache=cache)

    ran_all = sorted(folds_to_run) == list(range(N_FOLDS))
    if not ran_all:
        print("\nPartial run - rerun without --folds to finish and write a submission.")
        return

    print(f"\nOOF AUC = {roc_auc_score(y, oof):.6f}")

    # Scale rounds up for the full-data fit: CV models see (N_FOLDS-1)/N_FOLDS
    # of the data, so the same learning rate needs proportionally more rounds.
    mean_iter = int(np.mean(best_iters)) if best_iters else 1300
    n_rounds = int(mean_iter * N_FOLDS / (N_FOLDS - 1))

    print(f"\nFull-data models ({len(FULL_SEEDS)} seeds, {n_rounds} rounds)...")
    pt_full = run_full_seeds(X, Xt, y, cat_idx, n_rounds, cache=cache)

    # Average the CV bag with the full-data bag. The two agree very closely
    # (rank corr 0.99943), so this reduces variance rather than shifting the
    # ranking. Note: a spline-logistic GAM was also tried as a blend partner
    # and did NOT help -- optimal weight came out at 1.00 on LightGBM alone.
    final = 0.5 * pt_cv + 0.5 * pt_full

    out = OUT_PATH.replace('.csv', '_smoke.csv') if args.smoke else OUT_PATH
    write_submission(test_id, final, path=out, validate_against_sample=not args.smoke)


if __name__ == "__main__":
    main()
