"""
Experiments behind the final model - including the ones that failed.

Kept so the design choices in ev_pipeline.py are reproducible rather than
asserted. Each block prints the number that drove the decision.

    python experiments.py eda        # distributions + univariate target rates
    python experiments.py capacity   # num_leaves sweep (fold 0)
    python experiments.py features   # engineered features vs raw
    python experiments.py gam        # spline-logistic GAM + blend sweep
    python experiments.py seedbag    # does seed averaging help?

Summary of results:
    capacity   less capacity is better, monotonically, then flat
    features   13 engineered features scored WORSE (.94145 vs .94174)
    gam        GAM .9395, dominated - optimal blend weight is 1.00 on LGB
    seedbag    +0.00015 from 3 seeds; the one change that helped
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from scipy.stats import rankdata

from ev_pipeline import (DATA_DIR, TARGET, PARAMS, SEED, N_FOLDS,
                         build_features, load_data)

BASE = {k: v for k, v in PARAMS.items()}


def _fold0(X, y):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[0]


# --------------------------------------------------------------------------
def eda():
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")
    tr["y"] = (tr[TARGET] == "Yes").astype(int)

    print(f"train={tr.shape}  test={te.shape}")
    print(f"nulls: train={tr.isna().sum().sum()}  test={te.isna().sum().sum()}")
    print(f"positive rate = {tr['y'].mean():.5f}\n")

    for c in ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
              "Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level"]:
        g = tr.groupby(c)["y"].agg(["count", "mean"])
        g["train%"] = (g["count"] / len(tr) * 100).round(2)
        g["test%"] = (te[c].value_counts(normalize=True) * 100).round(2)
        print(f"--- {c} ---\n{g}\n")

    # The two floor-clipped columns.
    tr["inc_clip"] = (tr["Annual_Income_USD"] == 30000).astype(int)
    tr["com_clip"] = (tr["Daily_Commute_km"] == 5.0).astype(int)
    print("--- clip flags ---")
    print(tr.groupby("inc_clip")["y"].agg(["count", "mean"]))
    print(tr.groupby("com_clip")["y"].agg(["count", "mean"]))

    # Near-constant log-odds gap across Env levels => close to additive.
    print("\n--- Subsidy x Env (log-odds gap) ---")
    pv = tr.pivot_table(index="Environmental_Concern_Level",
                        columns="Subsidy_Available", values="y", aggfunc="mean")
    lo = lambda p: np.log(p / (1 - p))
    pv["logodds_gap"] = (lo(pv["Yes"]) - lo(pv["No"])).round(2)
    print(pv)


# --------------------------------------------------------------------------
def capacity():
    """Sweep model capacity on fold 0. Smaller wins, monotonically."""
    X, Xt, y, _, _, cat_idx = load_data()
    tr_idx, va_idx = _fold0(X, y)

    grid = [
        {"num_leaves": 255, "min_data_in_leaf": 500, "feature_fraction": 0.7, "lambda_l2": 10},
        {"num_leaves": 127, "min_data_in_leaf": 300, "feature_fraction": 0.8, "lambda_l2": 5},
        {"num_leaves": 63,  "min_data_in_leaf": 100, "feature_fraction": 0.8, "lambda_l2": 1},
        {"num_leaves": 31,  "min_data_in_leaf": 100, "feature_fraction": 0.9, "lambda_l2": 1},
        {"num_leaves": 15,  "min_data_in_leaf": 500, "feature_fraction": 1.0, "lambda_l2": 5},
        {"num_leaves": 7,   "min_data_in_leaf": 500, "feature_fraction": 1.0, "lambda_l2": 5},
    ]
    for g in grid:
        m = lgb.train({**BASE, **g, "seed": SEED},
                      lgb.Dataset(X[tr_idx], y[tr_idx], categorical_feature=cat_idx), 8000,
                      valid_sets=[lgb.Dataset(X[va_idx], y[va_idx], categorical_feature=cat_idx)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        auc = roc_auc_score(y[va_idx], m.predict(X[va_idx], num_iteration=m.best_iteration))
        print(f"  leaves={g['num_leaves']:>3}  auc={auc:.6f}  iters={m.best_iteration}", flush=True)


# --------------------------------------------------------------------------
def _fe_heavy(d):
    """The engineered set that did not work."""
    d = build_features(d)
    inc, com = d["Annual_Income_USD"], d["Daily_Commute_km"]
    d["log_inc"] = np.log(inc)
    d["inc_per_car"] = inc / d["Number_of_Cars_Owned"]
    d["tot_chg"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["chg_diff"] = d["Charging_Stations_Near_Home"] - d["Charging_Stations_Near_Work"]
    d["chg_access"] = d["tot_chg"] * (1 + d["home_chg"])
    d["env_subsidy"] = d["env"] * 2 + d["subsidy"]
    d["env_anx"] = d["env"] * 3 + d["range_anx"]
    d["env_x_sub"] = d["env"] * d["subsidy"]
    d["readiness"] = d["subsidy"] + d["home_chg"] + (d["range_anx"] == 0).astype(int)
    d["env_income"] = d["env"] * d["log_inc"]
    d["commute_per_chg"] = com / (1 + d["tot_chg"])
    d["inc_x_env_sub"] = d["log_inc"] * d["env"] * (1 + d["subsidy"])
    d["age_inc"] = d["Age"] * d["log_inc"]
    return d


def features():
    """Engineered features vs raw, full 5-fold. Raw wins."""
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    y = (tr[TARGET] == "Yes").astype(np.int8).values
    raw = tr.drop(columns=[TARGET, "id"])

    for name, frame in [("raw (15 feats)", build_features(raw)),
                        ("engineered (28 feats)", _fe_heavy(raw))]:
        X = frame.values.astype(np.float32)
        cat_idx = [frame.columns.get_loc(c) for c in ["gender", "city", "car"]]
        oof = np.zeros(len(y))
        for tr_i, va_i in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y):
            m = lgb.train({**BASE, "seed": SEED},
                          lgb.Dataset(X[tr_i], y[tr_i], categorical_feature=cat_idx), 4000,
                          valid_sets=[lgb.Dataset(X[va_i], y[va_i], categorical_feature=cat_idx)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[va_i] = m.predict(X[va_i], num_iteration=m.best_iteration)
        print(f"  {name:<24} OOF = {roc_auc_score(y, oof):.6f}", flush=True)


# --------------------------------------------------------------------------
def gam():
    """Spline-logistic GAM, then sweep the blend weight against LightGBM.

    The GAM reaching .9395 shows the process is close to logistic-additive.
    But it is dominated, not complementary: optimal blend weight is 1.00.
    """
    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    y = (tr[TARGET] == "Yes").astype(int).values
    d = tr.drop(columns=[TARGET, "id"])

    num = ["Age", "Annual_Income_USD", "Daily_Commute_km",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    cat = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
           "Subsidy_Available", "Range_Anxiety_Level",
           "Environmental_Concern_Level", "Number_of_Cars_Owned"]

    A = StandardScaler().fit_transform(
        SplineTransformer(n_knots=12, degree=3, include_bias=False).fit_transform(d[num]))
    D = pd.get_dummies(d[cat].astype(str), drop_first=True).values.astype(float)
    clip = np.column_stack([(d["Annual_Income_USD"] == 30000).astype(float),
                            (d["Daily_Commute_km"] == 5.0).astype(float)])
    Xg = np.hstack([A, D, clip]).astype(np.float32)

    oof_gam = np.zeros(len(y))
    for tr_i, va_i in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xg, y):
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xg[tr_i], y[tr_i])
        oof_gam[va_i] = lr.predict_proba(Xg[va_i])[:, 1]
    print(f"  GAM OOF = {roc_auc_score(y, oof_gam):.6f}")

    # Requires ev_pipeline.py to have been run (fold caches present).
    try:
        oof_lgb = np.load("/home/claude/work/oof_lgb.npy")
    except FileNotFoundError:
        print("  run ev_pipeline.py first to produce oof_lgb.npy")
        return

    a_lgb = roc_auc_score(y, oof_lgb)
    print(f"  LGB OOF = {a_lgb:.6f}")
    print(f"  rank corr = {np.corrcoef(rankdata(oof_lgb), rankdata(oof_gam))[0, 1]:.5f}\n")

    ro, rg = rankdata(oof_lgb) / len(y), rankdata(oof_gam) / len(y)
    for w in np.arange(0.5, 1.01, 0.05):
        print(f"  w_lgb={w:.2f}  blend OOF = {roc_auc_score(y, w * ro + (1 - w) * rg):.6f}")


# --------------------------------------------------------------------------
def seedbag():
    """Does averaging seeds help? Yes - the only change that did."""
    X, Xt, y, _, _, cat_idx = load_data()
    tr_idx, va_idx = _fold0(X, y)

    preds = []
    for s in [11, 22, 33]:
        m = lgb.train({**BASE, "seed": s, "bagging_seed": s + 1, "feature_fraction_seed": s + 2},
                      lgb.Dataset(X[tr_idx], y[tr_idx], categorical_feature=cat_idx), 4000,
                      valid_sets=[lgb.Dataset(X[va_idx], y[va_idx], categorical_feature=cat_idx)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(X[va_idx], num_iteration=m.best_iteration))
        print(f"  seed {s}  auc={roc_auc_score(y[va_idx], preds[-1]):.6f}", flush=True)
        if len(preds) > 1:
            bag = np.mean([rankdata(p) for p in preds], axis=0)
            print(f"     bag of {len(preds)} -> {roc_auc_score(y[va_idx], bag):.6f}", flush=True)


if __name__ == "__main__":
    fns = {"eda": eda, "capacity": capacity, "features": features,
           "gam": gam, "seedbag": seedbag}
    if len(sys.argv) < 2 or sys.argv[1] not in fns:
        print(__doc__)
        sys.exit(1)
    fns[sys.argv[1]]()
