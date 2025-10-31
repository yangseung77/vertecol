import os
import uuid
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, request, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # Headless rendering on server
import matplotlib.pyplot as plt
import gc

# ==============================================================
# Configuration
# ==============================================================
BENCHMARK_TRIALS = 50   # Reduced from 1000 for speed
BENCHMARK_TREES = 20     # Reduced from 200 for speed
CV_TREES = 100


# ==============================================================
# Flask & data paths
# ==============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

# Ensure plots dir exists
PLOTS_DIR = os.path.join(app.static_folder, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Reference data path (override with env var VERTS_PATH)
VERTS_PATH = os.getenv("VERTS_PATH", "verts.xlsx")

# Vertebrae canonical order (23)
VERTE_NAMES = [
    "C2","C3","C4","C5","C6","C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5"
]

# ==============================================================
# Load and normalize reference data (mirrors R pipeline semantics)
# ==============================================================
if not os.path.exists(VERTS_PATH):
    raise FileNotFoundError(f"Reference file not found: {VERTS_PATH}")

verts = pd.read_excel(VERTS_PATH)
verts.columns = [c.strip() for c in verts.columns]

# Guarantee Sum_Verts exists
if "Sum_Verts" not in verts.columns:
    missing = [v for v in VERTE_NAMES if v not in verts.columns]
    if missing:
        raise ValueError(f"Missing columns in {VERTS_PATH}: {missing}")
    verts["Sum_Verts"] = verts[VERTE_NAMES].sum(axis=1, skipna=True)

# Guarantee Sex exists and normalize to upper
if "Sex" not in verts.columns:
    verts["Sex"] = "UD"
verts["Sex"] = verts["Sex"].astype(str).str.strip().str.upper()

# ==============================================================
# Helpers
# ==============================================================

def sex_key_from_form(form_value: str) -> str:
    """Map UI text to R-like options: 'Pooled', 'M', 'F'."""
    s = (form_value or "").strip().lower()
    if s.startswith("pooled") or s in ("pooled", "all", "ud", "unknown"):
        return "Pooled"
    if s.startswith("m"):
        return "M"
    if s.startswith("f"):
        return "F"
    # default to pooled
    return "Pooled"


def get_sex_filtered_df(sex_key: str) -> pd.DataFrame:
    """Return a copy of verts filtered by sex; 'Pooled' returns all rows."""
    if sex_key == "Pooled":
        return verts.copy()
    if sex_key in ("M", "F"):
        df = verts[verts["Sex"] == sex_key]
        # if empty, fall back to pooled to avoid tiny-N failures
        return df.copy() if len(df) else verts.copy()
    # Unknown -> pooled
    return verts.copy()


def build_training_df(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    cols = ["Sum_Verts"] + predictors
    return df.loc[:, cols].dropna()


def safe_k(n: int, k_default: int = 5) -> int:
    if n < 4:
        return 0
    return min(k_default, max(2, min(n, n // 2)))


def cv_run(train_df: pd.DataFrame, predictors: list, method: str = "ols", k: int = 5, seed: int = 42):
    """Safe K-fold CV with R2 / RMSE, mirroring R's defaults."""
    n = len(train_df)
    k_eff = safe_k(n, k_default=k)
    if k_eff < 2:
        return {"summary": {"k": None, "R2_mean": np.nan, "R2_sd": np.nan, "RMSE_mean": np.nan, "RMSE_sd": np.nan},
                "details": []}

    kf = KFold(n_splits=k_eff, shuffle=True, random_state=seed)
    r2_list, rmse_list, details = [], [], []

    X_all = train_df[predictors]
    y_all = train_df["Sum_Verts"].values

    for i, (tr_idx, te_idx) in enumerate(kf.split(X_all), start=1):
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        if method == "ols":
            X_tr_ = sm.add_constant(X_tr, has_constant="add")
            X_te_ = sm.add_constant(X_te, has_constant="add")
            fit = sm.OLS(y_tr, X_tr_).fit()
            y_hat = fit.predict(X_te_)
            del fit #explicity delete to free memory
        else:
            rf = RandomForestRegressor(n_estimators=CV_TREES, random_state=seed, n_jobs=1, max_depth=10, min_samples_leaf=5)
            rf.fit(X_tr, y_tr)
            y_hat = rf.predict(X_te)
            del rf  # explicitly delete to free memory

        r2 = r2_score(y_te, y_hat)
        rmse = mean_squared_error(y_te, y_hat, squared=False)
        r2_list.append(r2); rmse_list.append(rmse)
        details.append({"fold": i, "R2": float(r2), "RMSE": float(rmse)})

        gc.collect()  # force garbage collection to free memory

    return {
        "summary": {
            "k": k_eff,
            "R2_mean": float(np.nanmean(r2_list)),
            "R2_sd": float(np.nanstd(r2_list, ddof=1)) if len(r2_list) > 1 else np.nan,
            "RMSE_mean": float(np.nanmean(rmse_list)),
            "RMSE_sd": float(np.nanstd(rmse_list, ddof=1)) if len(rmse_list) > 1 else np.nan
        },
        "details": details
    }


# -----------------------------
# RF quantile-style PI via tree-prediction distribution
# -----------------------------

def rf_predict_with_quantiles(rf: RandomForestRegressor, x_user: pd.DataFrame, q_low=0.025, q_high=0.975):
    """Approximate quantile intervals using distribution of per-tree predictions.
    This mirrors R's ranger(quantreg=TRUE) behavior conceptually.
    """
    # Per-tree predictions
    preds = np.array([est.predict(x_user)[0] for est in rf.estimators_], dtype=float)
    pred = float(preds.mean())
    lower = float(np.quantile(preds, q_low))
    upper = float(np.quantile(preds, q_high))
    return pred, lower, upper


# ==============================================================
# Fit + Predict (OLS with PI, RF with quantile-style PI)
# ==============================================================

def fit_and_predict(user_values: dict, sex_key: str, method: str = "ols", k: int = 5, seed: int = 42):
    """
    user_values: {"C2": 34.2, "T1": 28.4, ...}  (positives only are considered)
    """
    # keep only valid, positive predictors in canonical order
    use_vars = [v for v in VERTE_NAMES if v in user_values and pd.notnull(user_values[v]) and float(user_values[v]) > 0]
    if len(use_vars) == 0:
        raise ValueError("No valid vertebrae provided by user.")

    df_sex = get_sex_filtered_df(sex_key)
    train_df = build_training_df(df_sex, use_vars)
    if len(train_df) < 4:
        raise ValueError(f"Too few rows for training after filtering by sex={sex_key} and predictors={use_vars}")

    # CV (for UI summaries and benchmark overlays)
    cv = cv_run(train_df, use_vars, method=method, k=k, seed=seed)

    # Final fit on full training set
    X = train_df[use_vars]
    y = train_df["Sum_Verts"].values
    x_user = pd.DataFrame([{k: user_values[k] for k in use_vars}])[use_vars]

    if method == "ols":
        X_ = sm.add_constant(X, has_constant="add")
        fit = sm.OLS(y, X_).fit()
        x_user_ = sm.add_constant(x_user, has_constant="add")
        pred_frame = fit.get_prediction(x_user_).summary_frame(alpha=0.05)
        pred = float(pred_frame["mean"].iloc[0])
        pi_lower = float(pred_frame["obs_ci_lower"].iloc[0])
        pi_upper = float(pred_frame["obs_ci_upper"].iloc[0])
        model = fit
        rf_importance = None
    else:
        rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1, max_depth=15, min_samples_leaf=3)
        rf.fit(X, y)
        pred, pi_lower, pi_upper = rf_predict_with_quantiles(rf, x_user, 0.025, 0.975)
        model = rf
        # Variable importance (type-stable)
        rf_importance = (
            pd.DataFrame({"term": use_vars, "importance": rf.feature_importances_.astype(float)})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )

    return {
        "predictors": use_vars,
        "prediction": float(pred),
        "pi_lower": float(pi_lower),
        "pi_upper": float(pi_upper),
        "cv": cv,
        "model": model,
        "rf_importance": rf_importance,
        "train_n": int(len(train_df))
    }


# ==============================================================
# Random-feature benchmarking (FAST OOB for RF, CV for OLS)
# ==============================================================

def random_feature_benchmark_fixed(user_values: dict, sex_key: str, method: str = "ols",
                                   trials: int = 1000, k: int = 5, seed: int = 2025,
                                   rf_fast: bool = True, rf_trees: int = 200, rf_min_samples_leaf: int = 1):
    """
    p = number of user predictors. For each trial, sample p random features.
    - RF: if rf_fast, use OOB R² / OOB RMSE (fast path) instead of K-fold CV.
    - OLS: always K-fold CV.
    """
    p = len([v for v in VERTE_NAMES if v in user_values and float(user_values[v]) > 0])
    if p < 1:
        return {"summary": {}, "details": pd.DataFrame()}

    df_sex = get_sex_filtered_df(sex_key)
    all_vars = [v for v in VERTE_NAMES if v in df_sex.columns]

    rows = []
    rng = random.Random(seed)

    for t in range(1, trials + 1):
        feats = rng.sample(all_vars, k=p)
        train_df = build_training_df(df_sex, feats)
        if len(train_df) < 4:
            rows.append({"trial": t, "R2_mean": np.nan, "RMSE_mean": np.nan})
            continue

        if method == "rf" and rf_fast:
            X = train_df[feats].values 
            y = train_df["Sum_Verts"].values

            # OOB model
            rf = RandomForestRegressor(
                n_estimators=rf_trees,
                oob_score=True,
                bootstrap=True,
                random_state=seed + t,
                n_jobs=1,   #changed from -1 to reduce memory
                min_samples_leaf=rf_min_samples_leaf,
                max_depth=10    #added depth limit
            )
            rf.fit(X, y)
            r2_oob = float(rf.oob_score_)

            # Calculate OOB RMSE manually
            n_samples = len(y)
            oob_pred = np.zeros(n_samples)
            oob_count = np.zeros(n_samples)

            # Aggregate predictions from trees where each sample was OOB
            for estimator, samples in zip(rf.estimators_, rf.estimators_samples_):
                # Mask for out-of-bag samples (not used in this tree's bootstrap)
                mask_oob = np.ones(n_samples, dtype=bool)
                mask_oob[samples] = False

                # Predict for OOB samples
                if np.sum(mask_oob) > 0:
                    oob_pred[mask_oob] += estimator.predict(X[mask_oob])
                    oob_count[mask_oob] += 1

            # Average predictions and calculate RMSE
            mask_valid = oob_count > 0
            if np.sum(mask_valid) > 0:
                oob_pred[mask_valid] /= oob_count[mask_valid]
                rmse_oob = float(np.sqrt(np.mean((y[mask_valid] - oob_pred[mask_valid]) ** 2)))
            else:
                rmse_oob = np.nan

            rows.append({"trial": t, "R2_mean": r2_oob, "RMSE_mean": rmse_oob})

            #CRITICAL: Delete model and force garbage collection to free memory
            del rf
            if t % 10 == 0:
                gc.collect()

        else:
            cv = cv_run(train_df, feats, method=method, k=k, seed=seed + t)
            rows.append({"trial": t, "R2_mean": cv["summary"]["R2_mean"], "RMSE_mean": cv["summary"]["RMSE_mean"]})
            # Force garbage collection to free memory
            if t % 10 == 0:
                gc.collect()

    details = pd.DataFrame(rows)
    summary = {
        "p": p,
        "trials": trials,
        "valid_trials": int(np.sum(np.isfinite(details["R2_mean"]) & np.isfinite(details["RMSE_mean"]))),
        "R2_avg": float(np.nanmean(details["R2_mean"])) if len(details) else np.nan,
        "R2_sd": float(np.nanstd(details["R2_mean"], ddof=1)) if details["R2_mean"].notna().sum() > 1 else np.nan,
        "RMSE_avg": float(np.nanmean(details["RMSE_mean"])) if len(details) else np.nan,
        "RMSE_sd": float(np.nanstd(details["RMSE_mean"], ddof=1)) if details["RMSE_mean"].notna().sum() > 1 else np.nan,
    }
    return {"summary": summary, "details": details}


# ==============================================================
# Plot helpers (hist overlays with mean/user lines + labels)
# ==============================================================

def save_cv_boxplots(cv_details: list, title_prefix: str = "CV"):
    if not cv_details:
        return None
    r2_vals = [d["R2"] for d in cv_details if d["R2"] is not None]
    rmse_vals = [d["RMSE"] for d in cv_details if d["RMSE"] is not None]
    if len(r2_vals) == 0 or len(rmse_vals) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(r2_vals, vert=True)
    axes[0].set_title(f"{title_prefix} R²"); axes[0].set_ylabel("R²")

    axes[1].boxplot(rmse_vals, vert=True)
    axes[1].set_title(f"{title_prefix} RMSE"); axes[1].set_ylabel("RMSE")

    fname = f"cv_{uuid.uuid4().hex}.png"
    path = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plt.close('all')    # Extra safety
    del fig, axes   # Explicit deletion
    gc.collect()    # Force cleanup
    return url_for("static", filename=f"plots/{fname}")


def _label_positions(vals: np.ndarray, mean_val: float, user_val: float, bins: int):
    hist_counts, _ = np.histogram(vals, bins=bins)
    y_max = hist_counts.max() if len(hist_counts) else 1
    y_lab = y_max * 1.05
    xr = np.nanmax(vals) - np.nanmin(vals)
    off = max(0.02 * (xr if np.isfinite(xr) and xr > 0 else 1), 1e-6)
    mean_adj, user_adj = mean_val, user_val
    if np.isfinite(user_val) and np.isfinite(mean_val) and abs(user_val - mean_val) < 1.5 * off:
        mean_adj -= off; user_adj += off
    return mean_adj, user_adj, y_lab


def save_benchmark_histograms(bench_df: pd.DataFrame, rmse_user: float = np.nan, r2_user: float = np.nan):
    """Create benchmark histograms with legend showing benchmark mean vs user's result."""
    if bench_df is None or bench_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE Panel
    rmse_vals = bench_df["RMSE_mean"].dropna().values
    if len(rmse_vals) > 0:
        rmse_mean = float(np.nanmean(rmse_vals))
        axes[0].hist(rmse_vals, bins=30, edgecolor="black", alpha=0.7, label="Random features")
        if np.isfinite(rmse_mean):
            axes[0].axvline(rmse_mean, linestyle="--", color="gray", linewidth=2, 
                          label=f"Benchmark: {rmse_mean:.2f} mm")
        if np.isfinite(rmse_user):
            axes[0].axvline(rmse_user, linestyle="--", color="red", linewidth=2, 
                          label=f"Your model: {rmse_user:.2f} mm")
        axes[0].set_title("RMSE Distribution (Random-Feature Trials)")
        axes[0].set_xlabel("RMSE (mm)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend(loc="best", framealpha=0.9)

    # R² Panel
    r2_vals = bench_df["R2_mean"].dropna().values
    if len(r2_vals) > 0:
        r2_mean = float(np.nanmean(r2_vals))
        axes[1].hist(r2_vals, bins=30, edgecolor="black", alpha=0.7, label="Random features")
        if np.isfinite(r2_mean):
            axes[1].axvline(r2_mean, linestyle="--", color="gray", linewidth=2, 
                          label=f"Benchmark: {r2_mean:.3f}")
        if np.isfinite(r2_user):
            axes[1].axvline(r2_user, linestyle="--", color="red", linewidth=2, 
                          label=f"Your model: {r2_user:.3f}")
        axes[1].set_title("R² Distribution (Random-Feature Trials)")
        axes[1].set_xlabel("R² Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend(loc="best", framealpha=0.9)

    fname = f"bench_{uuid.uuid4().hex}.png"
    path = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plt.close('all')    # Extra safety
    del fig, axes   # Explicit deletion
    gc.collect()    # Force cleanup
    return url_for("static", filename=f"plots/{fname}")


# ==============================================================
# Routes
# ==============================================================
@app.route("/")
def predict():
    cervical = ["C2", "C3", "C4", "C5", "C6", "C7"]
    thoracic = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
    lumbar = ["L1", "L2", "L3", "L4", "L5"]
    return render_template("predict.html", cervical=cervical, thoracic=thoracic, lumbar=lumbar)

@app.route("/about")
def about():
    return render_template("about.html")

# simple in-memory store for one hop
_USER_STORE = {}

@app.route("/input", methods=["POST"])
def input():
    # Gather up to 23 vertebrae from form fields 'Vertebrae1'..'Vertebrae23'
    vals = []
    for i in range(1, 24):
        v = request.form.get(f"Vertebrae{i}", "").strip()
        vals.append(v)

    # Map to dict {name: value}
    user_values = {}
    for idx, name in enumerate(VERTE_NAMES):
        try:
            val = float(vals[idx]) if vals[idx] != "" else 0.0
        except Exception:
            val = 0.0
        if val > 0.0:  # only positive, mirrors R's positive filter
            user_values[name] = val

    sex_form = request.form.get("Sex", "Pooled")
    sex_key = sex_key_from_form(sex_form)

    reg_form = request.form.get("Reg", "Linear")
    method = "ols" if reg_form.lower().startswith("linear") else "rf"

    token = uuid.uuid4().hex
    _USER_STORE[token] = {"user_values": user_values, "sex_key": sex_key, "method": method}
    return redirect(url_for("getPrediction", token=token))

@app.route("/results", methods=["GET"])
def getPrediction():

    try:
        token = request.args.get("token")
        if not token or token not in _USER_STORE:
            return "Invalid request.", 400

        user_blob = _USER_STORE.pop(token)
        user_values = user_blob["user_values"]
        sex_key = user_blob["sex_key"]
        method = user_blob["method"]

        if len(user_values) == 0:
            return render_template("results.html")

        # Fit + Predict with ONLY user-provided features
        try:
            res = fit_and_predict(user_values, sex_key=sex_key, method=method, k=5, seed=42)
        except Exception as e:
            print(f"ERROR in fit_and_predict: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template("results.html")

        # CV plot
        cv_img = save_cv_boxplots(res["cv"]["details"], title_prefix=f"{method.upper()} ({sex_key})")
        
        # Random-feature benchmark (1000 trials; RF uses FAST OOB path)
        bench = random_feature_benchmark_fixed(
            user_values, 
            sex_key=sex_key, 
            method=method, 
            trials=BENCHMARK_TRIALS,
            k=5, 
            seed=2025,
            rf_fast=True, 
            rf_trees=BENCHMARK_TREES,
            rf_min_samples_leaf=5
        )

        # Benchmark histograms with mean (gray dashed) and user (red dashed)
        r2_user = res["cv"]["summary"]["R2_mean"]
        rmse_user = res["cv"]["summary"]["RMSE_mean"]
        bench_img = save_benchmark_histograms(bench["details"], rmse_user=rmse_user, r2_user=r2_user)

        # Benchmark summary
        bsum = bench.get("summary", {}) or {}

        # Optional: top-10 RF importance (if RF)
        rf_importance = None
        if method == "rf" and res.get("rf_importance") is not None and len(res["rf_importance"]) > 0:
            rf_importance = res["rf_importance"].head(10).copy()
        
        # Clean up model from memory before rendering
        if 'res' in locals() and 'model' in res:
            del res['model']
        gc.collect()
        
        return render_template(
            "results.html",
            # core outputs
            prediction=round(res["prediction"], 1),
            pi_lower=round(res["pi_lower"], 1),
            pi_upper=round(res["pi_upper"], 1),
            predictors=", ".join(res["predictors"]),
            sex=sex_key,
            method=method.upper(),
            train_n=res["train_n"],
            cv_k=res["cv"]["summary"]["k"],
            cv_r2_mean=None if r2_user is None else (None if np.isnan(r2_user) else round(float(r2_user), 4)),
            cv_rmse_mean=None if rmse_user is None else (None if np.isnan(rmse_user) else round(float(rmse_user), 4)),
            # images
            cv_img=cv_img,
            bench_img=bench_img,
            # benchmark summary
            bench_p=bsum.get("p"),
            bench_trials=bsum.get("trials"),
            bench_valid=bsum.get("valid_trials"),
            r2_bench_avg=bsum.get("R2_avg"),
            r2_bench_sd=bsum.get("R2_sd"),
            rmse_bench_avg=bsum.get("RMSE_avg"),
            rmse_bench_sd=bsum.get("RMSE_sd"),
            # optional extras
            rf_importance=None if rf_importance is None else rf_importance.to_dict(orient="records")
        )

    except Exception as e:
        gc.collect()
        return f"Prediction Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)