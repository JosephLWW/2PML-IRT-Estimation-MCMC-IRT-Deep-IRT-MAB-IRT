#!/usr/bin/env python3
"""Main script to run IRT (2PLM) with EAP via MCMC across all datasets (Table 2 and Table 3),
and to generate final CSVs with fit metrics."""


import os
import sys
import warnings
import time
import multiprocessing
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error

# Insert src directory into sys.path to import DataLoader
is_windows = os.name == "nt"
project_root_env = os.getenv("SIM_ROOT")
if project_root_env:
    PROJECT_ROOT = Path(project_root_env).expanduser()
elif is_windows:
    PROJECT_ROOT = Path.cwd()
else:
    PROJECT_ROOT = Path.home() / "Research_Project"

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import DataLoader


# ───────────────────────────────────────────────────────────
# 1. MCMC configuration
# ───────────────────────────────────────────────────────────
MCMC_CFG = dict(
    tune=1000,
    draws=2000,
    chains=4,
    cores=min(32, multiprocessing.cpu_count()),
    target_accept=0.95,
    init="adapt_diag",
    return_inferencedata=True,
    max_treedepth=12,
    progressbar=False,
)


# ───────────────────────────────────────────────────────────
# 2. Joint IRT model (2PLM)
# ───────────────────────────────────────────────────────────
def build_joint_irt_model(dgp):
    """
    Build a joint 2PL model that works whether dgp.u[k] has shape (N_k, total_items) or is already trimmed to (N_k, len(item_ids[k])).
    """
    J = dgp.total_items
    with pm.Model() as model:
        # Global item priors
        log_a = pm.Normal("log_a", 0.0, 1.0, shape=J)
        a = pm.Deterministic("a", pm.math.exp(log_a))
        b = pm.Normal("b", 1.0, np.sqrt(0.4), shape=J)

        # For each test k, define theta_k and the likelihood
        for k in range(dgp.K):
            ids_k = dgp.item_ids[k]
            u_k = dgp.u[k]
            N_k = u_k.shape[0]

            # Definir theta para ese test
            theta_k = pm.Normal(f"theta_{k}", 0.0, 1.0, shape=N_k)

            # Build logits using only the items in this test
            logits = 1.7 * a[ids_k] * (theta_k[:, None] - b[ids_k])

            # If u_k is not trimmed, extract the columns
            if u_k.shape[1] != len(ids_k):
                u_obs = u_k[:, ids_k]
            else:
                u_obs = u_k

            pm.Bernoulli(f"obs_{k}", p=pm.math.sigmoid(logits), observed=u_obs)

    return model


def fit_dataset_joint(dgp, seed=42, verbose=True):
    """
    Fit all tests jointly for the dgp object and return a list of dicts with 'theta', 'a', 'b', and 'status' for each test k.
    """
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Silenciar la salida de PyMC para que no llene la consola
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model = build_joint_irt_model(dgp)
            with model:
                idata = pm.sample(**MCMC_CFG, random_seed=seed)

    secs = time.time() - t0

    # Compute the max R̂ across variables 'a', 'b', and each 'theta_k'
    rhat_vals = []
    rhat_vals.append(az.rhat(idata, var_names=["a"]).to_array().values.max())
    rhat_vals.append(az.rhat(idata, var_names=["b"]).to_array().values.max())
    for k in range(dgp.K):
        rhat_theta_k = az.rhat(idata, var_names=[f"theta_{k}"]).to_array().values.max()
        rhat_vals.append(rhat_theta_k)
    rhat_max = float(np.nanmax(rhat_vals))
    status = "SUCCESS" if rhat_max <= 1.1 else ("WARNING" if rhat_max <= 1.2 else "POOR")

    # Extract point estimates (posterior mean) for a and b
    a_hat = idata.posterior["a"].mean(("chain", "draw")).values
    b_hat = idata.posterior["b"].mean(("chain", "draw")).values

    # Build fit list: for each test, extract theta_k and fill global vectors
    fits = []
    for k in range(dgp.K):
        theta_k = idata.posterior[f"theta_{k}"].mean(("chain", "draw")).values
        ids_k = dgp.item_ids[k]

        # Fill global vectors of length total_items with nan outside ids_k
        a_full = np.full(dgp.total_items, np.nan)
        a_full[ids_k] = a_hat[ids_k]
        b_full = np.full(dgp.total_items, np.nan)
        b_full[ids_k] = b_hat[ids_k]

        if verbose:
            print(
                f"   [joint] Test {k+1}/{dgp.K} — θ μ={theta_k.mean():+.2f} σ={theta_k.std():.2f}"
            )

        fits.append({"theta": theta_k, "a": a_full, "b": b_full, "status": status})

    print(f"   → Joint fit completed in {secs:.1f}s; R̂_max={rhat_max:.3f}")
    return fits


# ───────────────────────────────────────────────────────────
# 3. Per‑test IRT model (2PLM)
# ───────────────────────────────────────────────────────────
def build_irt_model(N, J, u):
    """
    Build a 2PLM for a single test:
      theta ~ N(0,1), log_a ~ N(0,1) -> a = exp(log_a),
      b ~ N(1, 0.4). Bernoulli likelihood with D=1.7.
    """
    with pm.Model() as m:
        theta = pm.Normal("theta", 0.0, 1.0, shape=N)
        log_a = pm.Normal("log_a", 0.0, 1.0, shape=J)
        a = pm.Deterministic("a", pm.math.exp(log_a))
        b = pm.Normal("b", 1.0, np.sqrt(0.4), shape=J)
        logits = 1.7 * a[None, :] * (theta[:, None] - b[None, :])
        p = pm.math.sigmoid(logits)
        pm.Bernoulli("obs", p=p, observed=u)
    return m


def fit_one_test(dgp, k, seed_base=42, verbose=True):
    """
    Fit a single test k from dgp:
    - Trim u_k and item_ids[k] to valid columns.
    - Return theta_hat, a_sub, b_sub and their global indices.
    """
    ids_global = np.array(dgp.item_ids[k], dtype=int)
    u_k_full = dgp.u[k]

    # Determine whether u_k_full includes only this test's columns or is wider
    if u_k_full.shape[1] == len(ids_global):
        u_k = u_k_full.copy()
        ids = ids_global.copy()
    else:
        # Filter only items whose global index < number of columns
        valid_mask = ids_global < u_k_full.shape[1]
        ids = ids_global[valid_mask]
        u_k = u_k_full[:, ids]

    if u_k.shape[1] < 2:
        return None  # do not calibrate if fewer than 2 items

    N, J = u_k.shape
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            m = build_irt_model(N, J, u_k)
            with m:
                idata = pm.sample(**MCMC_CFG, random_seed=seed_base + k)
    secs = time.time() - t0

    theta_hat = idata.posterior["theta"].mean(("chain", "draw")).values
    a_sub = idata.posterior["a"].mean(("chain", "draw")).values
    b_sub = idata.posterior["b"].mean(("chain", "draw")).values

    rhat = az.rhat(idata, var_names=["theta", "a", "b"]).to_array().values.max()
    status = "SUCCESS" if rhat <= 1.1 else ("WARNING" if rhat <= 1.2 else "POOR")

    if verbose:
        print(
            f"         [per‑test] Test {k+1}/{dgp.K}…  "
            f"θ μ={theta_hat.mean():+.2f} σ={theta_hat.std():.2f}  "
            f"a μ={np.nanmean(a_sub):.2f} σ={np.nanstd(a_sub):.2f}  "
            f"b μ={np.nanmean(b_sub):.2f} σ={np.nanstd(b_sub):.2f}  "
            f"r̂={rhat:.3f}  {secs:.1f}s"
        )

    return {
        "theta": theta_hat,
        "ids": ids,        # índices globales de los calibrated items
        "a_sub": a_sub,    # discriminations only for those items
        "b_sub": b_sub,    # difficulties only for those items
        "status": status
    }


# ───────────────────────────────────────────────────────────
# 4. Per‑test linking
# ───────────────────────────────────────────────────────────
def any_common(dgp):
    """
    Return True if at least one pair of adjacent tests share items.
    """
    for i in range(dgp.K - 1):
        if np.intersect1d(dgp.item_ids[i], dgp.item_ids[i + 1]).size:
            return True
    return False


def link_mean(fits, dgp):
    """
    Mean/mean linking using only sub‑vectors of a_sub and b_sub.
    Requires at least 3 common items; adjusts θ and b_sub (+C).
    """
    if not fits or fits[0] is None or fits[0]["status"] == "POOR":
        return fits

    ref = fits[0]
    ref_ids = ref["ids"]
    ref_b = ref["b_sub"]
    linked = [ref]

    for k in range(1, dgp.K):
        cur = fits[k]
        if cur is None or cur["status"] == "POOR":
            linked.append(cur)
            continue

        cur_ids = cur["ids"]
        cur_b = cur["b_sub"]

        common = np.intersect1d(ref_ids, cur_ids)
        if common.size < 3:
            linked.append(cur)
            continue

        b_ref_vals = []
        b_cur_vals = []
        for g in common:
            pos_ref = np.where(ref_ids == g)[0][0]
            pos_cur = np.where(cur_ids == g)[0][0]
            b_r = ref_b[pos_ref]
            b_c = cur_b[pos_cur]
            if np.isfinite(b_r) and np.isfinite(b_c):
                b_ref_vals.append(b_r)
                b_cur_vals.append(b_c)

        b_ref_arr = np.array(b_ref_vals)
        b_cur_arr = np.array(b_cur_vals)
        if b_ref_arr.size < 3:
            linked.append(cur)
            continue

        C = b_ref_arr.mean() - b_cur_arr.mean()

        linked.append({
            "theta": cur["theta"] + C,
            "ids": cur_ids,
            "a_sub": cur["a_sub"],        # no reescala a en mean/mean
            "b_sub": cur["b_sub"] + C,
            "status": cur["status"]
        })

    return linked


def link_mean_sd(fits, dgp):
    """
    Mean/SD linking (Haebara‑like) using only sub‑vectors:
    compute A = sd(b_ref)/sd(b_cur), C = mean(b_ref) − A*mean(b_cur),
    then θ_linked = A*θ_cur + C, b_sub_linked = A*b_sub + C, a_sub_linked = a_sub/A.
    """
    if not fits or fits[0] is None or fits[0]["status"] == "POOR":
        return fits

    ref = fits[0]
    ref_ids = ref["ids"]
    ref_b = ref["b_sub"]
    linked = [ref]

    for k in range(1, dgp.K):
        cur = fits[k]
        if cur is None or cur["status"] == "POOR":
            linked.append(cur)
            continue

        cur_ids = cur["ids"]
        cur_b = cur["b_sub"]

        common = np.intersect1d(ref_ids, cur_ids)
        if common.size < 3:
            linked.append(cur)
            continue

        b_ref_vals = []
        b_cur_vals = []
        for g in common:
            pos_ref = np.where(ref_ids == g)[0][0]
            pos_cur = np.where(cur_ids == g)[0][0]
            b_r = ref_b[pos_ref]
            b_c = cur_b[pos_cur]
            if np.isfinite(b_r) and np.isfinite(b_c):
                b_ref_vals.append(b_r)
                b_cur_vals.append(b_c)

        b_ref_arr = np.array(b_ref_vals)
        b_cur_arr = np.array(b_cur_vals)
        if b_ref_arr.size < 3:
            linked.append(cur)
            continue

        A = b_ref_arr.std() / (b_cur_arr.std() or 1.0)
        C = b_ref_arr.mean() - A * b_cur_arr.mean()

        theta_linked = A * cur["theta"] + C
        b_sub_linked = A * cur["b_sub"] + C
        a_sub_linked = cur["a_sub"] / A

        linked.append({
            "theta": theta_linked,
            "ids": cur_ids,
            "a_sub": a_sub_linked,
            "b_sub": b_sub_linked,
            "status": cur["status"]
        })

    return linked


# ───────────────────────────────────────────────────────────
# 5. Global metrics
# ───────────────────────────────────────────────────────────
def global_metrics(dgp, fits, per_test=False):
    """
    Compute RMSE, Pearson correlation, and Kendall's tau between true θ (from dgp)
    and estimated θ (from fits). For per_test=True, fits holds 'theta' sub‑vectors.
    Concatenate all tests and standardize θ_est to mean=0, sd=1.
    """
    true_list = []
    est_list = []
    if per_test:
        # cada fit es dict con 'theta' (solo N_k valores)
        for k, ft in enumerate(fits):
            if ft and ft["status"] != "POOR":
                true_list.append(dgp.theta[k])
                est_list.append(ft["theta"])
    else:
        # joint: fit["theta"] para cada test
        for k, fit in enumerate(fits):
            if fit and fit["status"] != "POOR":
                true_list.append(dgp.theta[k])
                est_list.append(fit["theta"])

    if not true_list:
        return None

    true_all = np.hstack(true_list)
    est_all = np.hstack(est_list)

    mask = np.isfinite(true_all) & np.isfinite(est_all)
    true_sel = true_all[mask]
    est_sel = est_all[mask]

    # Estandarizar θ_est
    mu_est = est_sel.mean()
    sd_est = est_sel.std(ddof=0)
    if sd_est == 0:
        est_std = est_sel - mu_est
    else:
        est_std = (est_sel - mu_est) / sd_est

    rmse_val = float(np.sqrt(mean_squared_error(true_sel, est_std)))
    pearson_val = float(pearsonr(true_sel, est_std)[0])
    kendall_val = float(kendalltau(true_sel, est_std)[0])

    return {"rmse": rmse_val, "pearson": pearson_val, "kendall": kendall_val, "n": int(mask.sum())}


# ───────────────────────────────────────────────────────────
# 6. BatchRunner
# ───────────────────────────────────────────────────────────
class BatchRunner:
    """
    Run IRT fitting for a list of datasets (dgp) and their metadata.
    Can run in joint mode (joint=True) or per‑test (joint=False).
    If link=True, apply link_mean or link_mean_sd over per‑test fits.
    """

    def __init__(self, datasets, metas, joint=False, link=True, verbose=False):
        self.datasets = datasets
        self.metas = metas
        self.joint = joint
        self.link = link
        self.verbose = verbose
        self.results = []

    def run(self):
        for dgp, meta in zip(self.datasets, self.metas):
            cfg = meta.get("config_id", "unknown")
            mode = "joint" if self.joint else "per-test"
            print(f"\nProcessing {cfg} ({mode})")

            if self.joint:
                fits = fit_dataset_joint(dgp, verbose=self.verbose)
                # En joint, link_mean no cambia nada porque b ya es global
                if self.link and not self.joint and any_common(dgp):
                    fits = link_mean(fits, dgp)
                m = global_metrics(dgp, fits, per_test=False)
                if m:
                    print(f"   ➜ RMSE={m['rmse']:.3f}  r={m['pearson']:.3f}  tau={m['kendall']:.3f}")
                else:
                    print("   ➜ Metrics unavailable (no valid tests)")
                self.results.append({"meta": meta, "metrics": m})
            else:
                # Modo per-test
                fits = []
                for k in range(dgp.K):
                    fit_k = fit_one_test(dgp, k, verbose=self.verbose)
                    fits.append(fit_k)
                if self.link:
                    # Para mean/mean linking, reemplazar por link_mean_sd si se prefiere
                    fits = link_mean(fits, dgp)
                m = global_metrics(dgp, fits, per_test=True)
                if m:
                    print(f"   ➜ RMSE={m['rmse']:.3f}  r={m['pearson']:.3f}  tau={m['kendall']:.3f}")
                else:
                    print("   ➜ Metrics unavailable (no valid tests)")
                self.results.append({"meta": meta, "metrics": m})

        return self.results

    def get_summary_table(self):
        """
        Build a DataFrame with one row per configuration:
        [Assignment, No. Items of Each Test, No. Common Items (Total),
         No. Examinees for Each Test (Total), Method, RMSE, Pearson, Kendall]
        """
        rows = []
        for entry in self.results:
            meta = entry["meta"]
            m = entry["metrics"]
            if m is None:
                continue

            assignment = meta.get("config_id", "")
            items_per_test = meta.get("item_count", "")
            common_items = meta.get("common_items", "")
            total_items = meta.get("total_items", "")
            ci_field = f"{common_items}"
            if total_items != "":
                ci_field = f"{common_items} ({total_items})"

            examinees_per_test = meta.get("examinee_count", "")
            total_examinees = ""
            if "examinee_count" in meta and "num_tests" in meta:
                total_examinees = meta["examinee_count"] * meta["num_tests"]
            et_field = f"{examinees_per_test}"
            if total_examinees != "":
                et_field = f"{examinees_per_test} ({total_examinees})"

            method = meta.get("method", "")

            rows.append({
                "Assignment": assignment,
                "No. Items of Each Test": items_per_test,
                "No. Common Items (Total No. Items)": ci_field,
                "No. Examinees for Each Test (Total No. Examinees)": et_field,
                "Method": method,
                "RMSE": round(m["rmse"], 4),
                "Pearson": round(m["pearson"], 4),
                "Kendall": round(m["kendall"], 4),
            })

        cols = [
            "Assignment",
            "No. Items of Each Test",
            "No. Common Items (Total No. Items)",
            "No. Examinees for Each Test (Total No. Examinees)",
            "Method",
            "RMSE",
            "Pearson",
            "Kendall",
        ]
        return pd.DataFrame(rows, columns=cols)


# ───────────────────────────────────────────────────────────
# 7. Main function
# ───────────────────────────────────────────────────────────
def main():
    loader = DataLoader()
    table2_datasets, table2_metas, _ = loader.load_table2()
    table3_datasets, table3_metas, _ = loader.load_table3()

    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    print("=== Running BatchRunner for Table 2 (per‑test) ===")
    runner2_pt = BatchRunner(
        datasets=table2_datasets,
        metas=table2_metas,
        joint=False,
        link=True,
        verbose=False,
    )
    runner2_pt.run()
    summary2_pt = runner2_pt.get_summary_table()
    csv_name2_pt = f"mcmc_table2_per-test_{timestamp}.csv"
    out2_pt = result_dir / csv_name2_pt
    summary2_pt.to_csv(out2_pt, index=False)
    print(f"✅ Table 2 per‑test CSV saved to: {out2_pt.resolve()}")

    print("\n=== Running BatchRunner for Table 3 (per‑test) ===")
    runner3_pt = BatchRunner(
        datasets=table3_datasets,
        metas=table3_metas,
        joint=False,
        link=True,
        verbose=False,
    )
    runner3_pt.run()
    summary3_pt = runner3_pt.get_summary_table()
    csv_name3_pt = f"mcmc_table3_per-test_{timestamp}.csv"
    out3_pt = result_dir / csv_name3_pt
    summary3_pt.to_csv(out3_pt, index=False)
    print(f"✅ Table 3 per‑test CSV saved to: {out3_pt.resolve()}")



if __name__ == "__main__":
    main()
