"""GLMUCB_IRT_v03.py — GLM‑UCB HYBRID (logistic) with per‑test masking and randomized
block/student order.

Replicates Tables 2 and 3 of *Deep‑IRT* by replacing Deep‑IRT / MCMC‑IRT with
a GLM‑UCB HYBRID (pure “HYBRID” version per Li et al., 2010):
    • x_global = [θ̂] for the global parameter w (scalar),
    • z_item   = [1, θ̂] for item‑specific deviations β_j (2D),
    • logit: z = w·x_global + β_j·z_item.

    """

from __future__ import annotations
import os, sys, math, logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ───── logger ─────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("GLMUCB-HYBRID")

# ───── paths ─────
ROOT = Path(os.getenv("SIM_ROOT", Path.cwd())).expanduser()
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
RES  = ROOT / "results";  RES.mkdir(exist_ok=True)

from data_loader import DataLoader


# ───────── metric utilities ─────────
def _standardize(v: np.ndarray) -> np.ndarray:
    s = v.std(ddof=0)
    return (v - v.mean()) / (s if s > 0 else 1.0)

def compute_metrics(hat: np.ndarray, true: np.ndarray
                    ) -> Tuple[float, float, float]:
    hat, true = map(_standardize, (hat, true))
    return (float(np.sqrt(mean_squared_error(true, hat))),
            float(pearsonr(true, hat)[0]),
            float(kendalltau(true, hat)[0]))


# ─────────── GLM‑UCB HYBRID (pure, logistic) ───────────
class GLMUCBHybridPurist:
    """
    GLM-UCB HYBRID purista (Li+2010) para link logístico:
      • x_g = [θ̂]  (dim 1)  → parámetro global escalar w
      • z_j = [1, θ̂] (dim 2) → desvío específico β_j = (β0_j, β1_j)
      Logit: z = w * x_g + β_j ⋅ z_j  = (β0_j) + (w + β1_j)*θ̂

    UCB: width = L * α_t * sqrt( x_gᵀ Vinv x_g + z_jᵀ Ainv_j z_j )
      V   := información global (1x1);    A_j := información específica (2x2)
      Vinv, Ainv_j son las covarianzas aproximadas (inversos del Hessiano).
    """

    def __init__(self,
                 n_items: int,
                 n_students: int,
                 *,
                 alpha0: float = 1.0,     # factor UCB
                 eta: float   = 0.03,     # paso para θ̂
                 lmbda: Optional[float] = None,
                 theta_clip: float = 3.0,
                 beta_clip: float  = 3.0,
                 seed: int = 0,
                 global_state: Optional[dict] = None):
        # dims híbridas
        self.dg = 1  # x_global dim
        self.dz = 2  # z_item dim
        self.d_total = self.dg + self.dz

        self.alpha0 = float(alpha0)
        self.eta   = float(eta)
        self.lmbda = 1.0 if lmbda is None else float(lmbda)
        self.theta_clip = float(theta_clip)
        self.beta_clip  = float(beta_clip)

        # L = sup |μ'| para logística = 1/4
        self.L = 0.25
        # Suelo de curvatura (estabilidad IRLS)
        self.kappa_min = 1e-2  # ↑ respecto a v03

        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0., 1e-3, size=n_students)  # prior casi 0

        # ---- Estado global (w) compartido entre tests ----
        if global_state is None:
            # Inicialización viva: w≈1.0
            self.V    = self.lmbda * np.eye(self.dg)
            self.Vinv = np.linalg.inv(self.V)
            self.c    = np.zeros(self.dg)
            self.w    = np.array([1.0], dtype=float)  # w escalar (como vector dim 1)
            self.t    = 1
        else:
            self.V    = global_state["V"].copy()
            self.Vinv = global_state["Vinv"].copy()
            self.c    = global_state["c"].copy()
            self.w    = global_state["w"].copy()
            self.t    = int(global_state["t"])

        # ---- Estado por-ítem β_j para este bloque/test ----
        self.A    = np.stack([self.lmbda*np.eye(self.dz) for _ in range(n_items)])     # (J,2,2)
        self.Ainv = np.stack([np.linalg.inv(Aj)          for Aj in self.A])            # (J,2,2)
        self.b    = np.zeros((n_items, self.dz))                                      # (J,2)
        self.beta = np.zeros((n_items, self.dz))                                      # (J,2)
        # Jitter en interceptos, pendiente específica arranca en 0 (suficiente con w=1)
        self.beta[:, 0] = rng.normal(0.0, 1e-2, size=n_items)  # β0_j ~ N(0, 1e-2)
        self.beta[:, 1] = 0.0                                  # β1_j = 0

        # buffers
        self._xg = np.empty(self.dg, dtype=float)  # [θ]
        self._z  = np.empty(self.dz, dtype=float)  # [1, θ]
        self._rng = rng

    def get_global_state(self) -> dict:
        return {"V": self.V, "Vinv": self.Vinv, "c": self.c, "w": self.w, "t": self.t}

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def alpha_t(self) -> float:
        # usar dimensión total (1 + 2) en el término log
        return self.alpha0 * math.sqrt(self.d_total * math.log(1.0 + self.t / self.lmbda))

    # ----- contextos -----
    def ctx_global(self, s: int) -> np.ndarray:
        self._xg[0] = self.theta[s]
        return self._xg

    def ctx_item(self, s: int) -> np.ndarray:
        self._z[0] = 1.0
        self._z[1] = self.theta[s]
        return self._z

    # ----- util -----
    def eff_slope(self, j: int) -> float:
        # pendiente total d z / d θ = w + β1_j
        return float(self.w[0] + self.beta[j, 1])

    def _project_params(self, j: int):
        # Clip L2 de beta_j
        normb = float(np.linalg.norm(self.beta[j]))
        if normb > self.beta_clip:
            self.beta[j] *= (self.beta_clip / max(normb, 1e-12))
        # Asegurar pendiente total mínima positiva
        if self.eff_slope(j) < 1e-3:
            self.beta[j, 1] = 1e-3 - self.w[0]

    # ----- UCB -----
    def ucb_all(self, xg: np.ndarray, z: np.ndarray) -> np.ndarray:
        # logits y anchuras para todos los ítems
        logits = self.w[0] * xg[0] + (self.beta @ z)          # (J,)
        mu = self.sigmoid(logits)
        Vinv_xg = self.Vinv @ xg                              # (1,)
        base = float(xg @ Vinv_xg)                            # escalar
        Ainv_z = self.Ainv @ z                                # (J,2)
        quad = base + (z * Ainv_z).sum(axis=1)                # (J,)
        width = self.L * self.alpha_t() * np.sqrt(np.maximum(quad, 1e-12))
        return mu + width

    # ----- updates -----
    def update_global_and_item(self, j: int, xg: np.ndarray, z: np.ndarray, r: int):
        # Forward
        logit = self.w[0] * xg[0] + float(self.beta[j] @ z)
        mu = 1.0 / (1.0 + math.exp(-logit))
        wght = max(mu * (1.0 - mu), self.kappa_min)

        # Global (Vinv 1x1): SM con wght * xg xg^T
        Vx = self.Vinv @ xg
        denom_g = 1.0 + wght * float(xg @ Vx)
        self.Vinv -= (wght * np.outer(Vx, Vx)) / max(denom_g, 1e-12)
        self.V    += wght * np.outer(xg, xg)
        self.c    += (r - mu) * xg
        self.w     = self.Vinv @ self.c

        # Item j (Ainv 2x2)
        Ainvz = self.Ainv[j] @ z
        denom = 1.0 + wght * float(z @ Ainvz)
        self.Ainv[j] -= (wght * np.outer(Ainvz, Ainvz)) / max(denom, 1e-12)
        self.A[j]    += wght * np.outer(z, z)
        self.b[j]    += (r - mu) * z
        self.beta[j]  = self.Ainv[j] @ self.b[j]

        self._project_params(j)

    def update_theta(self, s: int, xg: np.ndarray, z: np.ndarray, r: int, j: int):
        logit = self.w[0] * xg[0] + float(self.beta[j] @ z)
        mu = 1.0 / (1.0 + math.exp(-logit))
        slope_tot = self.eff_slope(j)
        self.theta[s] += self.eta * (r - mu) * slope_tot
        self.theta[s] = float(np.clip(self.theta[s], -self.theta_clip, self.theta_clip))
        self.t += 1


# ───── helpers for θ and linking ─────
def theta_to_list(theta_obj, parts) -> List[np.ndarray]:
    if isinstance(theta_obj, list):
        return theta_obj
    arr = np.asarray(theta_obj)
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2 and arr.shape[0] == parts:
        return [arr[k] for k in range(parts)]
    raise ValueError(f"θ con shape inesperado {arr.shape}")

def link_thetas_via_common_items(item_ids_parts, beta_parts, theta_parts, ref_block=0):
    """
    Linking usando ítems comunes con 'coeficientes efectivos' de cada ítem:
    intercepto = β0_j, pendiente = (w + β1_j). beta_parts[k] debe venir como (J,2): [β0, slope_eff].
    """
    T = len(item_ids_parts)
    As, Bs = [1.0]*T, [0.0]*T
    linked = [theta_parts[ref_block]]

    ids_ref = item_ids_parts[ref_block]
    idx_ref = {iid: k for k, iid in enumerate(ids_ref)}
    beta_ref = beta_parts[ref_block]  # (Jr, 2) → [intercept, slope_eff]

    for t in range(T):
        if t == ref_block:
            continue
        ids_t = item_ids_parts[t]
        beta_t = beta_parts[t]

        commons = [iid for iid in ids_t if iid in idx_ref]
        if len(commons) < 3:
            linked.append(theta_parts[t])
            continue

        y1, y0 = [], []
        for iid in commons:
            j_t  = int(np.where(ids_t == iid)[0][0])
            j_r  = idx_ref[iid]
            slope_t, slope_r = beta_t[j_t,1], beta_ref[j_r,1]
            int_t,   int_r   = beta_t[j_t,0], beta_ref[j_r,0]
            if slope_r <= 1e-8 or slope_t <= 1e-8:
                continue
            y1.append(slope_t / slope_r)                 # ≈ A
            y0.append((int_t - int_r) / slope_t)        # ≈ B/A

        if len(y1) < 3 or len(y0) < 3:
            linked.append(theta_parts[t]); continue

        # mediana robusta
        A_hat  = float(np.median(y1))
        BA_hat = float(np.median(y0))
        if not np.isfinite(A_hat) or abs(A_hat) < 1e-6:
            linked.append(theta_parts[t]); continue
        B_hat  = BA_hat * A_hat

        linked.append(A_hat * theta_parts[t] + B_hat)

    return linked, As, Bs

def link_thetas_mean_sd(theta_parts, ref_block=0):
    mu_ref = float(np.mean(theta_parts[ref_block]))
    sd_ref = float(np.std(theta_parts[ref_block], ddof=0) or 1.0)
    linked = []
    for t, th in enumerate(theta_parts):
        if t == ref_block:
            linked.append(th); continue
        mu_t = float(np.mean(th))
        sd_t = float(np.std(th, ddof=0) or 1.0)
        linked.append(((th - mu_t) / sd_t) * sd_ref + mu_ref)
    return linked


# ───── bandit loop per block/test (pure HYBRID) ─────
def _choose_eta(n_items: int) -> float:
    # Heurística de paso por tamaño de bloque (como se discutió)
    if n_items <= 10:
        return 0.10
    if n_items <= 30:
        return 0.05
    return 0.03

def _replay_pass(bandit: GLMUCBHybridPurist, Ucsr: csr_matrix, rng: np.random.Generator):
    """Una pasada extra sobre TODOS los pares (s,j) del bloque (orden barajado)."""
    n_s, n_i = Ucsr.shape
    order_s = rng.permutation(n_s)
    order_j = rng.permutation(n_i)
    for s in order_s:
        row = Ucsr.getrow(s)
        correct_idx = set(row.indices)
        for j in order_j:
            xg = bandit.ctx_global(s)
            z  = bandit.ctx_item(s)
            r = 1 if j in correct_idx else 0
            bandit.update_global_and_item(j, xg, z, r)
            bandit.update_theta(s, xg, z, r, j)

def run_bandit_block_hybrid(U: csr_matrix,
                            *,
                            alpha0: float,
                            eta: Optional[float],
                            theta_clip: float, beta_clip: float,
                            lmbda: Optional[float],
                            seed: int,
                            global_state: Optional[dict],
                            shuffle_students: bool = True,
                            extra_replay_passes: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Ejecuta HYBRID purista en un bloque: z = w·x_g + β_j·z_j.
    Devuelve θ_hat (SIN estandarizar), coeficientes efectivos [β0, slope_eff] del bloque y el estado global actualizado.
    """
    n_s, n_i = U.shape
    eta_use = _choose_eta(n_i) if eta is None else eta

    bandit = GLMUCBHybridPurist(n_i, n_s,
                                alpha0=alpha0, eta=eta_use,
                                lmbda=lmbda, theta_clip=theta_clip, beta_clip=beta_clip,
                                seed=seed, global_state=global_state)
    Ucsr   = U.tocsr()
    rng    = np.random.default_rng(seed)

    alive   = np.empty(n_i, dtype=bool)
    jitter  = np.empty(n_i, dtype=float)

    # sanity checks
    nnz_row = np.diff(Ucsr.indptr)
    zero_correct = np.where(nnz_row == 0)[0]
    if len(zero_correct):
        log.warning("[Block] Students with 0 correct answers: %d (examples %s)",
                    len(zero_correct), zero_correct[:10])

    nnz_col = np.diff(Ucsr.tocsc().indptr)
    dead_items = np.where(nnz_col == 0)[0]
    if len(dead_items):
        log.warning("[Block] Items never answered correctly: %d (examples %s)",
                    len(dead_items), dead_items[:10])

    # Order of students (aleatorio si shuffle_students)
    stu_order = rng.permutation(n_s) if shuffle_students else np.arange(n_s)

    # ── Pase principal: cada alumno ve cada ítem una vez (orden por UCB) ──
    for s in tqdm(stu_order, desc="▶ students(block)", leave=False, disable=False):
        alive.fill(True)
        jitter[:] = rng.random(n_i)  # tie‑breaker

        row = Ucsr.getrow(s)
        correct_idx = set(row.indices)

        for _ in range(n_i):
            xg = bandit.ctx_global(s)
            z  = bandit.ctx_item(s)
            scores = bandit.ucb_all(xg, z)
            scores[~alive] = -np.inf
            j = int(np.argmax(scores + 1e-6 * jitter))

            r = 1 if j in correct_idx else 0
            bandit.update_global_and_item(j, xg, z, r)
            bandit.update_theta(s, xg, z, r, j)
            alive[j] = False

    # ── Extra passes (replay) over the same (x, r) pairs ──
    for _ in range(int(extra_replay_passes)):
        _replay_pass(bandit, Ucsr, rng)

    frac_clip = float(np.mean(np.abs(bandit.theta) >= (theta_clip - 1e-12)))
    if frac_clip > 0.10:
        log.warning("[Block] clipped θ̂: %.1f%%", 100*frac_clip)

    # Effective per‑item coefficients: [intercepto, pendiente_total]
    beta_total_block = np.empty((n_i, 2), dtype=float)
    beta_total_block[:, 0] = bandit.beta[:, 0]
    beta_total_block[:, 1] = bandit.w[0] + bandit.beta[:, 1]

    return bandit.theta.copy(), beta_total_block, bandit.get_global_state()


# ───── evaluation (train per‑block; optional pooling depending on assignment) ─────
def evaluate_hybrid(U_parts, theta_parts, *,
                    assignment: str,
                    item_ids_parts=None,
                    alpha0: float = 1.0, eta: Optional[float] = None,
                    theta_clip: float = 3.0, beta_clip: float = 3.0,
                    lmbda: float = 1.0, seed: int = 0,
                    shuffle_blocks: bool = True,
                    shuffle_students: bool = True,
                    extra_replay_passes: int = 2):
    """
    Train HYBRID per‑block. If assignment contains "system", DO NOT share global state across blocks.
    """
    theta_parts = theta_to_list(theta_parts, len(U_parts))
    T = len(U_parts)
    rng = np.random.default_rng(seed)

    # contenedores en índice de bloque ORIGINAL
    hat_raw_list:  List[Optional[np.ndarray]] = [None]*T
    beta_list:     List[Optional[np.ndarray]] = [None]*T
    true_list:     List[Optional[np.ndarray]] = [None]*T

    # Order of blocks/tests
    block_order = rng.permutation(T) if shuffle_blocks else np.arange(T)

    # Pooling: sólo si NO es systematic
    assign_str = (assignment or "").lower()
    do_pooling = not ("system" in assign_str)

    global_state = None  # estado global entre bloques

    for k in block_order:
        U = U_parts[k]
        th = theta_parts[k]
        U_csr = csr_matrix(U, dtype=np.uint8)

        # si no pooling, reseteamos global_state en cada bloque
        gs_in = global_state if do_pooling else None

        hat_block_raw, beta_block, gs_out = run_bandit_block_hybrid(
            U_csr,
            alpha0=alpha0, eta=eta,
            theta_clip=theta_clip, beta_clip=beta_clip,
            lmbda=lmbda, seed=seed + k,
            global_state=gs_in,
            shuffle_students=shuffle_students,
            extra_replay_passes=extra_replay_passes,
        )

        hat_raw_list[k] = hat_block_raw
        beta_list[k]    = beta_block
        true_list[k]    = np.asarray(th)
        global_state    = gs_out if do_pooling else None

    # Convertir a listas “llenas”
    hat_raw_list = [np.asarray(h) for h in hat_raw_list]
    beta_list    = [np.asarray(b) for b in beta_list]
    true_list    = [np.asarray(t) for t in true_list]

    # Linking post-hoc de θ̂
    if item_ids_parts is not None:
        linked_list, _, _ = link_thetas_via_common_items(item_ids_parts, beta_list, hat_raw_list, ref_block=0)
        # fallback si no se pudo linkar algún bloque
        if any(linked_list[k] is hat_raw_list[k] for k in range(T)):
            linked_fallback = link_thetas_mean_sd(hat_raw_list, ref_block=0)
            for k in range(T):
                if linked_list[k] is hat_raw_list[k]:
                    linked_list[k] = linked_fallback[k]
        hat = np.concatenate(linked_list)
    else:
        hat = np.concatenate(link_thetas_mean_sd(hat_raw_list, ref_block=0))

    true = np.concatenate(true_list)
    return compute_metrics(hat, true)


# ──────────────── formatting helpers (same as Deep‑IRT) ────────────────
def _format_table2_fields(meta: dict) -> Tuple[str, str]:
    """Devuelve campos compuestos 'No. Common Items (Total No. Items)' y
    'No. Examinees for Each Test (Total No. Examinees)' como en los scripts Deep-IRT/MCMC."""
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

    return ci_field, et_field


# ──────────────── main ────────────────
def main():
    loader, seed = DataLoader(), 42

    # ───── Tabla 2 ─────────────────────────────────
    ds2, meta2, _ = loader.load_table2()
    rows2: List[dict] = []

    for dgp, meta in zip(ds2, meta2):
        # item_ids por bloque (si expuestos por el loader)
        item_ids_parts = None
        try:
            if hasattr(loader, "get_item_ids_per_block"):
                item_ids_parts = loader.get_item_ids_per_block(dgp)
        except Exception:
            item_ids_parts = None

        rmse, rho, tau = evaluate_hybrid(
            [csr_matrix(u) for u in dgp.u],
            dgp.theta,
            assignment=meta.get("method", ""),
            item_ids_parts=item_ids_parts,
            alpha0=1.0, eta=None,                 # eta heurística por nº items
            theta_clip=3.0, beta_clip=3.0,
            lmbda=1.0, seed=seed,
            shuffle_blocks=True, shuffle_students=True,
            extra_replay_passes=2)

        ci_field, et_field = _format_table2_fields(meta)

        rows2.append({
            "Assignment": meta.get("method", ""),
            "No. Items of Each Test": meta.get("item_count", ""),
            "No. Common Items (Total No. Items)": ci_field,
            "No. Examinees for Each Test (Total No. Examinees)": et_field,
            "Method": "GLM-UCB (hybrid)",
            "RMSE": round(rmse, 4),
            "Pearson": round(rho, 4),
            "Kendall": round(tau, 4),
        })

    cols2 = [
        "Assignment",
        "No. Items of Each Test",
        "No. Common Items (Total No. Items)",
        "No. Examinees for Each Test (Total No. Examinees)",
        "Method",
        "RMSE",
        "Pearson",
        "Kendall",
    ]
    df2 = pd.DataFrame(rows2, columns=cols2)

    ts = datetime.now().strftime("%m%d_%H%M%S")
    csv2 = Path(RES, f"glmucb_hybrid_table2_{ts}.csv")
    df2.to_csv(csv2, index=False)
    log.info("Table 2 saved to %s", csv2.relative_to(ROOT))

    # ───── Tabla 3 ─────────────────────────────────
    ds3, meta3, _ = loader.load_table3()
    rows3: List[dict] = []

    for dgp, meta in zip(ds3, meta3):
        item_ids_parts = None
        try:
            if hasattr(loader, "get_item_ids_per_block"):
                item_ids_parts = loader.get_item_ids_per_block(dgp)
        except Exception:
            item_ids_parts = None

        rmse, rho, tau = evaluate_hybrid(
            [csr_matrix(u) for u in dgp.u],
            dgp.theta,
            assignment="joint",  # Table 3 se ejecuta en modo conjunto para formato
            item_ids_parts=item_ids_parts,
            alpha0=1.0, eta=None,
            theta_clip=3.0, beta_clip=3.0,
            lmbda=1.0, seed=seed,
            shuffle_blocks=True, shuffle_students=True,
            extra_replay_passes=2)

        # µ1, µ2, σ² exactamente como en Deep-IRT
        mu1, mu2 = np.nan, np.nan
        if "mu_list" in meta:
            try:
                mu_vals = [float(x) for x in str(meta["mu_list"]).strip("[]").split(",")]
                if len(mu_vals) >= 2:
                    mu1, mu2 = mu_vals[0], mu_vals[1]
            except Exception:
                pass
        sigma2 = meta.get("sigma2", np.nan)

        rows3.append({
            "No. Examinees for Each Test": meta.get("examinee_count", ""),
            "No. Common Items":           meta.get("common_items", ""),
            "µ1": mu1,
            "µ2": mu2,
            "σ2": sigma2,
            "GLM-UCB (RMSE)": round(rmse, 4),
            "Pearson": round(rho, 4),
            "Kendall": round(tau, 4),
        })

    cols3 = [
        "No. Examinees for Each Test",
        "No. Common Items",
        "µ1",
        "µ2",
        "σ2",
        "GLM-UCB (RMSE)",
        "Pearson",
        "Kendall",
    ]
    df3 = pd.DataFrame(rows3, columns=cols3)
    csv3 = Path(RES, f"glmucb_hybrid_table3_{ts}.csv")
    df3.to_csv(csv3, index=False)
    log.info("Table 3 saved to %s", csv3.relative_to(ROOT))


if __name__ == "__main__":
    main()
