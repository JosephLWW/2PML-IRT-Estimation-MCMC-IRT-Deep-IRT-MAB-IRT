#!/usr/bin/env python3
"""DEEP_IRT_v01.py — Replication of the Deep‑IRT experiments (Tsutsumi, Kinoshita & Ueno, 2021).

• Loads the simulated datasets for Tables 2 and 3.
• Preprocesses response matrices ⇒ triplets (student_id, item_id, response).
• Trains a Deep‑IRT model per configuration.
• Evaluates θ accuracy via RMSE, Pearson ρ, and Kendall τ.
• Exports CSVs with exactly the columns shown in the paper:
    — results/deepirt_table2_<timestamp>.csv
    — results/deepirt_table3_<timestamp>.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import kendalltau, pearsonr
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader as TorchLoader, TensorDataset
from tqdm import tqdm

# ────────────────────────── Environment & paths ────────────────────────────────
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

RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

from data_loader import DataLoader  # noqa: E402

# ───────────────────────────── Deep‑IRT ─────────────────────────────────────
class _SubNet(nn.Module):
    """Two‑layer MLP (tanh) + embedding 50 → 50 → 1."""

    def __init__(self, n_entities: int, embed_dim: int = 50, hidden_dim: int = 50):
        super().__init__()
        self.embed = nn.Embedding(n_entities, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ids: Tensor) -> Tensor:  # (B,)
        return self.net(self.embed(ids)).squeeze(-1)


class DeepIRT(nn.Module):
    """PyTorch implementation faithful to the paper (fixed hyper‑parameters)."""

    HIDDEN_DIM = 50
    GAMMA = 0.1
    ALPHA_Le = 0.2
    ALPHA_He = 0.8
    ALPHA_Li = 0.2
    ALPHA_Hi = 0.8
    EPOCHS = 300
    LR = 1e-3
    BATCH = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    def __init__(self, n_students: int, n_items: int, *, batch_size: int = BATCH, device: str | torch.device = DEVICE):
        super().__init__()
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.student_net = _SubNet(n_students, self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.item_net = _SubNet(n_items, self.HIDDEN_DIM, self.HIDDEN_DIM)

        # Projection Δ → logits (fixed weights to preserve sign)
        self.out = nn.Linear(1, 2, bias=True)
        with torch.no_grad():
            self.out.weight.copy_(torch.tensor([[0.0], [1.0]]))
        self.out.weight.requires_grad_(False)

        self.to(device)
        self._device = torch.device(device)
        self._batch_size = batch_size

    # ────────────────────────── Forward & predict ────────────────────────
    def forward(self, student_ids: Tensor, item_ids: Tensor) -> Tensor:
        delta = self.student_net(student_ids) - self.item_net(item_ids)
        return self.out(delta.unsqueeze(-1))  # (B,2)

    @torch.inference_mode()
    def predict_proba(self, student_ids: Tensor, item_ids: Tensor) -> Tensor:
        logits = self.forward(student_ids.to(self._device), item_ids.to(self._device))
        return torch.softmax(logits, dim=1)[:, 1].cpu()

    # ───────────────────────────── Train ─────────────────────────────────
    def fit(self, triplets: np.ndarray) -> None:
        data = torch.as_tensor(triplets, dtype=torch.long, device=self._device)
        s_id, i_id, y = data[:, 0], data[:, 1], data[:, 2].float()

        # Hit‑rate weights γ (here fixed=0.1)
        weights = torch.full_like(y, self.GAMMA)

        loader = TorchLoader(
            TensorDataset(s_id, i_id, y.long(), weights),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
        )
        optim = torch.optim.AdamW(self.parameters(), lr=self.LR, weight_decay=1e-2)
        best_loss = float('inf')
        patience = 300          # disabled
        patience_counter = 0

        self.train()
        for epoch in range(self.EPOCHS):
            total = 0.0
            for s_b, i_b, y_b, w_b in loader:
                optim.zero_grad()
                logits = self.forward(s_b, i_b)
                loss_i = F.cross_entropy(logits, y_b, reduction="none")
                loss = (w_b * loss_i).mean()
                loss.backward()
                optim.step()
                total += loss.item() * y_b.size(0)
            avg_loss = total / len(y)          # mean loss for this epoch
            if avg_loss < best_loss:           # improvement ➜ reset counter
                best_loss = avg_loss
                patience_counter = 0
            else:                              # no improvement ➜ count
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"      Early-stopping at epoch {epoch+1}")
                    break                      # terminate training loop
            if (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch+1:3d}/{self.EPOCHS}  loss={total/len(y):.4f}")
        self.eval()

    # ─────────── Helpers post‑hoc ───────────
    @torch.inference_mode()
    def ability(self) -> np.ndarray:
        ids = torch.arange(self.student_net.embed.num_embeddings, device=self._device)
        return self.student_net(ids).cpu().numpy()

    @torch.inference_mode()
    def difficulty(self) -> np.ndarray:
        ids = torch.arange(self.item_net.embed.num_embeddings, device=self._device)
        return self.item_net(ids).cpu().numpy()

    @torch.inference_mode()
    def evaluate(self, theta_true: np.ndarray, beta_true: np.ndarray | None = None) -> Dict[str, Tuple[float, float, float]]:
        """Return metrics (RMSE, ρ, τ) for θ and β (if beta_true is not None)."""
        # θ
        theta_hat = self.ability()
        theta_hat = (theta_hat - theta_hat.mean()) / theta_hat.std()
        theta_true = (theta_true - theta_true.mean()) / theta_true.std()
        rmse_t = float(np.sqrt(((theta_hat - theta_true) ** 2).mean()))
        rho_t = float(pearsonr(theta_hat, theta_true)[0])
        tau_t = float(kendalltau(theta_hat, theta_true)[0])

        out = {"theta": (rmse_t, rho_t, tau_t)}

        if beta_true is not None:
            beta_hat = self.difficulty()
            beta_hat = (beta_hat - beta_hat.mean()) / beta_hat.std()
            beta_true = (beta_true - beta_true.mean()) / beta_true.std()
            rmse_b = float(np.sqrt(((beta_hat - beta_true) ** 2).mean()))
            rho_b = float(pearsonr(beta_hat, beta_true)[0])
            tau_b = float(kendalltau(beta_hat, beta_true)[0])
            out["beta"] = (rmse_b, rho_b, tau_b)
        return out

# ───────────────────────────── Utils ───────────────────────────────────────

def build_triplets(R: coo_matrix) -> np.ndarray:  # (N_obs,3)
    return np.vstack([R.row, R.col, R.data]).T.astype(np.int64)


def prepare_table2(datasets, metas):
    """Train Deep‑IRT and return a DataFrame with the exact columns of Table 2."""

    rows = []
    for dgp, meta in zip(datasets, metas):
        # ── Crear triplets + θ,β truth ────────────────────────────────────
        student_map, item_map = {}, {}
        r_rows, r_cols, r_vals = [], [], []

        for pop_idx, resp_matrix in enumerate(dgp.u):
            item_ids = dgp.item_ids[pop_idx]
            for stu_idx, resp_vector in enumerate(resp_matrix):
                s_key = (pop_idx, stu_idx)
                student_map.setdefault(s_key, len(student_map))
                s_id = student_map[s_key]
                for j_local, u_ij in enumerate(resp_vector):
                    g_it = int(item_ids[j_local])
                    item_map.setdefault(g_it, len(item_map))
                    it_id = item_map[g_it]
                    r_rows.append(s_id); r_cols.append(it_id); r_vals.append(int(u_ij))

        R = coo_matrix((r_vals, (r_rows, r_cols)), shape=(len(student_map), len(item_map)), dtype=np.int8)
        triplets = build_triplets(R)

        theta_true = np.zeros(len(student_map), dtype=np.float32)
        for (pop_idx, stu_idx), new_id in student_map.items():
            theta_true[new_id] = dgp.theta[pop_idx][stu_idx]

        beta_true = np.zeros(len(item_map), dtype=np.float32)
        for g_it, new_id in item_map.items():
            beta_true[new_id] = dgp.b[g_it]

        # ── Entrenar / evaluar ──────────────────────────────────────────
        print(f"\n[Table 2] {meta['method']} | J={meta['item_count']} | C={meta['common_items']}")
        model = DeepIRT(len(student_map), len(item_map))
        model.fit(triplets)
        rmse, rho, tau = model.evaluate(theta_true, beta_true)["theta"]

        rows.append({
            "Assignment": meta.get("method", ""),
            "No. Items of Each Test": meta["item_count"],
            "No. Common Items (Total No. Items)": f"{meta['common_items']} ({len(item_map)})",
            "No. Examinees for Each Test (Total No. Examinees)": f"{meta.get('examinee_count', '')} ({len(student_map)})",
            "Method": "Deep-IRT",
            "RMSE": round(rmse, 4),
            "Pearson": round(rho, 4),
            "Kendall": round(tau, 4),
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


def prepare_table3(datasets, metas):
    """Train Deep‑IRT and return a DataFrame with the columns of Table 3."""

    rows = []
    for dgp, meta in zip(datasets, metas):
        # Triplets + θ ground‑truth
        student_map, item_map = {}, {}
        r_rows, r_cols, r_vals = [], [], []

        for pop_idx, resp_matrix in enumerate(dgp.u):
            item_ids = dgp.item_ids[pop_idx]
            for stu_idx, resp_vector in enumerate(resp_matrix):
                s_key = (pop_idx, stu_idx)
                student_map.setdefault(s_key, len(student_map))
                s_id = student_map[s_key]
                for j_local, u_ij in enumerate(resp_vector):
                    g_it = int(item_ids[j_local])
                    item_map.setdefault(g_it, len(item_map))
                    it_id = item_map[g_it]
                    r_rows.append(s_id); r_cols.append(it_id); r_vals.append(int(u_ij))

        R = coo_matrix((r_vals, (r_rows, r_cols)), shape=(len(student_map), len(item_map)), dtype=np.int8)
        triplets = build_triplets(R)

        theta_true = np.zeros(len(student_map), dtype=np.float32)
        for (pop_idx, stu_idx), new_id in student_map.items():
            theta_true[new_id] = dgp.theta[pop_idx][stu_idx]

        # Entrenar / evaluar
        print(f"\n[Table 3] J={meta['item_count']} | C={meta['common_items']}")
        model = DeepIRT(len(student_map), len(item_map))
        model.fit(triplets)
        rmse, rho, tau = model.evaluate(theta_true)["theta"]

        # µ1, µ2, σ² taken from meta; handle varying formats robustly
        mu1, mu2 = np.nan, np.nan
        if "mu_list" in meta:
            try:
                mu_vals = [float(x) for x in str(meta["mu_list"]).strip("[]").split(",")]
                if len(mu_vals) >= 2:
                    mu1, mu2 = mu_vals[0], mu_vals[1]
            except Exception:
                pass
        sigma2 = meta.get("sigma2", np.nan)

        rows.append({
            "No. Examinees for Each Test": meta.get("examinee_count", ""),
            "No. Common Items": meta["common_items"],
            "µ1": mu1,
            "µ2": mu2,
            "σ2": sigma2,
            "Deep-IRT (RMSE)": round(rmse, 4),
            "Pearson": round(rho, 4),
            "Kendall": round(tau, 4),
        })

    cols = [
        "No. Examinees for Each Test",
        "No. Common Items",
        "µ1",
        "µ2",
        "σ2",
        "Deep-IRT (RMSE)",
        "Pearson",
        "Kendall",
    ]
    return pd.DataFrame(rows, columns=cols)

# ─────────────────────────────── MAIN ──────────────────────────────────────

def main() -> None:
    print("Loading datasets …")
    loader = DataLoader()
    datasets2, metas2, _ = loader.load_table2()
    datasets3, metas3, _ = loader.load_table3()

    df2 = prepare_table2(datasets2, metas2)
    df3 = prepare_table3(datasets3, metas3)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    f2 = RESULT_DIR / f"deepirt_table2_{timestamp}.csv"
    f3 = RESULT_DIR / f"deepirt_table3_{timestamp}.csv"
    df2.to_csv(f2, index=False)
    df3.to_csv(f3, index=False)

    print("\n✅ CSV files saved:")
    print(f"   • {f2.relative_to(PROJECT_ROOT)}")
    print(f"   • {f3.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()