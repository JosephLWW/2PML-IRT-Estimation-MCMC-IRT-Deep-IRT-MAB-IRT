# 2PML IRT Estimation: MCMC-IRT, Deep-IRT and MAB-IRT
Author: Joseph Wan-Wang
Research supervisor: Dr. Jeremy Miles (Google)

**Goal.** This repository (i) faithfully replicates the simulation experiments from **Deep‑IRT** (Tsutsumi, Kinoshita & Ueno, 2021) comparing a Bayesian 2PL (MCMC‑IRT) to a deep‑learning alternative, and (ii) contributes a conceptual extension that frames adaptive item selection as a contextual bandit via a **GLM‑UCB hybrid** (MAB‑IRT). All methods are evaluated under the same 2PL data‑generating process (DGP) and metrics for comparability.

---
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run any method (default settings replicate Table 2/3 conditions)
python MCMC_IRT_v01.py
python DEEP_IRT_v02.py
python CUCB_IRT_v03.py
```

---

## Repository Structure (minimal)

- `MCMC_IRT_v01.py` — Bayesian 2PL via MCMC/EAP (PyMC).
- `DEEP_IRT_v02.py` — PyTorch re‑implementation of Deep‑IRT.
- `CUCB_IRT_v03.py` — Contextual bandit (GLM‑UCB) for adaptive item selection (MAB‑IRT).
- `simulation_data/` (optional) — Previously generated datasets (if you prefer to not re‑simulate).
- `requirements.txt` — Python package requirements.
- `README.md` — this file.

---

## Experimental Design (What the scripts reproduce)

### 2PL Data‑Generating Process (DGP)
- Abilities: \(\theta_i \sim \mathcal{N}(0, 1)\)
- Items: \(\log a_j \sim \mathcal{N}(0, 1),\; b_j \sim \mathcal{N}(1, 0.4)\)
- Response model (2PL, scaling constant \(D=1.7\)):  
  \[ P(u_{ij}=1\mid\theta_i, a_j, b_j) = \sigma\big(D\, a_j(\theta_i - b_j)\big). \]

### Single‑Population scenarios ("Table 2")
- Test forms: \(K=10\) (no shared examinees across forms).
- Items per form \(J\in\{10,30,50\}\); examinees per form \(N\in\{50,100,500,1000\}\).
- Common (anchor) items between adjacent forms: \(C\in\{0,5\}\).
- Assignment to forms:
  - **Random** (i.i.d. abilities across forms), or
  - **Systematic** (sort by \(\theta\) and partition into 10 equal slices).

### Two‑Population extension ("Table 3")
- Balanced mixture of two Gaussians: \(\tfrac12\mathcal{N}(\mu_1,\sigma^2)+\tfrac12\mathcal{N}(\mu_2,\sigma^2)\).
- Grid (symmetric means, paired variances):
  - Means: \((\mu_1,\mu_2)\in\{(-0.3,+0.3),(-0.5,+0.5),(-0.7,+0.7),(-0.9,+0.9)\}\)
  - Variances: \(\sigma^2\in\{0.7,0.5,0.3,0.1\}\)
- Common items: \(C\in\{0,5\}\).

---

## Methods

### MCMC‑IRT (2PL baseline)
- Bayesian 2PL with priors \(\theta_i\!\sim\!\mathcal{N}(0,1),\; \log a_j\!\sim\!\mathcal{N}(0,1),\; b_j\!\sim\!\mathcal{N}(1,0.4)\).
- Posterior sampling (multiple chains; standard convergence diagnostics).
- **Linking**: when forms are calibrated separately, a **mean/mean (shift‑only)** post‑hoc transform aligns scales using shared items.

### Deep‑IRT (PyTorch replica)
- Dual‑MLP architecture mapping (examinee embedding, item embedding) → logit.
- Optimizer: Adam/AdamW; early stopping optional.
- **Standardization**: estimated abilities are z‑scored per form before RMSE.

### MAB‑IRT (GLM‑UCB hybrid)
- Poses item selection as a contextual bandit with a logistic GLM per item:  
  \( P(Y_{ij}=1\mid\hat{\theta}_i)=\sigma(\beta_{0j}+\beta_{1j}\hat{\theta}_i) \).
- Selects next item by an Upper‑Confidence Bound (UCB):  
  \(\text{UCB}_{ij}=\hat p_{ij}+\alpha\,\mathrm{SE}_{ij}\).
- **Linking**: post‑hoc affine alignment across forms via shared items.

---

## Evaluation & Reporting

- **Metrics** (per form): RMSE, Pearson’s \(r\), Kendall’s \(\tau\) between \(\hat{\theta}\) and true \(\theta\).
- **Standardization**: for Deep‑IRT (and when needed elsewhere), \(\hat{\theta}\) is standardized per form prior to RMSE to match the DGP scale.
- **Outputs**: each script writes per‑setting CSVs with the above metrics (and logs). Check script output for exact file paths/names.

---

## Reproducibility

- Global seed: **42**.
- Scripts log key hyperparameters and any convergence diagnostics.
- Hardware: CPU‑only is fine; Deep‑IRT benefits from GPU but does not require it.

---

## References (selection)
- Tsutsumi, E., Kinoshita, R., & Ueno, M. (2021). *Deep Item Response Theory…* **Electronics**, 10(9), 1020.
- van der Linden, W. J., & Barrett, M. D. (2016). Linking item response model parameters. **Psychometrika**, 81(3), 650–673.
- Filippi, S., Cappé, O., Garivier, A., & Szepesvári, C. (2010). Parametric bandits: The generalized linear case. **NeurIPS**.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual‑bandit approach to personalized news. **WWW**.

---

## Citation
If you use this code, please cite the repository and the papers above. Example:
> Wan-Wang, J. (2025). *2PML‑IRT Estimation: MCMC‑IRT, Deep‑IRT, and MAB‑IRT (Replication Package).* GitHub. Version 0.1

---

## License
TBD.
