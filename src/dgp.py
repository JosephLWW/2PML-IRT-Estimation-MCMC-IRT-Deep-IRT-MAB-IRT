"""
Data Generating Process for 2PLM with optional:
  - common_items: number of items shared between adjacent tests
  - systematic: if True, assigns examinees systematically by theta rank
"""
import numpy as np

class DGP2PLM:
    """
    Data Generating Process for 2PLM with optional:
      - common_items: number of items shared between adjacent tests
      - systematic: if True, assigns examinees systematically by theta rank
    """
    def __init__(self, num_items, num_examinees, num_tests=10,
                 common_items=0, systematic=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.K = num_tests
        self.J = num_items
        self.N = num_examinees
        self.common = common_items
        self.systematic = systematic

        self.theta = None        # (K, N)
        self.global_theta = None # (K*N,)
        self.a = None            # (total_items,)
        self.b = None            # (total_items,)
        self.u = None            # ✅ CAMBIO: Lista de matrices (N, J) por test
        self.item_ids = None     # list length K
        self.total_items = None
        self.assignments = None  # (K, N) if systematic

    def generate_parameters(self):
        # 1. Sample abilities
        self.theta = np.random.normal(0, 1, size=(self.K, self.N))
        if self.systematic:
            self.global_theta = self.theta.flatten()

        # 2. Build item_ids and determine total unique items
        if self.common > 0:
            self.build_local_overlap()
        else:
            self.item_ids = [
                np.arange(k * self.J, (k + 1) * self.J)
                for k in range(self.K)
            ]
            self.total_items = self.K * self.J

        # 3. Sample item parameters on global item pool
        log_a = np.random.normal(0, 1, size=self.total_items)
        self.a = np.exp(log_a)
        self.b = np.random.normal(1, np.sqrt(0.4), size=self.total_items)

    def generate_multi_population(self, mu_list, sigma2, seed=None):
        """
        Multi-population ability generation (Section 5.3).
        - mu_list: list of means [µ1, µ2, ..., µK] for each of the K tests
        - sigma2: common variance σ2 for each population
        - seed: optional RNG seed for reproducibility

        This method replaces the standard theta sampling with:
          θ_{k,i} ~ N(µ_k, σ2), for k=0..K-1, i=1..N
        """
        if seed is not None:
            np.random.seed(seed)
        if len(mu_list) != self.K:
            raise ValueError(f"Expected {self.K} means, got {len(mu_list)}")

        # Generate theta for each test from its own normal distribution
        std = np.sqrt(sigma2)
        self.theta = np.zeros((self.K, self.N))
        for k, mu in enumerate(mu_list):
            self.theta[k] = np.random.normal(mu, std, size=self.N)

        # Update global_theta for systematic assignment if needed
        if self.systematic:
            self.global_theta = self.theta.flatten()

        # Rebuild item mapping and sample item parameters
        if self.common > 0:
            self.build_local_overlap()
        else:
            self.item_ids = [
                np.arange(k * self.J, (k + 1) * self.J)
                for k in range(self.K)
            ]
            self.total_items = self.K * self.J

        log_a = np.random.normal(0, 1, size=self.total_items)
        self.a = np.exp(log_a)
        self.b = np.random.normal(1, np.sqrt(0.4), size=self.total_items)

    def build_local_overlap(self):
        shared = self.common
        self.item_ids = []
        next_id = 0
        for k in range(self.K):
            prev_shared = shared if k > 0 else 0
            next_shared = shared if k < self.K - 1 else 0
            unique = self.J - prev_shared - next_shared

            block = []
            # add shared items from previous test
            if prev_shared:
                block += self.item_ids[k-1][-prev_shared:].tolist()
            # add unique new items
            unique_block = list(range(next_id, next_id + unique))
            block += unique_block
            next_id += unique
            # add shared items for next test
            if next_shared:
                shared_block = list(range(next_id, next_id + next_shared))
                block += shared_block
                next_id += next_shared

            self.item_ids.append(np.array(block))

        self.total_items = next_id

    @staticmethod
    def p_correct(theta_ij, a_j, b_j, D=1.7):
        return 1.0 / (1 + np.exp(-D * a_j * (theta_ij - b_j)))

    def systematic_assign(self):
        KN = self.K * self.N
        sorted_idx = np.argsort(self.global_theta)
        base, rem = divmod(KN, self.K)
        sizes = [base + (1 if k < rem else 0) for k in range(self.K)]

        self.assignments = np.full((self.K, self.N), -1, dtype=int)
        ptr = 0
        for k, size in enumerate(sizes):
            block = sorted_idx[ptr:ptr+size]
            if size >= self.N:
                self.assignments[k] = block[:self.N]
            else:
                pad = np.full(self.N - size, -1, dtype=int)
                self.assignments[k] = np.concatenate([block, pad])
            ptr += size

    def simulate_responses(self):
        # ✅ CORRECCIÓN PRINCIPAL: Crear lista de matrices (N, J)
        self.u = []

        if self.systematic:
            self.systematic_assign()

            # reshape theta according to assignments
            new_theta = np.full((self.K, self.N), np.nan)
            for k in range(self.K):
                valid = self.assignments[k] >= 0
                idxs = self.assignments[k][valid]
                new_theta[k, :len(idxs)] = self.global_theta[idxs]
            self.theta = new_theta

            # ✅ CORRECCIÓN: simulate para cada test usando solo sus ítems
            for k in range(self.K):
                ids = self.item_ids[k]
                test_responses = np.zeros((self.N, self.J), dtype=int)
                
                for pos_i, gi in enumerate(self.assignments[k]):
                    if gi < 0:
                        raise ValueError(f"Emply slort for test {k} at position {pos_i}")
                    theta_i = self.global_theta[gi]
                    p = self.p_correct(theta_i, self.a[ids], self.b[ids])
                    test_responses[pos_i, :] = np.random.binomial(1, p)
                
                self.u.append(test_responses)
        else:
            # ✅ CORRECCIÓN: simulate para cada test usando solo sus ítems
            for k in range(self.K):
                ids = self.item_ids[k]
                test_responses = np.zeros((self.N, self.J), dtype=int)
                
                for i in range(self.N):
                    p = self.p_correct(self.theta[k, i], self.a[ids], self.b[ids])
                    test_responses[i, :] = np.random.binomial(1, p)
                
                self.u.append(test_responses)

    def summary(self):
        if self.systematic and self.assignments is not None:
            per_test = [(self.assignments[k] >= 0).sum() for k in range(self.K)]
        else:
            per_test = [self.N] * self.K

        # ✅ MEJORAR SUMMARY con información adicional
        base_summary = {
            'examinees_per_test': per_test,
            'total_examinees': sum(per_test),
            'common_items_per_adjacent': self.common,
            'total_items': self.total_items,
            'items_per_test': self.J,
            'tests': self.K
        }
        
        # Agregar información de las matrices de respuestas si están disponibles
        if hasattr(self, 'u') and self.u is not None:
            if isinstance(self.u, list):
                base_summary['response_matrices_shapes'] = [matrix.shape for matrix in self.u]
            else:
                base_summary['response_tensor_shape'] = self.u.shape
        
        return base_summary