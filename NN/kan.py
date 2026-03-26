import numpy as np


class KAN:
    """Simple one hidden layer KAN with wavelet edge functions (NumPy only)"""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_wavelets=8,
        wavelet="mexican_hat",
        seed=None,
        coeff_scale=0.1,
        input_domain=None,
        hidden_domain=(-2.0, 2.0),
    ):
        if seed is not None:
            np.random.seed(seed)

        if n_wavelets < 2:
            raise ValueError("n_wavelets must be >= 2")

        supported = ("mexican_hat", "morlet")
        if wavelet not in supported:
            raise ValueError(f"wavelet must be one of {supported}")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.n_wavelets = int(n_wavelets)
        self.wavelet = wavelet

        # Learnable wavelet-combination coefficients.
        self.c1 = coeff_scale * np.random.randn(self.input_size, self.hidden_size, self.n_wavelets)
        self.c2 = coeff_scale * np.random.randn(self.hidden_size, self.output_size, self.n_wavelets)

        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

        self._input_basis_ready = False
        self.centers1 = None
        self.scales1 = None

        if input_domain is not None:
            self.centers1, self.scales1 = self._make_feature_grids(self.input_size, input_domain)
            self._input_basis_ready = True

        self.centers2, self.scales2 = self._make_feature_grids(self.hidden_size, hidden_domain)

    @staticmethod
    def _safe_scales(scales):
        return np.maximum(scales, 1e-6)

    def _make_feature_grids(self, n_features, domain):
        lo, hi = domain
        if hi <= lo:
            raise ValueError("Invalid domain: upper bound must be > lower bound")

        centers_1d = np.linspace(lo, hi, self.n_wavelets)
        if self.n_wavelets == 1:
            spacing = hi - lo
        else:
            spacing = (hi - lo) / (self.n_wavelets - 1)
        base_scale = max(spacing, 1e-2)

        centers = np.tile(centers_1d[None, :], (n_features, 1))
        scales = np.full((n_features, self.n_wavelets), base_scale)
        return centers, scales

    def _init_input_basis_from_data(self, X):
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        centers = np.zeros((self.input_size, self.n_wavelets))
        scales = np.zeros((self.input_size, self.n_wavelets))

        for i in range(self.input_size):
            lo = float(mins[i])
            hi = float(maxs[i])
            if hi <= lo:
                hi = lo + 1e-2

            c = np.linspace(lo, hi, self.n_wavelets)
            spacing = (hi - lo) / (self.n_wavelets - 1)
            s = max(spacing, 1e-2)

            centers[i] = c
            scales[i] = s

        self.centers1 = centers
        self.scales1 = self._safe_scales(scales)
        self._input_basis_ready = True

    def _wavelet_and_derivative(self, u):
        if self.wavelet == "mexican_hat":
            # psi(u) = (1 - u^2) * exp(-u^2 / 2)
            exp_term = np.exp(-0.5 * u * u)
            psi = (1.0 - u * u) * exp_term
            dpsi_du = (u * u * u - 3.0 * u) * exp_term
            return psi, dpsi_du

        # morlet (real): psi(u) = cos(5u) * exp(-u^2 / 2)
        exp_term = np.exp(-0.5 * u * u)
        cos_term = np.cos(5.0 * u)
        sin_term = np.sin(5.0 * u)
        psi = cos_term * exp_term
        dpsi_du = exp_term * (-5.0 * sin_term - u * cos_term)
        return psi, dpsi_du

    def _basis(self, X, centers, scales):
        # X: (n, f), centers/scales: (f, k) -> basis: (n, f, k)
        safe_scales = self._safe_scales(scales)
        u = (X[:, :, None] - centers[None, :, :]) / safe_scales[None, :, :]
        psi, dpsi_du = self._wavelet_and_derivative(u)
        dpsi_dx = dpsi_du / safe_scales[None, :, :]
        return psi, dpsi_dx

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"X must have shape (n_samples, {self.input_size})")

        if not self._input_basis_ready:
            self._init_input_basis_from_data(X)

        b1, _ = self._basis(X, self.centers1, self.scales1)
        edge_1 = np.einsum("nik,ihk->nih", b1, self.c1)
        z1 = edge_1.sum(axis=1) + self.b1

        # KAN often uses simple node transforms; here we keep identity.
        a1 = z1

        b2, db2_da1 = self._basis(a1, self.centers2, self.scales2)
        edge_2 = np.einsum("nhk,hok->nho", b2, self.c2)
        y_pred = edge_2.sum(axis=1) + self.b2

        cache = {
            "X": X,
            "b1": b1,
            "z1": z1,
            "a1": a1,
            "b2": b2,
            "db2_da1": db2_da1,
        }
        return y_pred, cache

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    @staticmethod
    def _mse(y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(diff * diff)

    def loss(self, X, y_true):
        y_pred, _ = self.forward(X)
        y_true = np.asarray(y_true, dtype=float)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true must have shape {y_pred.shape}")
        return self._mse(y_pred, y_true)
