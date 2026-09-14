

import numpy as np


def fgn_acf(n, H):
    k = np.arange(n)
    return 0.5 * (np.abs(k - 1) ** (2 * H)
                  + np.abs(k + 1) ** (2 * H)
                  - 2 * np.abs(k) ** (2 * H))


def fgn(q, H, T=1.0, rng=None, both=False):
    if rng is None:
        rng = np.random.default_rng()

    N = 2 ** q + 1
    M = 2 * (N - 1)

    
    rho = fgn_acf(N, H)
    c = np.concatenate([rho, rho[N - 2:0:-1]])

    
    lam = np.fft.fft(c).real
    if lam.min() < -1e-10:
        raise ValueError(
            f"circulant embedding failed: min eigenvalue {lam.min():.3e} < 0. "
            f"Increase q or use a larger embedding."
        )
    lam = np.clip(lam, 0.0, None)  
    
    zeta = rng.standard_normal(M)
    w = np.fft.fft(np.sqrt(lam) * np.fft.ifft(zeta))

    scale = (T / N) ** H
    first = w.real[:N - 1] * scale
    if not both:
        return first
    return first, w.imag[:N - 1] * scale


def fbm(q, H, T=1.0, rng=None):
    """Generate one fBm path on [0, T] with N-1 = 2**q points."""
    return np.cumsum(fgn(q, H, T, rng))


def _validate(q=8, H=0.7, n_paths=4000, seed=0):
    """Check the sampler against the theoretical variance and autocovariance."""
    rng = np.random.default_rng(seed)
    N = 2 ** q + 1
    inc = np.array([fgn(q, H, 1.0, rng) for _ in range(n_paths)])

    var_ratio = inc.var() / (1.0 / N) ** (2 * H)
    unit = inc / (1.0 / N) ** H
    lags = np.array([1, 2, 3])
    emp = [np.mean(unit[:, :-l] * unit[:, l:]) for l in lags]
    theo = fgn_acf(4, H)[1:]

    print(f"H = {H}")
    print(f"  variance ratio (should be 1): {var_ratio:.4f}")
    print(f"  acf empirical:   {np.round(emp, 4)}")
    print(f"  acf theoretical: {np.round(theo, 4)}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    q, T = 10, 1.0
    t = np.linspace(T / (2 ** q + 1), T, 2 ** q)

    fig, ax = plt.subplots(figsize=(8, 5))
    for H in (0.2, 0.5, 0.8):
        ax.plot(t, np.cumsum(fgn(q, H, T, rng)), lw=0.8, label=f"$H = {H}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$B_H(t)$")
    ax.set_title("Fractional Brownian motion (Wood-Chan circulant embedding)")
    ax.legend()
    fig.tight_layout()
    plt.show()

    for H in (0.3, 0.5, 0.7):
        _validate(H=H)
