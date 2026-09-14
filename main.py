import numpy as np
import matplotlib.pyplot as plt

q = 10
H = 0.5
T = 1.0
def generate_fBm(q, h, T):
    N = 2**q+1
    M = 2 * (N - 1)
    k = np.arange(0,N)
    rho = 0.5 * (np.abs(k - 1)**(2*h) +np.abs(k+1)**(2*h)-2*np.abs(k)**(2*h))
    c = np.concatenate([rho[:N], rho[N-2:0:-1]])
    lambs = np.fft.fft(c).real
    zeta = np.random.normal(size=M)
    zeta = np.fft.fft(zeta)
    zeta *= np.sqrt(lambs)
    zeta = np.fft.fft(zeta).real[:N-1]
    zeta *= (T/N)**h
    return np.cumsum(zeta)

series = generate_fBm(q, H, T)
plt.plot(np.linspace(0,T,2**q), series)
plt.xlabel('$t$')
plt.ylabel('$B_H(t)$')
plt.title(f"Fractional Brownian Motion: H = {H}")
plt.show()
