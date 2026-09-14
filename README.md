# Fractional Brownian Motion and the S&P 500

Simulation of fractional Brownian motion by circulant embedding, and estimation of the
Hurst parameter of the S&P 500 index using discrete variation filters.

Term project, *Complex Systems*, Department of Physics, Sharif University of Technology,
Spring 2026.

## Contents

- `fractional-brownian-motion-sp500.pdf` — full report (in Persian; all figures and equations are in English/standard notation)
- `code/` — Python implementation (NumPy, SciPy, Matplotlib)
- `figures/` — generated figures

## What it does

**Simulation.** fBm paths are generated with the Wood–Chan circulant embedding method:
the Toeplitz covariance matrix of fractional Gaussian noise is embedded in a larger
circulant matrix, which is diagonalised by the DFT, so that sampling reduces to two FFTs
and an elementwise multiplication.

**Estimation.** The Hurst parameter is estimated from discrete variations. A filter of
order `r` is applied to remove the dependence between increments; the filter is then
dilated by factors `m`, and self-similarity implies that the variance of the filtered
series scales as `m^(2H)`. A log–log regression of the empirical variances against `m`
gives `2H` as its slope.

**Application.** S&P 500 daily closing prices are modelled as fractional geometric
Brownian motion, `S_t = S_0 exp(μt + σB^H_t)`. Taking logarithms turns the exponential
growth into a linear drift, which the order-2 increment filter annihilates exactly, so
the drift never enters the variance computation.

## Result

    H = 0.46 ± 0.07   (95% confidence)

The standard error was computed twice and independently: once from the explicit
asymptotic variance of the regression estimator, and once from 10^5 Monte Carlo
simulations of fBm at the estimated H. The two distributions agree closely
(simulated std 0.0350, explicit formula 0.0305), which is what the last figure of the
report shows.

## Authors

Mahdi Abolhasani, Hassan Kazemi, Bahar Ghasemzadeh.

My contribution: the Wood–Chan simulator, the discrete-variation estimator and its
Monte Carlo validation, and the S&P 500 analysis.

## References

1. B. B. Mandelbrot and J. W. Van Ness, *Fractional Brownian Motions, Fractional Noises
   and Applications*, SIAM Review 10(4), 422–437, 1968.
2. G. Shevchenko, *Fractional Brownian Motion in a Nutshell*, arXiv:1406.1956, 2014.
3. Y. S. Mishura, *Stochastic Calculus for Fractional Brownian Motion and Related
   Processes*, Lecture Notes in Mathematics 1929, Springer, 2008.
# fractional-brownian-motion-sp500
