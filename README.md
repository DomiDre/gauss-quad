# gauss-quad

[![Crates.io Version](https://img.shields.io/crates/v/gauss-quad?logo=Rust)](https://crates.io/crates/gauss-quad)
[![docs.rs](https://img.shields.io/docsrs/gauss-quad?logo=docs.rs)](https://docs.rs/gauss-quad/latest/gauss_quad/)
[![Build Status](https://github.com/domidre/gauss-quad/actions/workflows/rust.yml/badge.svg)](https://github.com/domidre/gauss-quad/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/DomiDre/gauss-quad/graph/badge.svg?token=YUP5Y77ER2)](https://codecov.io/gh/DomiDre/gauss-quad)

The `gauss-quad` crate is a small library to calculate integrals of the type

$$\int_a^b f(x) w(x) \mathrm{d}x$$

using [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).

Here $f(x)$ is a user supplied function
and $w(x)$ is a weight function that depends on which rule is used.
Gaussian quadrature is interesting because a rule of degree n can exactly integrate
all polynomials of degree 2n-1 or less.

To use the crate, the desired quadrature rule has to be included in the program, e.g. for a Gauss-Legendre rule

```rust
 use gauss_quad::GaussLegendre;
```

The general call structure is to first initialize the n-point quadrature rule setting the degree n via

```rust
 let quad = QUADRATURE_RULE::new(n)?;
```

where QUADRATURE_RULE can currently be set to calculate either:

| QUADRATURE_RULE | Integral                                                   |
| --------------- | ---------------------------------------------------------- |
| Midpoint        | $$\int_a^b f(x) \mathrm{d}x$$                              |
| Simpson         | $$\int_a^b f(x) \mathrm{d}x$$                              |
| GaussLegendre   | $$\int_a^b f(x) \mathrm{d}x$$                              |
| GaussJacobi     | $$\int_a^b f(x)(1-x)^\alpha (1&plus;x)^\beta \mathrm{d}x$$ |
| GaussLaguerre   | $$\int_{0}^\infty f(x)x^\alpha e^{-x} \mathrm{d}x$$        |
| GaussHermite    | $$\int_{-\infty}^\infty f(x) e^{-x^2} \mathrm{d}x$$        |

For the quadrature rules that take an additional parameter, such as Gauss-Laguerre and Gauss-Jacobi, the parameters have to be added to the initialization, e.g.

```rust
 let quad = GaussLaguerre::new(n, alpha)?;
```

Then to calculate the integral of a function call

```rust
let integral = quad.integrate(a, b, f(x));
```

where `a` and `b` (both `f64`) are the integral bounds and `f(x)` is the integrand which implements the trait `Fn(f64) -> f64`.
For example to integrate a parabola from 0 to 1 one can use a lambda expression as integrand and call:

```rust
let integral = quad.integrate(0.0, 1.0, |x| x * x);
```

If the integral is improper, as in the case of Gauss-Laguerre and Gauss-Hermite integrals, no integral bounds should be passed and the call simplifies to

```rust
let integral = quad.integrate(f(x));
```

Rules can be nested into double and higher integrals:

```rust
let double_integral = quad.integrate(a, b, |x| quad.integrate(c(x), d(x), |y| f(x, y)));
```

If the computation time for the evaluation of the integrand is large (≫100 µs), the `rayon` feature can be used to parallelize the computation on multiple cores (for quicker to compute integrands any gain is overshadowed by the overhead from parallelization)

```rust
let slow_integral = quad.par_integrate(a, b, |x| f(x));
```
