# gauss-quad

[![Crates.io Version](https://img.shields.io/crates/v/gauss-quad?logo=Rust)](https://crates.io/crates/gauss-quad)
[![docs.rs](https://img.shields.io/docsrs/gauss-quad?logo=docs.rs)](https://docs.rs/gauss-quad/latest/gauss_quad/)
[![Github Repository Link](https://img.shields.io/badge/github-DomiDre%2Fgauss--quad-8da0cb?logo=github)](https://github.com/DomiDre/gauss-quad)
[![Build Status](https://github.com/domidre/gauss-quad/actions/workflows/rust.yml/badge.svg)](https://github.com/domidre/gauss-quad/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/DomiDre/gauss-quad/graph/badge.svg?token=YUP5Y77ER2)](https://codecov.io/gh/DomiDre/gauss-quad)

This crate lets you integrate functions quickly with [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).

Gaussian quadrature approximates an integral by only evaluating the integrand at a few carefully chosen points (called nodes),
multiplying the results with carefully chosen weights and summing the results.

Through different techniques for choosing the nodes and weights, different types of functions can be integrated while evaluating them
a very small number of times. 

This enables fast and efficient integration of functions.
A quadrature rule with n nodes can perfectly integrate a polynomial of degree 2n-1 or less,
and other functions are integrated well if they are well-approximated by a polynomial.


The crate currently supports the following Gaussian quadrature rules:


- [Gauss-Legendre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
- [Gauss-Jacobi](https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature)
- [Gauss-Laguerre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature) (generalized)
- [Gauss-Hermite](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
- [Gauss-Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature) of the first and second kinds

as well as the following [Newton-Cotes formulas](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas):

- [Midpoint](https://en.wikipedia.org/wiki/Riemann_sum#Midpoint_rule)
- [Trapezoid](https://en.wikipedia.org/wiki/Trapezoidal_rule)
- [Simpsons](https://en.wikipedia.org/wiki/Simpson%27s_rule)

The crate is `no_std` compatible (but needs an allocator due to its use of the `alloc` crate).

## Examples

Integrate x^2 from 0 to 1 with Gauss-Legendre quadrature:

```rust
use gauss_quad::GaussLegendre;
use core::num::NonZeroUsize;
use approx::assert_abs_diff_eq;

let integrator = GaussLegendre::new(2.try_into()?);

let integral = integrator.integrate(0.0, 1.0, |x| x * x);

assert_abs_diff_eq!(integral, 1.0/3.0);
# Ok::<Result<(), Box<dyn core::error::Error>>>(())
```

Integrate x^2 * e^(-x^2) over the whole real line:

```rust
use gauss_quad::GaussHermite;
use approx::assert_abs_diff_eq;

let quad = GaussHermite::new(10.try_into().unwrap());

let integral = quad.integrate(|x| x.powi(2));

assert_abs_diff_eq!(integral, core::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-14);
```

Rules can be nested into double and higher integrals:

```rust
let double_integral = integrator.integrate(a, b, |x| integrator.integrate(c(x), d(x), |y| f(x, y)));
```

If it takes a long time to evaluate the integrand (≫100 µs), the `rayon` feature can be used to parallelize the computation on multiple cores:

```rust
let slow_integral = integrator.par_integrate(a, b, |x| f(x));
```

This can also be nested with other rules to integrate double and higher integrals in parallel,
and the quadrature rules can be different as well:

```rust
let slow_double_integral = legendre_integrator.par_integrate(a, b, |x| hermite_integrator.integrate(|y| f(x, y)))
```

<br>

### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
</sub>
