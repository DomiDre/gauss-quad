//! # gauss-quad
//!
//! **gauss-quad** is a [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature) library for numerical integration.
//!
//! ## Quadrature rules
//!
//! **gauss-quad** implements the following quadrature rules:
//! * [Gauss-Legendre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature)
//! * [Gauss-Jacobi](https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature)
//! * [Gauss-Laguerre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature) (generalized)
//! * [Gauss-Hermite](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
//! * [Gauss-Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature)
//! * [Trapezoid](https://en.wikipedia.org/wiki/Trapezoidal_rule)
//! * [Midpoint](https://en.wikipedia.org/wiki/Riemann_sum#Midpoint_rule)
//! * [Simpson](https://en.wikipedia.org/wiki/Simpson%27s_rule)
//!
//! ## Using **gauss-quad**
//!
//! To use any of the quadrature rules in your project, first initialize the rule with
//! a specified degree and then you can use it for integration. The degree
//! is a non-zero positive integer that determines the number of nodes used in the quadrature rule,
//! and as such the number of points at which the integrand is evaluated.
//!
//! ```
//! use gauss_quad::GaussLegendre;
//! // This macro is used in these docs to compare floats.
//! // The assertion succeeds if the two sides are within floating point error,
//! // or an optional epsilon.
//! use approx::assert_abs_diff_eq;
//! use core::num::NonZeroUsize;
//!
//! // initialize the quadrature rule
//! let degree = NonZeroUsize::new(10).unwrap();
//! let quad = GaussLegendre::new(degree);
//!
//! // Use the rule to integrate a function
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//! assert_abs_diff_eq!(integral, 1.0 / 3.0);
//! ```
//!
//! Select the degree, n, such that 2n-1 is the largest degree of polynomial that you want to integrate with the rule.
//!
//! ## Setting up a quadrature rule
//!
//! Using a quadrature rule takes two steps:
//! 1. Initialization
//! 2. Integration
//!
//! First, rules must be initialized using some specific input parameters.
//!
//! Then, you can integrate functions using those rules:
//!
//! ```
//! # use gauss_quad::*;
//! # let degree = core::num::NonZeroUsize::new(5).unwrap();
//! # let alpha = FiniteAboveNegOneF64::new(1.2).unwrap();
//! # let beta = FiniteAboveNegOneF64::new(1.2).unwrap();
//! # let a = 0.0;
//! # let b = 1.0;
//! # let c = -10.;
//! # let d = 100.;
//! let gauss_legendre = GaussLegendre::new(degree);
//! // Integrate on the domain [a, b]
//! let x_cubed = gauss_legendre.integrate(a, b, |x| x * x * x);
//!
//! let gauss_jacobi = GaussJacobi::new(degree, alpha, beta);
//! // Integrate on the domain [c, d]
//! let double_x = gauss_jacobi.integrate(c, d, |x| 2.0 * x);
//!
//! let gauss_laguerre = GaussLaguerre::new(degree, alpha);
//! // No explicit domain, Gauss-Laguerre integration is done on the domain [0, ∞).
//! let piecewise = gauss_laguerre.integrate(|x| if x > 0.0 && x < 2.0 { x } else { 0.0 });
//!
//! let gauss_hermite = GaussHermite::new(degree);
//! // Again, no explicit domain since Gauss-Hermite integration is done over the domain (-∞, ∞).
//! let golden_polynomial = gauss_hermite.integrate(|x| x * x - x - 1.0);
//! ```
//!
//! ## Different rules have different requirements
//!
//! Quadrature rules are only defined for a certain set of input values.
//! They take parameters of types that guarantee valid input.
//!
//! [`GaussJacobi`] for example, requires alpha and beta parameters larger than -1.0.
//! These are of type [`FiniteAboveNegOneF64`], which can only be created
//! from finite non-NAN values above -1.0:
//!
//! ```
//! use gauss_quad::{GaussJacobi, FiniteAboveNegOneF64};
//! use core::num::NonZeroUsize;
//!
//! let degree = NonZeroUsize::new(10).unwrap();
//! let alpha = FiniteAboveNegOneF64::new(0.1).unwrap();
//!
//! // Trying to create a `beta` below -1.0 results in `None`.
//! let beta = FiniteAboveNegOneF64::new(-1.1);
//! assert!(beta.is_none());
//!
//! // This is valid:
//! let beta = FiniteAboveNegOneF64::new(-0.5).unwrap();
//! let quad = GaussJacobi::new(degree, alpha, beta);
//! ```
//!
//! [`GaussLaguerre`] is used to evaluate an improper integral over the domain [0, ∞).
//! This means no domain bounds are needed in the `integrate` call.
//!
//! ```
//! # use gauss_quad::{GaussLaguerre, FiniteAboveNegOneF64};
//! # use approx::assert_abs_diff_eq;
//! # use core::f64::consts::PI;
//! # use core::num::NonZeroUsize;
//! // Initialize the quadrature rule
//! let degree = NonZeroUsize::new(2).unwrap();
//! let alpha = FiniteAboveNegOneF64::new(0.5).unwrap();
//! let quad = GaussLaguerre::new(degree, alpha);
//!
//! // Use the rule to integrate a function
//! let integral = quad.integrate(|x| x * x);
//!
//! assert_abs_diff_eq!(integral, 15.0 * PI.sqrt() / 8.0, epsilon = 1e-14);
//! ```
//!
//! Make sure to read the specific quadrature rule's documentation before using it.
//!
//! ## Passing functions to quadrature rules
//!
//! The `integrate` method takes any integrand that implements the [`FnMut(f64) -> f64`](FnMut) trait, i.e. functions of
//! one `f64` parameter.
//!
//! ```
//! # use gauss_quad::legendre::GaussLegendre;
//! # use approx::assert_abs_diff_eq;
//! # use core::num::NonZeroUsize;
//!
//! // Initialize the quadrature rule
//! let degree = NonZeroUsize::new(2).unwrap();
//! let quad = GaussLegendre::new(degree);
//!
//! // Use the rule to integrate a closure
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//!
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//!
//! assert_abs_diff_eq!(integral, 1.0 / 3.0);
//!
//! // You can also pass a function pointer
//! fn x_cubed(x: f64) -> f64 {
//!    x * x * x
//! }
//!
//! let integral_x_cubed = quad.integrate(left_bound, right_bound, x_cubed);
//! assert_abs_diff_eq!(integral_x_cubed, 1.0 / 4.0);
//! ```
//!
//! ## Double integrals
//!
//! It is possible to use this crate to do double and higher integrals:
//!
//! ```
//! # use gauss_quad::legendre::GaussLegendre;
//! # use approx::assert_abs_diff_eq;
//! let rule = GaussLegendre::new(3.try_into().unwrap());
//!
//! // integrate x^2*y over the triangle in the xy-plane where x ϵ [0, 1] and y ϵ [0, x]:
//! let double_int = rule.integrate(0.0, 1.0, |x| rule.integrate(0.0, x, |y| x * x * y));
//!
//! assert_abs_diff_eq!(double_int, 0.1);
//! ```
//!
//! However, the time complexity of the integration then scales with the number of nodes to
//! the power of the depth of the integral, e.g. O(n³) for triple integrals.
//!
//! ## Feature flags
//!
//! `serde`: implements the [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) traits from
//! the [`serde`](https://crates.io/crates/serde) crate for the quadrature rule structs.
//!
//! `rayon`: enables a parallel version of the `integrate` function on the quadrature rule structs. Can speed up integration if evaluating the integrand is expensive (takes ≫100 µs).

// Only enable the nighlty `doc_auto_cfg` feature when
// the `docsrs` configuration attribute is defined, which it is on docs.rs.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

use nalgebra::{Dyn, Matrix, VecStorage};

type DMatrixf64 = Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>;

pub mod chebyshev;
mod data_api;
mod gamma;
pub mod hermite;
pub mod jacobi;
pub mod laguerre;
pub mod legendre;
pub mod midpoint;
pub mod simpson;
pub mod trapezoid;

#[doc(inline)]
pub use chebyshev::{GaussChebyshevFirstKind, GaussChebyshevSecondKind};
#[doc(inline)]
pub use data_api::{
    FiniteAboveNegOneF64, InfNanNegOneOrLessError, Node, ParseFiniteAboveNegOneF64Error, Weight,
};
#[doc(inline)]
pub use hermite::GaussHermite;
#[doc(inline)]
pub use jacobi::GaussJacobi;
#[doc(inline)]
pub use laguerre::GaussLaguerre;
#[doc(inline)]
pub use legendre::GaussLegendre;
#[doc(inline)]
pub use midpoint::Midpoint;
#[doc(inline)]
pub use simpson::Simpson;
#[doc(inline)]
pub use trapezoid::Trapezoid;
