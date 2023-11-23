//! # gauss-quad
//!
//! **gauss-quad** is a Gaussian quadrature library for numerical integration.
//!
//! ## Quadrature rules
//! **gauss-quad** implements the following quadrature rules:
//! * Gauss-Legendre
//! * Gauss-Jacobi
//! * Gauss-Laguerre
//! * Gauss-Hermite
//! * Midpoint
//! * Simpson
//!
//! ## Using **gauss-quad**
//! To use any of the quadrature rules in your project, first initialize the rule with
//! a specified degree and then you can use it for integration, e.g.:
//! ```
//! use gauss_quad::GaussLegendre;
//! // This macro is used in these docs to compare floats.
//! // The assertion succeeds if the two sides are within floating point error,
//! // or an optional epsilon.
//! use approx::assert_abs_diff_eq;
//!
//! // initialize the quadrature rule
//! let degree = 10;
//! let quad = GaussLegendre::new(degree);
//!
//! // use the rule to integrate a function
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//! assert_abs_diff_eq!(integral, 1.0 / 3.0);
//! ```
//!
//! ## Setting up a quadrature rule
//! Using a quadrature rule takes two steps:
//! 1. Initialization
//! 2. Integration
//!
//! First, rules must be initialized using some specific input parameters.
//!
//! Then, you can integrate functions using those rules:
//! ```
//! # use gauss_quad::*;
//! # let degree = 5;
//! # let alpha = 1.2;
//! # let beta = 1.2;
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
//! // no explicit domain, Gauss-Laguerre integration is done on the domain [0, ∞).
//! let piecewise = gauss_laguerre.integrate(|x| if x > 0.0 && x < 2.0 { x } else { 0.0 });
//!
//! let gauss_hermite = GaussHermite::new(degree);
//! // again, no explicit domain since integration is done over the domain (-∞, ∞).
//! let constant = gauss_hermite.integrate(|x| if x > -1.0 && x < 1.0 { 2.0 } else { 1.0 });
//! ```
//!
//! ## Specific quadrature rules
//! Different rules may take different parameters.
//!
//! For example, the `GaussLaguerre` rule requires both a `degree` and an `alpha`
//! parameter.
//!
//! `GaussLaguerre` is also defined as an improper integral over the domain [0, ∞).
//! This means no domain bounds are needed in the `integrate` call.
//! ```
//! # use gauss_quad::GaussLaguerre;
//! // initialize the quadrature rule
//! let degree = 10;
//! let alpha = 0.5;
//! let quad = GaussLaguerre::new(degree, alpha);
//!
//! // use the rule to integrate a function
//! let integral = quad.integrate(|x| x * x);
//! ```
//!
//! ## Panics and errors
//! Quadrature rules are only defined for a certain set of input values.
//! For example, every rule is only defined for degrees where `degree > 1`.
//! ```should_panic
//! # use gauss_quad::GaussLaguerre;
//! let degree = 1;
//! let quad = GaussLaguerre::new(degree, 0.1); // panics!
//! ```
//!
//! Specific rules may have other requirements.
//! `GaussJacobi` for example, requires alpha and beta parameters larger than -1.0.
//! ```should_panic
//! # use gauss_quad::GaussJacobi;
//! let degree = 10;
//! let alpha = 0.1;
//! let beta = -1.1;
//!
//! let quad = GaussJacobi::new(degree, alpha, beta); // panics!
//! ```
//! Make sure to read the specific quadrature rule's documentation before using it.
//!
//! Error handling is very simple: bad input values will cause the program to panic
//! and abort with a short error message.
//!
//! ## Passing functions to quadrature rules
//! The `integrate` method expects functions of the form `Fn(f64) -> f64`, i.e. functions of
//! one parameter.
//!
//! ```
//! # use gauss_quad::GaussLegendre;
//!
//! // initialize the quadrature rule
//! let degree = 10;
//! let quad = GaussLegendre::new(degree);
//!
//! // use the rule to integrate a function
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//! ```
//! # Features
//! `serde`: implements the [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) traits from
//! the [`serde`](https://crates.io/crates/serde) crate for the quatrature rule structs.

use nalgebra::{Dyn, Matrix, VecStorage};
type DMatrixf64 = Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>;
use core::f64::consts::PI;

mod data_api;
mod gamma;
#[cfg(test)]
mod gaussian_quadrature;
pub mod hermite;
pub mod jacobi;
pub mod laguerre;
pub mod legendre;
pub mod midpoint;
pub mod simpson;

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

/// A node in a quadrature rule.
pub type Node = f64;
/// A weight in a quadrature rule.
pub type Weight = f64;
