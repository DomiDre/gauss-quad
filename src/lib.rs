//! # gauss-quad
//!
//! **gauss-quad** is a Gaussian quadrature library for numerical integration.
//!
//! ## Quadrature rules
//!
//! **gauss-quad** implements the following quadrature rules:
//! * Gauss-Legendre
//! * Gauss-Jacobi
//! * Gauss-Laguerre
//! * Gauss-Hermite
//! * Midpoint
//! * Simpson
//!
//! ## Using **gauss-quad**
//!
//! To use any of the quadrature rules in your project, first initialize the rule with
//! a specified degree and then you can use it for integration, e.g.:
//! ```
//! use gauss_quad::GaussLegendre;
//! # use gauss_quad::legendre::GaussLegendreError;
//! // This macro is used in these docs to compare floats.
//! // The assertion succeeds if the two sides are within floating point error,
//! // or an optional epsilon.
//! use approx::assert_abs_diff_eq;
//!
//! // initialize the quadrature rule
//! let degree = 10;
//! let quad = GaussLegendre::new(degree)?;
//!
//! // use the rule to integrate a function
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//! assert_abs_diff_eq!(integral, 1.0 / 3.0);
//! # Ok::<(), GaussLegendreError>(())
//! ```
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
//! ```
//! # use gauss_quad::*;
//! # let degree = 5;
//! # let alpha = 1.2;
//! # let beta = 1.2;
//! # let a = 0.0;
//! # let b = 1.0;
//! # let c = -10.;
//! # let d = 100.;
//! let gauss_legendre = GaussLegendre::new(degree)?;
//! // Integrate on the domain [a, b]
//! let x_cubed = gauss_legendre.integrate(a, b, |x| x * x * x);
//!
//! let gauss_jacobi = GaussJacobi::new(degree, alpha, beta)?;
//! // Integrate on the domain [c, d]
//! let double_x = gauss_jacobi.integrate(c, d, |x| 2.0 * x);
//!
//! let gauss_laguerre = GaussLaguerre::new(degree, alpha)?;
//! // no explicit domain, Gauss-Laguerre integration is done on the domain [0, ∞).
//! let piecewise = gauss_laguerre.integrate(|x| if x > 0.0 && x < 2.0 { x } else { 0.0 });
//!
//! let gauss_hermite = GaussHermite::new(degree)?;
//! // again, no explicit domain since Gauss-Hermite integration is done over the domain (-∞, ∞).
//! let golden_polynomial = gauss_hermite.integrate(|x| x * x - x - 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Specific quadrature rules
//!
//! Different rules may take different parameters.
//!
//! For example, the `GaussLaguerre` rule requires both a `degree` and an `alpha`
//! parameter.
//!
//! `GaussLaguerre` is also defined as an improper integral over the domain [0, ∞).
//! This means no domain bounds are needed in the `integrate` call.
//! ```
//! # use gauss_quad::laguerre::{GaussLaguerre, GaussLaguerreError};
//! # use approx::assert_abs_diff_eq;
//! # use core::f64::consts::PI;
//! // initialize the quadrature rule
//! let degree = 2;
//! let alpha = 0.5;
//! let quad = GaussLaguerre::new(degree, alpha)?;
//!
//! // use the rule to integrate a function
//! let integral = quad.integrate(|x| x * x);
//!
//! assert_abs_diff_eq!(integral, 15.0 * PI.sqrt() / 8.0, epsilon = 1e-14);
//! # Ok::<(), GaussLaguerreError>(())
//! ```
//!
//! ## Errors
//!
//! Quadrature rules are only defined for a certain set of input values.
//! For example, every rule is only defined for degrees where `degree > 1`.
//! ```
//! # use gauss_quad::GaussLaguerre;
//! let degree = 1;
//! assert!(GaussLaguerre::new(degree, 0.1).is_err());
//! ```
//!
//! Specific rules may have other requirements.
//! `GaussJacobi` for example, requires alpha and beta parameters larger than -1.0.
//! ```
//! # use gauss_quad::jacobi::GaussJacobi;
//! let degree = 10;
//! let alpha = 0.1;
//! let beta = -1.1;
//!
//! assert!(GaussJacobi::new(degree, alpha, beta).is_err())
//! ```
//! Make sure to read the specific quadrature rule's documentation before using it.
//!
//! ## Passing functions to quadrature rules
//!
//! The `integrate` method takes any integrand that implements the [`Fn(f64) -> f64`](Fn) trait, i.e. functions of
//! one `f64` parameter.
//!
//! ```
//! # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
//! # use approx::assert_abs_diff_eq;
//!
//! // initialize the quadrature rule
//! let degree = 2;
//! let quad = GaussLegendre::new(degree)?;
//!
//! // use the rule to integrate a function
//! let left_bound = 0.0;
//! let right_bound = 1.0;
//!
//! let integral = quad.integrate(left_bound, right_bound, |x| x * x);
//!
//! assert_abs_diff_eq!(integral, 1.0 / 3.0);
//! # Ok::<(), GaussLegendreError>(())
//! ```
//!
//! ## Double integrals
//!
//! It is possible to use this crate to do double and higher integrals:
//! ```
//! # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
//! # use approx::assert_abs_diff_eq;
//! let rule = GaussLegendre::new(3)?;
//!
//! // integrate x^2*y over the triangle in the xy-plane where x ϵ [0, 1] and y ϵ [0, x]:
//! let double_int = rule.integrate(0.0, 1.0, |x| rule.integrate(0.0, x, |y| x * x * y));
//!
//! assert_abs_diff_eq!(double_int, 0.1);
//! # Ok::<(), GaussLegendreError>(())
//! ```
//! However, the time complexity of the integration then scales with the number of nodes to
//! the power of the depth of the integral, e.g. O(n³) for triple integrals.
//!
//! ## Feature flags
//!
//! `serde`: implements the [`Serialize`](serde::Serialize) and [`Deserialize`](serde::Deserialize) traits from
//! the [`serde`](https://crates.io/crates/serde) crate for the quadrature rule structs.

use nalgebra::{Dyn, Matrix, VecStorage};

type DMatrixf64 = Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>;

mod data_api;
mod gamma;
pub mod hermite;
pub mod jacobi;
pub mod laguerre;
pub mod legendre;
pub mod midpoint;
pub mod simpson;

#[doc(inline)]
pub use data_api::{Node, Weight};
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
