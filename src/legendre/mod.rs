//! Numerical integration using the Gauss-Legendre quadrature rule.
//!
//! In Gauss-Legendre quadrature rules the integrand is evaluated at
//! the unique points such that a degree `n` rule can integrate
//! degree `2n - 1` degree polynomials exactly.
//!
//! Evaluation point x_i of a degree n rule is the i:th root
//! of Legendre polynomial P_n and its weight is  
//! w = 2 / ((1 - x_i)(P'_n(x_i))^2).
//!
//!
//! # Example
//! ```
//! use gauss_quad::legendre::GaussLegendre;
//! # use gauss_quad::legendre::GaussLegendreError;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLegendre::new(10)?;
//! let integral = quad.integrate(-1.0, 1.0,
//!     |x| 0.125 * (63.0 * x.powi(5) - 70.0 * x.powi(3) + 15.0 * x)
//! );
//! assert_abs_diff_eq!(integral, 0.0);
//! # Ok::<(), GaussLegendreError>(())
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

mod bogaert;

use bogaert::NodeWeightPair;

use crate::{impl_node_weight_rule, impl_node_weight_rule_iterators, Node, Weight};

/// A Gauss-Legendre quadrature scheme.
///
/// These rules can integrate functions on the domain [a, b].
///
/// # Examples
/// Basic usage:
/// ```
/// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
/// # use approx::assert_abs_diff_eq;
/// // initialize a Gauss-Legendre rule with 3 nodes
/// let quad = GaussLegendre::new(3)?;
///
/// // numerically integrate x^2 - 1/3 over the domain [0, 1]
/// let integral = quad.integrate(0.0, 1.0, |x| x * x - 1.0 / 3.0);
///
/// assert_abs_diff_eq!(integral, 0.0);
/// # Ok::<(), GaussLegendreError>(())
/// ```
/// The nodes and weights are computed in `O(n)` time,
/// so large quadrature rules are feasible:
/// ```
/// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
/// # use approx::assert_abs_diff_eq;
/// let quad = GaussLegendre::new(1_000_000)?;
/// let integral = quad.integrate(-3.0, 3.0, |x| x.sin());
/// assert_abs_diff_eq!(integral, 0.0);
/// # Ok::<(), GaussLegendreError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLegendre {
    node_weight_pairs: Vec<(Node, Weight)>,
}

impl GaussLegendre {
    /// Initializes a Gauss-Legendre quadrature rule of the given degree by computing the needed nodes and weights.
    ///
    /// Uses the [algorithm by Ignace Bogaert](https://doi.org/10.1137/140954969), which has linear time
    /// complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2.
    pub fn new(deg: usize) -> Result<Self, GaussLegendreError> {
        if deg < 2 {
            return Err(GaussLegendreError);
        }

        Ok(Self {
            node_weight_pairs: (1..deg + 1)
                .map(|k| NodeWeightPair::new(deg, k).into_tuple())
                .collect(),
        })
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    /// Same as [`new`](GaussLegendre::new) but runs in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2.
    pub fn par_new(deg: usize) -> Result<Self, GaussLegendreError> {
        if deg < 2 {
            return Err(GaussLegendreError);
        }

        Ok(Self {
            node_weight_pairs: (1..deg + 1)
                .into_par_iter()
                .map(|k| NodeWeightPair::new(deg, k).into_tuple())
                .collect(),
        })
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    /// Perform quadrature integration of given integrand from `a` to `b`.
    /// # Example
    /// Basic usage
    /// ```
    /// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
    /// # use approx::assert_abs_diff_eq;
    /// let glq_rule = GaussLegendre::new(3)?;
    /// assert_abs_diff_eq!(glq_rule.integrate(-1.0, 1.0, |x| x.powi(5)), 0.0);
    /// # Ok::<(), GaussLegendreError>(())
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x_val, w_val)| integrand(Self::argument_transformation(*x_val, a, b)) * w_val)
            .sum();
        Self::scale_factor(a, b) * result
    }

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    /// Same as [`integrate`](GaussLegendre::integrate) but runs in parallel.
    ///
    /// # Example
    ///
    /// ```
    /// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
    /// # use approx::assert_abs_diff_eq;
    /// let glq_rule = GaussLegendre::par_new(1_000_000)?;
    ///
    /// assert_abs_diff_eq!(glq_rule.par_integrate(0.0, 1.0, |x| x.ln()), -1.0, epsilon = 1e-12);
    /// # Ok::<(), GaussLegendreError>(())
    /// ```
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let result: f64 = self
            .node_weight_pairs
            .par_iter()
            .map(|(x_val, w_val)| integrand(Self::argument_transformation(*x_val, a, b)) * w_val)
            .sum();
        Self::scale_factor(a, b) * result
    }
}

impl_node_weight_rule! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

impl_node_weight_rule_iterators! {GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

/// The error returned by [`GaussLegendre::new`] if it's given a degree of 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLegendreError;

use core::fmt;
impl fmt::Display for GaussLegendreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the degree of the Gauss-Legendre quadrature rule must be at least 2"
        )
    }
}

impl std::error::Error for GaussLegendreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_degree_3() {
        let (x, w): (Vec<_>, Vec<_>) = GaussLegendre::new(3).unwrap().into_iter().unzip();

        let x_should = [0.7745966692414834, 0.0000000000000000, -0.7745966692414834];
        let w_should = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i]);
        }
    }

    #[test]
    fn check_legendre_error() {
        assert!(GaussLegendre::new(0).is_err());
        assert!(GaussLegendre::new(1).is_err());
    }

    #[test]
    fn check_derives() {
        let quad = GaussLegendre::new(10);
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussLegendre::new(3);
        assert_ne!(quad, other_quad);
    }
}
