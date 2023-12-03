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
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLegendre::new(10);
//! let integral = quad.integrate(-1.0, 1.0,
//!     |x| 0.125 * (63.0 * x.powi(5) - 70.0 * x.powi(3) + 15.0 * x)
//! );
//! assert_abs_diff_eq!(integral, 0.0);
//! ```

mod bogaert;

pub mod iterators;
use iterators::{
    GaussLegendreIntoIter, GaussLegendreIter, GaussLegendreNodes, GaussLegendreWeights,
};

use bogaert::NodeWeightPair;

use crate::{impl_node_weight_rule, Node, Weight};

/// A Gauss-Legendre quadrature scheme.
///
/// These rules can integrate functions on the domain [a, b].
///
/// # Examples
/// Basic usage:
/// ```
/// # use gauss_quad::GaussLegendre;
/// # use approx::assert_abs_diff_eq;
/// // initialize a Gauss-Legendre rule with 3 nodes
/// let quad = GaussLegendre::new(3);
///
/// // numerically integrate x^2 - 1/3 over the domain [0, 1]
/// let integral = quad.integrate(0.0, 1.0, |x| x * x - 1.0 / 3.0);
///
/// assert_abs_diff_eq!(integral, 0.0);
/// ```
/// The nodes and weights are computed in `O(n)` time,
/// so large quadrature rules are feasible:
/// ```
/// # use gauss_quad::GaussLegendre;
/// # use approx::assert_abs_diff_eq;
/// let quad = GaussLegendre::new(1_000_000);
/// let integral = quad.integrate(-3.0, 3.0, |x| x.sin());
/// assert_abs_diff_eq!(integral, 0.0);
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
    pub fn new(deg: usize) -> Self {
        Self {
            node_weight_pairs: (1..deg + 1)
                .map(|k| NodeWeightPair::new(deg, k).into_tuple())
                .collect(),
        }
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
    /// # use gauss_quad::GaussLegendre;
    /// # use approx::assert_abs_diff_eq;
    /// let glq_rule = GaussLegendre::new(3);
    /// assert_abs_diff_eq!(glq_rule.integrate(-1.0, 1.0, |x| x.powi(5)), 0.0);
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
}

impl_node_weight_rule! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn check_degree_3() {
        let rule = GaussLegendre::new(3);

        let x_should = [0.7745966692414834, 0.0000000000000000, -0.7745966692414834];
        let w_should = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
        for (&node, should) in rule.nodes().zip(x_should) {
            assert_abs_diff_eq!(node, should);
        }
        for (&weight, should) in rule.weights().zip(w_should) {
            assert_abs_diff_eq!(weight, should);
        }
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
