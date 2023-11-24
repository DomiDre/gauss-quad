//! Numerical integration using the Gauss-Hermite quadrature rule.
//!
//! This rule can integrate integrands of the form  
//! e^(-x^2) * f(x)  
//! over the domain (-∞, ∞).
//!
//! # Example
//! Integrate x^2 * e^(-x^2)
//! ```
//! use gauss_quad::hermite::GaussHermite;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussHermite::new(10);
//! let integral = quad.integrate(|x| x.powi(2));
//! assert_abs_diff_eq!(integral, core::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-14);
//! ```

pub mod iterators;
use iterators::{GaussHermiteIter, GaussHermiteNodes, GaussHermiteWeights};

use crate::{impl_data_api, DMatrixf64, Node, Weight, PI};

/// A Gauss-Hermite quadrature scheme.
///
/// These rules can integrate integrands of the form e^(-x^2) * f(x) over the domain (-∞, ∞).
/// # Example
/// Integrate e^(-x^2) * cos(x)
/// ```
/// # use gauss_quad::GaussHermite;
/// # use approx::assert_abs_diff_eq;
/// # use core::f64::consts::{E, PI};
/// // initialize a Gauss-Hermite rule with 20 nodes
/// let quad = GaussHermite::new(20);
///
/// // numerically integrate a function over (-∞, ∞) using the Gauss-Hermite rule
/// let integral = quad.integrate(|x| x.cos());
///
/// assert_abs_diff_eq!(integral, PI.sqrt() / E.powf(0.25), epsilon = 1e-14);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussHermite {
    node_weight_pairs: Vec<(Node, Weight)>,
}

impl GaussHermite {
    /// Initializes Gauss-Hermite quadrature rule of the given degree by computing the needed nodes and weights.
    ///
    /// Applies the Golub-Welsch algorithm to determine Gauss-Hermite nodes & weights.
    /// Constructs the companion matrix A for the Hermite Polynomial using the relation:
    /// 1/2 H_{n+1} + n H_{n-1} = x H_n
    /// A similar matrix that is symmetrized is constructed by D A D^{-1}
    /// Resulting in a symmetric tridiagonal matrix with
    /// 0 on the diagonal & sqrt(n/2) on the off-diagonal
    /// root & weight finding are equivalent to eigenvalue problem
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn new(deg: usize) -> GaussHermite {
        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let element = (idx_f64 * 0.5).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = element;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = element;
            }
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        // zip together the iterator over nodes with the one over weights and return as Vec<(f64, f64)>
        GaussHermite {
            node_weight_pairs: eigen
                .eigenvalues
                .iter()
                .copied()
                .zip(
                    (eigen.eigenvectors.row(0).map(|x| x * x) * PI.sqrt())
                        .iter()
                        .copied(),
                )
                .collect(),
        }
    }

    /// Perform quadrature of e^(-x^2) * `integrand` over the domain (-∞, ∞).
    pub fn integrate<F>(&self, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x_val, w_val)| integrand(*x_val) * w_val)
            .sum();
        result
    }
}

impl_data_api! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn iterator_sanity_check() {
        for deg in (10..=100).step_by(10) {
            let rule = GaussHermite::new(deg);
            assert_eq!(rule.degree(), deg);
            for ((ni, wi), (nn, ww)) in rule.iter().zip(rule.nodes().zip(rule.weights())) {
                assert_abs_diff_eq!(ni, nn);
                assert_eq!(wi, ww);
            }
            for ((ni, wi), (nn, ww)) in rule
                .as_node_weight_pairs()
                .iter()
                .zip(rule.nodes().zip(rule.weights()))
            {
                assert_abs_diff_eq!(ni, nn);
                assert_eq!(wi, ww);
            }
        }
    }

    #[test]
    fn golub_welsch_3() {
        let rule = GaussHermite::new(3);
        let x_should = [1.224_744_871_391_589, 0.0, -1.224_744_871_391_589];
        let w_should = [
            0.295_408_975_150_919_35,
            1.181_635_900_603_677_4,
            0.295_408_975_150_919_35,
        ];
        for (&node, x_val) in rule.nodes().zip(x_should) {
            assert_abs_diff_eq!(node, x_val, epsilon = 1e-15);
        }
        for (&weight, w_val) in rule.weights().zip(w_should) {
            assert_abs_diff_eq!(weight, w_val, epsilon = 1e-15);
        }
    }

    #[test]
    fn check_derives() {
        let quad = GaussHermite::new(10);
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussHermite::new(3);
        assert_ne!(quad, other_quad);
    }
}
