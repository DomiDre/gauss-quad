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
//! # use gauss_quad::hermite::GaussHermiteError;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussHermite::new(10)?;
//! let integral = quad.integrate(|x| x.powi(2));
//! assert_abs_diff_eq!(integral, core::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-14);
//! # Ok::<(), GaussHermiteError>(())
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{impl_node_weight_rule, impl_node_weight_rule_iterators, DMatrixf64, Node, Weight, PI};

/// A Gauss-Hermite quadrature scheme.
///
/// These rules can integrate integrands of the form e^(-x^2) * f(x) over the domain (-∞, ∞).
/// # Example
/// Integrate e^(-x^2) * cos(x)
/// ```
/// # use gauss_quad::hermite::{GaussHermite, GaussHermiteError};
/// # use approx::assert_abs_diff_eq;
/// # use core::f64::consts::{E, PI};
/// // initialize a Gauss-Hermite rule with 20 nodes
/// let quad = GaussHermite::new(20)?;
///
/// // numerically integrate a function over (-∞, ∞) using the Gauss-Hermite rule
/// let integral = quad.integrate(|x| x.cos());
///
/// assert_abs_diff_eq!(integral, PI.sqrt() / E.powf(0.25), epsilon = 1e-14);
/// # Ok::<(), GaussHermiteError>(())
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
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2.
    pub fn new(deg: usize) -> Result<Self, GaussHermiteError> {
        if deg < 2 {
            return Err(GaussHermiteError);
        }
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
        Ok(GaussHermite {
            node_weight_pairs: eigen
                .eigenvalues
                .iter()
                .copied()
                .zip(
                    eigen
                        .eigenvectors
                        .row(0)
                        .map(|x| x * x * PI.sqrt())
                        .iter()
                        .copied(),
                )
                .collect(),
        })
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

    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    /// Same as [`integrate`](GaussHermite::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let result: f64 = self
            .node_weight_pairs
            .par_iter()
            .map(|(x_val, w_val)| integrand(*x_val) * w_val)
            .sum();
        result
    }
}

impl_node_weight_rule! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}

impl_node_weight_rule_iterators! {GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}

/// The error returned by [`GaussHermite::new`] if it is given a degree of 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussHermiteError;

use core::fmt;
impl fmt::Display for GaussHermiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the degree of the Gauss-Hermite quadrature rule must be at least 2"
        )
    }
}

impl std::error::Error for GaussHermiteError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golub_welsch_3() {
        let (x, w): (Vec<_>, Vec<_>) = GaussHermite::new(3).unwrap().into_iter().unzip();
        let x_should = [1.224_744_871_391_589, 0.0, -1.224_744_871_391_589];
        let w_should = [
            0.295_408_975_150_919_35,
            1.181_635_900_603_677_4,
            0.295_408_975_150_919_35,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-15);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn check_hermite_error() {
        assert!(GaussHermite::new(0).is_err());
        assert!(GaussHermite::new(1).is_err());
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
