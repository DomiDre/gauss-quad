//! Numerical integration using the Gauss-Hermite quadrature rule.
//!
//! This rule can integrate integrands of the form  
//! e^(-x^2) * f(x)  
//! over the domain (-∞, ∞).
//!
//! # Example
//!
//! Integrate x^2 * e^(-x^2)
//! ```
//! use gauss_quad::hermite::GaussHermite;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussHermite::new(10).unwrap();
//!
//! let integral = quad.integrate(|x| x.powi(2));
//!
//! assert_abs_diff_eq!(integral, core::f64::consts::PI.sqrt() / 2.0, epsilon = 1e-14);
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;

use crate::{
    DMatrixf64, Node, Weight, __impl_node_weight_rule, data_api::NODE_WEIGHT_RULE_INLINE_SIZE,
};

use core::f64::consts::PI;

/// A Gauss-Hermite quadrature scheme.
///
/// These rules can integrate integrands of the form e^(-x^2) * f(x) over the domain (-∞, ∞).
///
/// # Example
///
/// Integrate e^(-x^2) * cos(x)
/// ```
/// # use gauss_quad::hermite::GaussHermite;
/// # use approx::assert_abs_diff_eq;
/// # use core::f64::consts::{E, PI};
/// // initialize a Gauss-Hermite rule with 20 nodes
/// let quad = GaussHermite::new(20).unwrap();
///
/// // numerically integrate a function over (-∞, ∞) using the Gauss-Hermite rule
/// let integral = quad.integrate(|x| x.cos());
///
/// assert_abs_diff_eq!(integral, PI.sqrt() / E.powf(0.25), epsilon = 1e-14);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussHermite {
    node_weight_pairs: SmallVec<[(Node, Weight); NODE_WEIGHT_RULE_INLINE_SIZE]>,
}

impl GaussHermite {
    /// Initializes Gauss-Hermite quadrature rule of the given degree by computing the needed nodes and weights.
    ///
    /// A rule of degree n can integrate polynomials of degree 2n-1 exactly.
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
    /// Returns an error if `degree` is smaller than 2.
    pub fn new(degree: usize) -> Option<Self> {
        if degree < 2 {
            return None;
        }
        let mut companion_matrix = DMatrixf64::from_element(degree, degree, 0.0);
        // Initialize symmetric companion matrix
        for idx in 0..degree - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let element = (idx_f64 * 0.5).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = element;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = element;
            }
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        // zip together the iterator over nodes with the one over weights and collect
        let mut node_weight_pairs: SmallVec<[(Node, Weight); NODE_WEIGHT_RULE_INLINE_SIZE]> = eigen
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
            .collect();

        // sort the nodes and weights by the nodes
        node_weight_pairs
            .sort_unstable_by(|(node1, _), (node2, _)| node1.partial_cmp(node2).unwrap());

        Some(GaussHermite { node_weight_pairs })
    }

    /// Perform quadrature of e^(-x^2) * `integrand`(x) over the domain (-∞, ∞).
    pub fn integrate<F>(&self, mut integrand: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x_val, w_val)| integrand(*x_val) * w_val)
            .sum();
        result
    }

    #[cfg(feature = "rayon")]
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

__impl_node_weight_rule! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn check_sorted() {
        for deg in (2..100).step_by(10) {
            let rule = GaussHermite::new(deg).unwrap();
            assert!(rule.as_node_weight_pairs().is_sorted());
        }
    }

    #[test]
    fn golub_welsch_3() {
        let (x, w): (Vec<_>, Vec<_>) = GaussHermite::new(3).unwrap().into_iter().unzip();
        let x_should = [-1.224_744_871_391_589, 0.0, 1.224_744_871_391_589];
        let w_should = [
            0.295_408_975_150_919_35,
            1.181_635_900_603_677_4,
            0.295_408_975_150_919_35,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-15);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn check_hermite_error() {
        assert!(GaussHermite::new(0).is_none());
        assert!(GaussHermite::new(1).is_none());
    }

    #[test]
    fn check_derives() {
        let quad = GaussHermite::new(10).unwrap();
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussHermite::new(3).unwrap();
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_iterators() {
        let rule = GaussHermite::new(3).unwrap();
        let ans = core::f64::consts::PI.sqrt() / 2.0;

        assert_abs_diff_eq!(
            ans,
            rule.iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            ans,
            rule.nodes()
                .zip(rule.weights())
                .fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            ans,
            rule.into_iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );
    }

    #[test]
    fn integrate_one() {
        let quad = GaussHermite::new(5).unwrap();
        let integral = quad.integrate(|_x| 1.0);
        assert_abs_diff_eq!(integral, PI.sqrt(), epsilon = 1e-14);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_integrate_one() {
        let quad = GaussHermite::new(5).unwrap();
        let integral = quad.par_integrate(|_x| 1.0);
        assert_abs_diff_eq!(integral, PI.sqrt(), epsilon = 1e-15);
    }
}
