//! Numerical integration using the generalized Gauss-Laguerre quadrature rule.
//!
//! A Gauss-Laguerre rule of degree `n` has nodes and weights chosen such that it
//! can integrate polynomials of degree 2`n`-1 exactly
//! with the weighing function w(x, `alpha`) = x^`alpha` * e^(-x) over the domain `[0, ∞)`.
//!
//! # Examples
//!
//! ```
//! use gauss_quad::laguerre::GaussLaguerre;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLaguerre::new(10.try_into().unwrap(), 1.0.try_into().unwrap());
//!
//! let integral = quad.integrate(|x| x.powi(2));
//!
//! assert_abs_diff_eq!(integral, 6.0, epsilon = 1e-14);
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::gamma::gamma;
use crate::{DMatrixf64, FiniteAboveNegOneF64, Node, Weight, __impl_node_weight_rule};

use core::num::NonZeroUsize;

/// A Gauss-Laguerre quadrature scheme.
///
/// These rules can perform integrals with integrands of the form x^alpha * e^(-x) * f(x) over the domain [0, ∞).
///
/// # Example
///
/// Compute the factorial of 5:
///
/// ```
/// # use gauss_quad::laguerre::GaussLaguerre;
/// # use approx::assert_abs_diff_eq;
/// // initialize a Gauss-Laguerre rule with 4 nodes
/// let quad = GaussLaguerre::new(4.try_into().unwrap(), 0.0.try_into().unwrap());
///
/// // numerically evaluate the integral x^5*e^(-x),
/// // which is a definition of the gamma function of six
/// let fact_5 = quad.integrate(|x| x.powi(5));
///
/// assert_abs_diff_eq!(fact_5, 1.0 * 2.0 * 3.0 * 4.0 * 5.0, epsilon = 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLaguerre {
    node_weight_pairs: Box<[(Node, Weight)]>,
    alpha: FiniteAboveNegOneF64,
}

impl GaussLaguerre {
    /// Initializes Gauss-Laguerre quadrature rule of the given degree by computing the nodes and weights
    /// needed for the given `alpha` parameter.
    ///
    /// A rule of degree n can integrate polynomials of degree 2n-1 exactly.
    ///
    /// Applies the Golub-Welsch algorithm to determine Gauss-Laguerre nodes & weights.
    /// Constructs the companion matrix A for the Laguerre Polynomial using the relation:
    /// -n L_{n-1} + (2n+1) L_{n} -(n+1) L_{n+1} = x L_n
    /// The constructed matrix is symmetric and tridiagonal with
    /// (2n+1) on the diagonal & -(n+1) on the off-diagonal (n = row number).
    /// Root & weight finding are equivalent to eigenvalue problem.
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn new(degree: NonZeroUsize, alpha: FiniteAboveNegOneF64) -> Self {
        let mut companion_matrix = DMatrixf64::from_element(degree.get(), degree.get(), 0.0);

        let mut diag = alpha.get() + 1.0;
        // Initialize symmetric companion matrix
        for idx in 0..degree.get() - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let off_diag = (idx_f64 * (idx_f64 + alpha.get())).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx)) = diag;
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = off_diag;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = off_diag;
            }
            diag += 2.0;
        }
        unsafe {
            *companion_matrix.get_unchecked_mut((degree.get() - 1, degree.get() - 1)) = diag;
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        let scale_factor = gamma(alpha.get() + 1.0);

        // zip together the iterator over nodes with the one over weights and return as Box<[(f64, f64)]>
        let mut node_weight_pairs: Box<[(f64, f64)]> = eigen
            .eigenvalues
            .into_iter()
            .copied()
            .zip(
                (eigen.eigenvectors.row(0).map(|x| x * x) * scale_factor)
                    .into_iter()
                    .copied(),
            )
            .collect();

        node_weight_pairs
            .sort_unstable_by(|(node1, _), (node2, _)| node1.partial_cmp(node2).unwrap());

        GaussLaguerre {
            node_weight_pairs,
            alpha,
        }
    }

    /// Perform quadrature of  
    /// x^`alpha` * e^(-x) * `integrand`(x)  
    /// over the domain `[0, ∞)`, where `alpha` was given in the call to [`new`](Self::new).
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
    /// Same as [`integrate`](GaussLaguerre::integrate) but runs in parallel.
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

    /// Returns the value of the `alpha` parameter of the rule.
    #[inline]
    pub const fn alpha(&self) -> FiniteAboveNegOneF64 {
        self.alpha
    }
}

__impl_node_weight_rule! {GaussLaguerre, GaussLaguerreNodes, GaussLaguerreWeights, GaussLaguerreIter, GaussLaguerreIntoIter}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use core::f64::consts::PI;

    #[test]
    fn check_sorted() {
        for deg in (2..100).step_by(10) {
            for alpha in [-0.9, -0.5, 0.0, 0.5] {
                let rule = GaussLaguerre::new(deg.try_into().unwrap(), alpha.try_into().unwrap());
                assert!(rule.as_node_weight_pairs().is_sorted());
            }
        }
    }

    #[test]
    fn check_degree_1() {
        let rule = GaussLaguerre::new(1.try_into().unwrap(), 0.0.try_into().unwrap());

        for constant in (0..100).step_by(10).map(f64::from) {
            assert_abs_diff_eq!(rule.integrate(|x| constant * x), constant, epsilon = 1e-13);
        }
    }

    #[test]
    fn golub_welsch_2_alpha_5() {
        let (x, w): (Vec<_>, Vec<_>) =
            GaussLaguerre::new(2.try_into().unwrap(), 5.0.try_into().unwrap())
                .into_iter()
                .unzip();
        let x_should = [4.354_248_688_935_409, 9.645_751_311_064_59];
        let w_should = [82.677_868_380_553_63, 37.322_131_619_446_37];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-12);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_0() {
        let (x, w): (Vec<_>, Vec<_>) =
            GaussLaguerre::new(3.try_into().unwrap(), 0.0.try_into().unwrap())
                .into_iter()
                .unzip();
        let x_should = [
            0.415_774_556_783_479_1,
            2.294_280_360_279_042,
            6.289_945_082_937_479_4,
        ];
        let w_should = [
            0.711_093_009_929_173,
            0.278_517_733_569_240_87,
            0.010_389_256_501_586_135,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_1_5() {
        let (x, w): (Vec<_>, Vec<_>) =
            GaussLaguerre::new(3.try_into().unwrap(), 1.5.try_into().unwrap())
                .into_iter()
                .unzip();
        let x_should = [
            1.220_402_317_558_883_8,
            3.808_880_721_467_068,
            8.470_716_960_974_048,
        ];
        let w_should = [
            0.730_637_894_350_016,
            0.566_249_100_686_605_7,
            0.032_453_393_142_515_25,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_negative() {
        let (x, w): (Vec<_>, Vec<_>) =
            GaussLaguerre::new(5.try_into().unwrap(), (-0.9).try_into().unwrap())
                .into_iter()
                .unzip();
        let x_should = [
            0.020_777_151_319_288_104,
            0.808_997_536_134_602_1,
            2.674_900_020_624_07,
            5.869_026_089_963_398,
            11.126_299_201_958_641,
        ];
        let w_should = [
            8.738_289_241_242_436,
            0.702_782_353_089_744_5,
            0.070_111_720_632_849_48,
            0.002_312_760_116_115_564,
            1.162_358_758_613_074_8E-5,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn check_derives() {
        let quad = GaussLaguerre::new(10.try_into().unwrap(), 1.0.try_into().unwrap());
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussLaguerre::new(10.try_into().unwrap(), 2.0.try_into().unwrap());
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_iterators() {
        let rule = GaussLaguerre::new(3.try_into().unwrap(), 0.5.try_into().unwrap());

        let ans = 15.0 / 8.0 * core::f64::consts::PI.sqrt();

        assert_abs_diff_eq!(
            rule.iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            ans,
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            rule.nodes()
                .zip(rule.weights())
                .fold(0.0, |tot, (n, w)| tot + n * n * w),
            ans,
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            rule.into_iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            ans,
            epsilon = 1e-14
        );
    }

    #[test]
    fn check_some_integrals() {
        let rule = GaussLaguerre::new(10.try_into().unwrap(), (-0.5).try_into().unwrap());

        assert_abs_diff_eq!(
            rule.integrate(|x| x * x),
            3.0 * PI.sqrt() / 4.0,
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            rule.integrate(|x| x.sin()),
            (PI.sqrt() * (PI / 8.0).sin()) / (2.0_f64.powf(0.25)),
            epsilon = 1e-7,
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_check_some_integrals() {
        let rule = GaussLaguerre::new(10.try_into().unwrap(), (-0.5).try_into().unwrap());

        assert_abs_diff_eq!(
            rule.par_integrate(|x| x * x),
            3.0 * PI.sqrt() / 4.0,
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            rule.par_integrate(|x| x.sin()),
            (PI.sqrt() * (PI / 8.0).sin()) / (2.0_f64.powf(0.25)),
            epsilon = 1e-7,
        );
    }
}
