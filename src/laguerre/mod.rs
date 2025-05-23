//! Numerical integration using the generalized Gauss-Laguerre quadrature rule.
//!
//! A Gauss-Laguerre rule of degree `n` has nodes and weights chosen such that it
//! can integrate polynomials of degree 2`n`-1 exactly
//! with the weighing function w(x, `alpha`) = x^`alpha` * e^(-x) over the domain `[0, ∞)`.
//!
//! # Examples
//! ```
//! use gauss_quad::laguerre::GaussLaguerre;
//! # use gauss_quad::laguerre::GaussLaguerreError;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLaguerre::new(10, 1.0)?;
//!
//! let integral = quad.integrate(|x| x.powi(2));
//!
//! assert_abs_diff_eq!(integral, 6.0, epsilon = 1e-14);
//! # Ok::<(), GaussLaguerreError>(())
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::gamma::gamma;
use crate::{DMatrixf64, Node, Weight, __impl_node_weight_rule};

use std::backtrace::Backtrace;

/// A Gauss-Laguerre quadrature scheme.
///
/// These rules can perform integrals with integrands of the form x^alpha * e^(-x) * f(x) over the domain [0, ∞).
///
/// # Example
///
/// Compute the factorial of 5:
/// ```
/// # use gauss_quad::laguerre::{GaussLaguerre, GaussLaguerreError};
/// # use approx::assert_abs_diff_eq;
/// // initialize a Gauss-Laguerre rule with 4 nodes
/// let quad = GaussLaguerre::new(4, 0.0)?;
///
/// // numerically evaluate the integral x^5*e^(-x),
/// // which is a definition of the gamma function of six
/// let fact_5 = quad.integrate(|x| x.powi(5));
///
/// assert_abs_diff_eq!(fact_5, 1.0 * 2.0 * 3.0 * 4.0 * 5.0, epsilon = 1e-12);
/// # Ok::<(), GaussLaguerreError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLaguerre {
    node_weight_pairs: Vec<(Node, Weight)>,
    alpha: f64,
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
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2, or if `alpha` is smaller than -1.
    pub fn new(deg: usize, alpha: f64) -> Result<Self, GaussLaguerreError> {
        match (deg >= 2, (alpha.is_finite() && alpha > -1.0)) {
            (true, true) => Ok(()),
            (false, true) => Err(GaussLaguerreErrorReason::Degree),
            (true, false) => Err(GaussLaguerreErrorReason::Alpha),
            (false, false) => Err(GaussLaguerreErrorReason::DegreeAlpha),
        }
        .map_err(GaussLaguerreError::new)?;

        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);

        let mut diag = alpha + 1.0;
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let off_diag = (idx_f64 * (idx_f64 + alpha)).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx)) = diag;
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = off_diag;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = off_diag;
            }
            diag += 2.0;
        }
        unsafe {
            *companion_matrix.get_unchecked_mut((deg - 1, deg - 1)) = diag;
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        let scale_factor = gamma(alpha + 1.0);

        // zip together the iterator over nodes with the one over weights and return as Vec<(f64, f64)>
        let mut node_weight_pairs: Vec<(f64, f64)> = eigen
            .eigenvalues
            .into_iter()
            .copied()
            .zip(
                (eigen.eigenvectors.row(0).map(|x| x * x) * scale_factor)
                    .into_iter()
                    .copied(),
            )
            .collect();
        node_weight_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(GaussLaguerre {
            node_weight_pairs,
            alpha,
        })
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
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }
}

__impl_node_weight_rule! {GaussLaguerre, GaussLaguerreNodes, GaussLaguerreWeights, GaussLaguerreIter, GaussLaguerreIntoIter}

/// The error returned by [`GaussLaguerre::new`] if given a degree, `deg`, less than 2 and/or an `alpha` of -1 or less.
#[derive(Debug)]
pub struct GaussLaguerreError {
    reason: GaussLaguerreErrorReason,
    backtrace: Backtrace,
}

use core::fmt;
impl fmt::Display for GaussLaguerreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DEGREE_LIMIT: &str = "degree must be at least 2";
        const ALPHA_LIMIT: &str = "alpha must be larger than -1.0";
        match self.reason() {
            GaussLaguerreErrorReason::Degree => write!(f, "{DEGREE_LIMIT}"),
            GaussLaguerreErrorReason::Alpha => write!(f, "{ALPHA_LIMIT}"),
            GaussLaguerreErrorReason::DegreeAlpha => write!(f, "{DEGREE_LIMIT}, and {ALPHA_LIMIT}"),
        }
    }
}

impl GaussLaguerreError {
    /// Captures a backtrace and creates a new error with the given reason.
    #[inline]
    pub(crate) fn new(reason: GaussLaguerreErrorReason) -> Self {
        Self {
            reason,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns the reason for the error.
    #[inline]
    pub fn reason(&self) -> GaussLaguerreErrorReason {
        self.reason
    }

    /// Returns a [`Backtrace`] to where the error was created.
    ///
    /// This backtrace is captured with [`Backtrace::capture`], see it for more information about how to make it display information when printed.
    #[inline]
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl std::error::Error for GaussLaguerreError {}

/// The reason for the `GaussLaguerreError`, returned by the [`GaussLaguerreError::reason`] function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GaussLaguerreErrorReason {
    /// The given degree was less than 2.
    Degree,
    /// The given `alpha` exponent was less than or equal to -1.
    Alpha,
    /// The given degree was less than 2 and the given `alpha` exponent was less than or equal to -1.
    DegreeAlpha,
}

impl GaussLaguerreErrorReason {
    /// Returns true if the given degree, `deg`, was bad.
    #[inline]
    pub fn was_bad_degree(&self) -> bool {
        matches!(self, Self::Degree | Self::DegreeAlpha)
    }

    /// Returns true if the given `alpha` was bad.
    #[inline]
    pub fn was_bad_alpha(&self) -> bool {
        matches!(self, Self::Alpha | Self::DegreeAlpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use core::f64::consts::PI;

    #[test]
    fn golub_welsch_2_alpha_5() {
        let (x, w): (Vec<_>, Vec<_>) = GaussLaguerre::new(2, 5.0).unwrap().into_iter().unzip();
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
        let (x, w): (Vec<_>, Vec<_>) = GaussLaguerre::new(3, 0.0).unwrap().into_iter().unzip();
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
        let (x, w): (Vec<_>, Vec<_>) = GaussLaguerre::new(3, 1.5).unwrap().into_iter().unzip();
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
        let (x, w): (Vec<_>, Vec<_>) = GaussLaguerre::new(5, -0.9).unwrap().into_iter().unzip();
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
    fn check_laguerre_error() {
        // test error kinds, and if print outputs are as expected
        let laguerre_rule = GaussLaguerre::new(0, -0.25);
        assert!(laguerre_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussLaguerreErrorReason::Degree));
        assert_eq!(
            format!("{}", laguerre_rule.err().unwrap()),
            "degree must be at least 2"
        );
        assert_eq!(
            GaussLaguerre::new(0, -0.25).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::Degree)
        );

        assert_eq!(
            GaussLaguerre::new(1, -0.25).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::Degree)
        );

        let laguerre_rule = GaussLaguerre::new(5, -1.0);
        assert!(laguerre_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussLaguerreErrorReason::Alpha));
        assert_eq!(
            format!("{}", laguerre_rule.err().unwrap()),
            "alpha must be larger than -1.0"
        );

        assert_eq!(
            GaussLaguerre::new(5, -1.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::Alpha)
        );
        assert_eq!(
            GaussLaguerre::new(5, -2.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::Alpha)
        );

        let laguerre_rule = GaussLaguerre::new(0, -1.0);
        assert!(laguerre_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussLaguerreErrorReason::DegreeAlpha));
        assert_eq!(
            format!("{}", laguerre_rule.err().unwrap()),
            "degree must be at least 2, and alpha must be larger than -1.0"
        );

        assert_eq!(
            GaussLaguerre::new(0, -1.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::DegreeAlpha)
        );

        assert_eq!(
            GaussLaguerre::new(0, -2.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussLaguerre::new(1, -1.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussLaguerre::new(1, -2.0).map_err(|e| e.reason()),
            Err(GaussLaguerreErrorReason::DegreeAlpha)
        );
    }

    #[test]
    fn check_derives() {
        let quad = GaussLaguerre::new(10, 1.0).unwrap();
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussLaguerre::new(10, 2.0).unwrap();
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_iterators() {
        let rule = GaussLaguerre::new(3, 0.5).unwrap();

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
        let rule = GaussLaguerre::new(10, -0.5).unwrap();

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
        let rule = GaussLaguerre::new(10, -0.5).unwrap();

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
