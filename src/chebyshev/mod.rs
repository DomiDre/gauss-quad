//! Numerical integration using the Gauss-Chebyshev quadrature rule.
//!
//! This rule can integrate formulas on the form f(x) * (1 - x^2)^`a` on finite intervals, where `a` is either -1/2 or 1/2.

use crate::{Node, Weight, __impl_node_weight_rule};

use core::{f64::consts::PI, fmt};
use std::backtrace::Backtrace;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// A Gauss-Chebyshev quadrature scheme of the first kind.
///
/// Used to integrate functions of the form
/// f(x) / sqrt(1 - x^2) on finite intervals.
///
/// # Example
///
/// ```
/// # use gauss_quad::chebyshev::{GaussChebyshevFirstKind, GaussChebyshevError};
/// # use approx::assert_relative_eq;
/// # use core::f64::consts::PI;
/// let rule = GaussChebyshevFirstKind::new(2)?;
///
/// assert_relative_eq!(rule.integrate(0.0, 2.0, |x| x), PI);
/// # Ok::<(), GaussChebyshevError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussChebyshevFirstKind {
    node_weight_pairs: Vec<(Node, Weight)>,
}

impl GaussChebyshevFirstKind {
    /// Create a new `GaussChebyshevFirstKind` rule that can integrate functions of the form f(x) / sqrt(1 - x^2).
    ///
    /// # Errors
    ///
    /// Returns an error if `degree` is less than 2.
    pub fn new(degree: usize) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError::new());
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
                .rev()
                .map(|i| ((PI * (2.0 * (i as f64) - 1.0) / (2.0 * n)).cos(), PI / n))
                .collect(),
        })
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    #[cfg(feature = "rayon")]
    /// Same as [`new`](Self::new) but runs in parallel.
    pub fn par_new(degree: usize) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError::new());
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
                .into_par_iter()
                .map(|i| ((PI * (2.0 * (i as f64) - 1.0) / (2.0 * n)).cos(), PI / n))
                .collect(),
        })
    }

    /// Returns the value of the integral of the given `integrand` in the inverval \[`a`, `b`\].
    ///
    /// # Example
    ///
    /// ```
    /// # use gauss_quad::chebyshev::{GaussChebyshevFirstKind, GaussChebyshevError};
    /// # use approx::assert_relative_eq;
    /// # use core::f64::consts::PI;
    /// let rule = GaussChebyshevFirstKind::new(2)?;
    ///
    /// assert_relative_eq!(rule.integrate(-1.0, 1.0, |x| 1.5 * x * x - 0.5), PI / 4.0);
    /// # Ok::<(), GaussChebyshevError>(())
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Self::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Sync + Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .par_iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }
}

__impl_node_weight_rule! {GaussChebyshevFirstKind, GaussChebyshevFirstKindNodes, GaussChebyshevFirstKindWeights, GaussChebyshevFirstKindIter, GaussChebyshevFirstKindIntoIter}

/// A Gauss-Chebyshev quadrature scheme of the second kind.
///
/// Used to integrate functions of the form
/// f(x) * sqrt(1 - x^2) on finite intervals.
///
/// # Example
///
/// ```
/// # use gauss_quad::chebyshev::{GaussChebyshevSecondKind, GaussChebyshevError};
/// # use approx::assert_relative_eq;
/// # use core::f64::consts::PI;
/// let rule = GaussChebyshevSecondKind::new(2)?;
///
/// assert_relative_eq!(rule.integrate(-1.0, 1.0, |x| x * x), PI / 8.0);
/// # Ok::<(), GaussChebyshevError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussChebyshevSecondKind {
    node_weight_pairs: Vec<(Node, Weight)>,
}

impl GaussChebyshevSecondKind {
    /// Create a new `GaussChebyshev` rule that can integrate functions of the form f(x) * sqrt(1 - x^2).
    ///
    /// # Errors
    ///
    /// Returns an error if `degree` is less than 2.
    pub fn new(degree: usize) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError::new());
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
                .rev()
                .map(|i| {
                    let over_n_plus_1 = 1.0 / (n + 1.0);
                    let sin_val = (PI * i as f64 * over_n_plus_1).sin();
                    (
                        (PI * i as f64 * over_n_plus_1).cos(),
                        PI * over_n_plus_1 * sin_val * sin_val,
                    )
                })
                .collect(),
        })
    }

    #[cfg(feature = "rayon")]
    /// Same as [`new`](Self::new) but runs in parallel.
    pub fn par_new(degree: usize) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError::new());
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
                .into_par_iter()
                .map(|i| {
                    let over_n_plus_1 = 1.0 / (n + 1.0);
                    let sin_val = (PI * i as f64 * over_n_plus_1).sin();
                    (
                        (PI * i as f64 * over_n_plus_1).cos(),
                        PI * over_n_plus_1 * sin_val * sin_val,
                    )
                })
                .collect(),
        })
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    /// Returns the value of the integral of the given `integrand` in the inverval \[`a`, `b`\].
    ///
    /// # Example
    ///
    /// ```
    /// # use gauss_quad::chebyshev::{GaussChebyshevSecondKind, GaussChebyshevError};
    /// # use approx::assert_relative_eq;
    /// # use core::f64::consts::PI;
    /// let rule = GaussChebyshevSecondKind::new(2)?;
    ///
    /// assert_relative_eq!(rule.integrate(-1.0, 1.0, |x| 1.5 * x * x - 0.5), -PI / 16.0);
    /// # Ok::<(), GaussChebyshevError>(())
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Self::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let result: f64 = self
            .node_weight_pairs
            .par_iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }
}

__impl_node_weight_rule! {GaussChebyshevSecondKind, GaussChebyshevSecondKindNodes, GaussChebyshevSecondKindWeights, GaussChebyshevSecondKindIter, GaussChebyshevSecondKindIntoIter}

/// The error returned when attempting to create a [`GaussChebyshevFirstKind`] or [`GaussChebyshevSecondKind`] struct with a degree less than 2.
#[derive(Debug)]
pub struct GaussChebyshevError(Backtrace);

impl GaussChebyshevError {
    pub(crate) fn new() -> Self {
        Self(Backtrace::capture())
    }

    /// Returns a [`Backtrace`] to where the error was created.
    ///
    /// This backtrace is captured with [`Backtrace::capture`], see it for more information about how to make it display information when printed.
    pub fn backtrace(&self) -> &Backtrace {
        &self.0
    }
}

impl fmt::Display for GaussChebyshevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree must be at least 2")
    }
}

impl std::error::Error for GaussChebyshevError {}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;

    use super::{GaussChebyshevFirstKind, GaussChebyshevSecondKind};

    use core::f64::consts::PI;

    #[test]
    fn check_error() {
        assert!(GaussChebyshevFirstKind::new(1).is_err());
        assert!(GaussChebyshevSecondKind::new(1).is_err());
    }

    #[test]
    fn check_sorted() {
        for deg in (2..100).step_by(10) {
            let rule1 = GaussChebyshevFirstKind::new(deg).unwrap();
            assert!(rule1.as_node_weight_pairs().is_sorted());
            let rule2 = GaussChebyshevSecondKind::new(deg).unwrap();
            assert!(rule2.as_node_weight_pairs().is_sorted());
        }
    }

    #[test]
    fn check_chebyshev_1st_deg_5() {
        // Source: https://mathworld.wolfram.com/Chebyshev-GaussQuadrature.html
        let ans = [
            (-0.5 * (0.5 * (5.0 + f64::sqrt(5.0))).sqrt(), PI / 5.0),
            (-0.5 * (0.5 * (5.0 - f64::sqrt(5.0))).sqrt(), PI / 5.0),
            (0.0, PI / 5.0),
            (0.5 * (0.5 * (5.0 - f64::sqrt(5.0))).sqrt(), PI / 5.0),
            (0.5 * (0.5 * (5.0 + f64::sqrt(5.0))).sqrt(), PI / 5.0),
        ];

        let rule = GaussChebyshevFirstKind::new(5).unwrap();

        for ((x, w), (x_should, w_should)) in rule.into_iter().zip(ans.into_iter()) {
            assert_abs_diff_eq!(x, x_should);
            assert_abs_diff_eq!(w, w_should);
        }
    }

    #[test]
    fn check_chebyshev_2nd_deg_5() {
        // I couldn't find lists of nodes and weights to compare to. So this function computes
        // them itself with formulas from Wikipedia.

        let deg = 5;
        let rule = GaussChebyshevSecondKind::new(deg).unwrap();
        let deg = deg as f64;

        for (i, (x, w)) in rule.into_iter().enumerate() {
            // Source: https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
            let ii = deg - i as f64;
            let x_should = (ii * PI / (deg + 1.0)).cos();
            let w_should = PI / (deg + 1.0) * (ii * PI / (deg + 1.0)).sin().powi(2);

            assert_abs_diff_eq!(x, x_should);
            assert_abs_diff_eq!(w, w_should);
        }
    }

    #[test]
    fn check_integral_of_line() {
        let rule = GaussChebyshevFirstKind::new(2).unwrap();

        assert_abs_diff_eq!(rule.integrate(0.0, 2.0, |x| x), PI);
    }

    #[test]
    fn check_integral_of_legendre_2() {
        let rule1 = GaussChebyshevFirstKind::new(2).unwrap();
        let rule2 = GaussChebyshevSecondKind::new(2).unwrap();

        fn legendre_2(x: f64) -> f64 {
            1.5 * x * x - 0.5
        }

        assert_abs_diff_eq!(rule1.integrate(-1.0, 1.0, legendre_2), PI / 4.0);
        assert_abs_diff_eq!(rule2.integrate(-1.0, 1.0, legendre_2), -PI / 16.0);
    }

    #[test]
    fn check_integral_of_parabola() {
        let rule = GaussChebyshevSecondKind::new(2).unwrap();

        assert_abs_diff_eq!(rule.integrate(-1.0, 1.0, |x| x * x), PI / 8.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_integrate() {
        let rule1 = GaussChebyshevFirstKind::par_new(2).unwrap();
        let rule2 = GaussChebyshevSecondKind::par_new(2).unwrap();

        assert_abs_diff_eq!(rule1.par_integrate(0.0, 2.0, |x| x), PI);
        assert_abs_diff_eq!(rule2.par_integrate(-1.0, 1.0, |x| x * x), PI / 8.0);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn check_par_error() {
        assert!(GaussChebyshevFirstKind::new(0).is_err());
        assert!(GaussChebyshevSecondKind::new(0).is_err());
    }
}
