//! Numerical integration using the Chebyshev-Gauss quadrature rule.
//!
//! This rule can integrate formulas on the form (1 - x^2)^`a` f(x) where `a` is either -1/2 or 1/2,
//! on intervals finite intervals.

// We could use this to delegate some special cases of GaussJacobi.

use crate::{Node, Weight, __impl_node_weight_rule};

use core::{f64::consts::PI, fmt};

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
            return Err(GaussChebyshevError);
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
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
    pub fn par_new(degree: usize) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError);
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
                .into_par_iter()
                .map(|i| ((PI * (2.0 * (i as f64) - 1.0) / (2.0 * n)).cos(), PI / n))
                .collect(),
        })
    }

    /// Returns the value of the integral of the given `integrand` in the inverval [`a`, `b`].
    ///
    /// # Example
    ///
    /// ```
    /// # use gauss_quad::chebyshev::{GaussChebyshevFirstKind, GaussChebyshevError};
    /// # use approx::assert_relative_eq;
    /// # use core::f64::consts::PI;
    /// let rule = GaussChebyshevFirstKind::new(3)?;
    ///
    /// assert_relative_eq!(rule.integrate(-1.0, 1.0, |x| 1.5 * x * x - 0.5), PI / 4.0);
    /// # Ok::<(), GaussChebyshevError>(())
    /// ```
    pub fn integrate<F: Fn(f64) -> f64>(&self, a: f64, b: f64, integrand: F) -> f64 {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Self::integrate) but runs in parallel.
    pub fn par_integrate<F: Fn(f64) -> f64 + Sync>(&self, a: f64, b: f64, integrand: F) -> f64 {
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
            return Err(GaussChebyshevError);
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: (1..degree + 1)
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
            return Err(GaussChebyshevError);
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

    /// Returns the value of the integral of the given `integrand` in the inverval [`a`, `b`].
    ///
    /// # Example
    ///
    /// ```
    /// # use gauss_quad::chebyshev::{GaussChebyshevSecondKind, GaussChebyshevError};
    /// # use approx::assert_relative_eq;
    /// # use core::f64::consts::PI;
    /// let rule = GaussChebyshevSecondKind::new(3)?;
    ///
    /// assert_relative_eq!(rule.integrate(-1.0, 1.0, |x| 1.5 * x * x - 0.5), -PI / 16.0);
    /// # Ok::<(), GaussChebyshevError>(())
    /// ```
    pub fn integrate<F: Fn(f64) -> f64>(&self, a: f64, b: f64, integrand: F) -> f64 {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x, w)| integrand(Self::argument_transformation(*x, a, b)) * w)
            .sum();
        result * Self::scale_factor(a, b)
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Self::integrate) but runs in parallel.
    pub fn par_integrate<F: Fn(f64) -> f64 + Sync>(&self, a: f64, b: f64, integrand: F) -> f64 {
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
#[derive(Debug, Clone, Copy, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussChebyshevError;

impl fmt::Display for GaussChebyshevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree must be at least 2")
    }
}

impl std::error::Error for GaussChebyshevError {}
