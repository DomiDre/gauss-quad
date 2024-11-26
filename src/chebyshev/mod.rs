//! Numerical integration using the Chebyshev-Gauss quadrature rule.
//!
//! This rule can integrate formulas on the form (1 - x^2)^`a` f(x) where `a` is either -1/2 or 1/2,
//! on intervals finite intervals.

// We could use this to delegate some special cases of GaussJacobi.

use crate::{__impl_node_weight_rule, Node, Weight};

use core::{f64::consts::PI, fmt};

/// A Gauss-Chebyshev quadrature scheme used to integrate functions of the form
/// (1 - x^2)^`a` f(x) where `a` is either -1/2 or 1/2, on finite intervals.
///
/// # Example
///
/// ```
/// # use gauss_quad::chebyshev::{GaussChebyshev, GaussChebyshevForm, GaussChebyshevError};
/// # use approx::assert_relative_eq;
/// # use core::f64::consts::PI;
/// let rule = GaussChebyshev::new(5, GaussChebyshevForm::Divide)?;
///
/// assert_relative_eq!(rule.integrate(0.0, 2.0, |x| x), PI);
/// # Ok::<(), GaussChebyshevError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussChebyshev {
    node_weight_pairs: Vec<(Node, Weight)>,
    form: GaussChebyshevForm,
}

/// The error returned when attempting to create a [`GaussChebyshev`] struct with a degree less than 2.
#[derive(Debug, Clone, Copy, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussChebyshevError;

impl fmt::Display for GaussChebyshevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree must be at least 2")
    }
}

impl std::error::Error for GaussChebyshevError {}

/// Determines which form of function the [`GaussChebyshev`] quadrature rule struct should integrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GaussChebyshevForm {
    /// Integrate functions of the form `f(x) * sqrt(1 - x^2)`
    Multiply,
    /// Integrate functions of the form `f(x) / sqrt(1 - x^2)`
    Divide,
}

impl GaussChebyshev {
    /// Create a new `GaussChebyshev` rule that can integrate functions of the chosen form.
    ///
    /// # Errors
    ///
    /// Returns an error if `degree` is less than 2.
    pub fn new(degree: usize, form: GaussChebyshevForm) -> Result<Self, GaussChebyshevError> {
        if degree < 2 {
            return Err(GaussChebyshevError);
        }

        let n = degree as f64;

        Ok(Self {
            node_weight_pairs: match form {
                GaussChebyshevForm::Divide => (1..degree + 1)
                    .map(|i| ((PI * (2.0 * (i as f64) - 1.0) / (2.0 * n)).cos(), PI / n))
                    .collect(),
                GaussChebyshevForm::Multiply => (1..degree + 1)
                    .map(|i| {
                        let over_n_plus_1 = 1.0 / (n + 1.0);
                        let sin_val = (PI * i as f64 * over_n_plus_1).sin();
                        (
                            (PI * i as f64 * over_n_plus_1).cos(),
                            PI * over_n_plus_1 * sin_val * sin_val,
                        )
                    })
                    .collect(),
            },
            form,
        })
    }

    /// Returns the form of the rule, that is, whether it is designed for integrating functions of the form
    /// f(x) * sqrt(1 - x^2) or f(x) / sqrt(1 - x^2).
    pub const fn form(&self) -> GaussChebyshevForm {
        self.form
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
    /// # use gauss_quad::chebyshev::{GaussChebyshev, GaussChebyshevError, GaussChebyshevForm};
    /// # use approx::assert_relative_eq;
    /// # use core::f64::consts::PI;
    /// let rule = GaussChebyshev::new(3, GaussChebyshevForm::Multiply)?;
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
}

__impl_node_weight_rule! {GaussChebyshev, GaussChebyshevNodes, GaussChebyshevWeights, GaussChebyshevIter, GaussChebyshevIntoIter}
