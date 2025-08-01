//! Numerical integration using the trapezoid rule.
//!
//! This method approximates the integral of a function by dividing the area under the curve into trapezoids
//! and summing their areas.
//!
//! # Example
//!
//! ```
//! use gauss_quad::Trapezoid;
//! use approx::assert_abs_diff_eq;
//!
//! // initialize a trapezoid rule with 1000 grid points.
//! let rule = Trapezoid::new(1000.try_into().unwrap());
//!
//! // numerically integrate a function from -1.0 to 1.0 using the rule.
//! let integral = rule.integrate(-1.0, 1.0, |x| x * x - 1.0);
//!
//! assert_abs_diff_eq!(integral, -4.0 / 3.0, epsilon = 1e-5);
//! ```

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use core::num::NonZeroU32;

use crate::__impl_node_rule;

/// A trapezoid rule.
///
/// This rule does not allocate anything on the heap.
///
/// # Example
///
/// ```
/// use gauss_quad::Trapezoid;
/// # use approx::assert_abs_diff_eq;
///
/// // initialize a trapezoid rule with 1000 grid points.
/// let rule = Trapezoid::new(1000.try_into().unwrap());
///
/// // numerically integrate a function from -1.0 to 1.0 using the rule.
/// let integral = rule.integrate(-1.0, 1.0, |x| x * x - 1.0);
///
/// assert_abs_diff_eq!(integral, -4.0 / 3.0, epsilon = 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trapezoid {
    degree: NonZeroU32,
}

impl Trapezoid {
    /// Create a new instance of the Trapezoid rule with the given degree.
    pub const fn new(degree: NonZeroU32) -> Self {
        Self { degree }
    }

    /// Integrate a function using the trapezoidal rule.
    ///
    /// # Example
    ///
    /// ```
    /// use gauss_quad::Trapezoid;
    /// # use approx::assert_abs_diff_eq;
    ///
    /// let rule = Trapezoid::new(1000.try_into().unwrap());
    ///
    /// assert_abs_diff_eq!(rule.integrate(1.0, 2.0, |x| x * x), 7.0 / 3.0, epsilon = 1e-6);
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let degree = self.degree.get();
        let delta_x = (b - a) / f64::from(degree);
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..degree)
            .map(|x| integrand(a + f64::from(x) * delta_x))
            .sum();
        (edge_points + sum) * delta_x
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Self::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let degree = self.degree.get();
        let delta_x = (b - a) / f64::from(degree);
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..degree)
            .into_par_iter()
            .map(|x| integrand(a + f64::from(x) * delta_x))
            .sum();
        (edge_points + sum) * delta_x
    }
}

__impl_node_rule! {Trapezoid, TrapezoidIter}

impl From<NonZeroU32> for Trapezoid {
    #[inline]
    fn from(degree: NonZeroU32) -> Self {
        Self { degree }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_integration() {
        let quad = Trapezoid::new(1000.try_into().unwrap());
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.000001);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_test_integration() {
        let quad = Trapezoid::new(1000).unwrap();
        let integral = quad.par_integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.000001);
    }

    #[test]
    fn verify_from_equivalence() {
        let new = Trapezoid::new(100.try_into().unwrap());
        let from = Trapezoid::from(NonZeroU32::new(100).unwrap());
        assert_eq!(new, from);
    }
}
