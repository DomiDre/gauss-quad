//! Numerical integration using the uniform [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
//!
//! This rule can integrate functions on finite intervals, [a, b].

use std::{backtrace::Backtrace, iter::FusedIterator};

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::Node;

/// A trapezoid rule.
///
/// This rule does not allocate anything on the heap.
///
/// # Example
///
/// ```
/// use gauss_quad::Trapezoid;
/// # use gauss_quad::trapezoid::TrapezoidError;
/// # use approx::assert_abs_diff_eq;
///
/// // initialize a trapezoid rule with 1000 grid points.
/// let rule = Trapezoid::new(1000)?;
///
/// // numerically integrate a function from -1.0 to 1.0 using the rule.
/// let integral = rule.integrate(-1.0, 1.0, |x| x * x - 1.0);
///
/// assert_abs_diff_eq!(integral, -4.0 / 3.0, epsilon = 1e-5);
///
/// # Ok::<(), TrapezoidError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trapezoid {
    degree: usize,
}

impl Trapezoid {
    /// Create a new instance of the Trapezoid rule with the given degree.
    ///
    /// # Errors
    ///
    /// Returns an error if the degree is less than 2.
    pub fn new(degree: usize) -> Result<Self, TrapezoidError> {
        if degree < 2 {
            return Err(TrapezoidError::new());
        }

        Ok(Self { degree })
    }

    /// Integrate a function using the trapezoidal rule.
    ///
    /// # Example
    ///
    /// ```
    /// use gauss_quad::Trapezoid;
    /// # use gauss_quad::trapezoid::TrapezoidError;
    /// # use approx::assert_abs_diff_eq;
    ///
    /// let rule = Trapezoid::new(1000)?;
    ///
    /// assert_abs_diff_eq!(rule.integrate(1.0, 2.0, |x| x * x), 7.0 / 3.0, epsilon = 1e-6);
    /// # Ok::<(), TrapezoidError>(())
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let delta_x = (b - a) / self.degree as f64;
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..self.degree)
            .map(|x| integrand(a + x as f64 * delta_x))
            .sum();
        (edge_points + sum) * delta_x
    }

    #[cfg(feature = "rayon")]
    /// Same as [integrate](Self::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let delta_x = (b - a) / self.degree as f64;
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..self.degree)
            .into_par_iter()
            .map(|x| integrand(a + x as f64 * delta_x))
            .sum();
        (edge_points + sum) * delta_x
    }

    /// Returns the degree of the rule.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Changes the degree of the rule to the given value if possible.
    ///
    /// # Errors
    ///
    /// Returns an error if the given degree is less than 2.
    pub fn change_degree(&mut self, new_degree: usize) -> Result<(), TrapezoidError> {
        match Self::new(new_degree) {
            Ok(rule) => {
                *self = rule;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Returns an iterator over the nodes of the rule.
    pub fn nodes(&self) -> TrapezoidNodes {
        TrapezoidNodes::new(self.degree)
    }
}

#[derive(Debug, Clone)]
pub struct TrapezoidNodes(core::iter::Map<core::ops::RangeInclusive<usize>, fn(usize) -> f64>);

impl TrapezoidNodes {
    pub(crate) fn new(degree: usize) -> Self {
        Self((0..=degree).map(|x| x as f64))
    }
}

impl Iterator for TrapezoidNodes {
    type Item = Node;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
}

impl DoubleEndedIterator for TrapezoidNodes {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl FusedIterator for TrapezoidNodes {}

/// The error returned when the given degree of the [`Trapezoid`] rule is 0 or 1.
#[derive(Debug)]
pub struct TrapezoidError {
    backtrace: Backtrace,
}

impl TrapezoidError {
    pub(crate) fn new() -> Self {
        Self {
            backtrace: Backtrace::capture(),
        }
    }

    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl std::fmt::Display for TrapezoidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the degree of the Trapezoid rule must be greater than 1."
        )
    }
}

impl std::error::Error for TrapezoidError {}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_abs_diff_eq;

    #[test]
    fn test_error_in_new() {
        assert!(Trapezoid::new(0).is_err());
        assert!(Trapezoid::new(1).is_err());
        assert!(Trapezoid::new(2).is_ok());
    }

    #[test]
    fn integrate_parabola() {
        let rule = Trapezoid::new(1000).unwrap();
        assert_eq!(rule.degree(), 1000);
        assert_abs_diff_eq!(
            rule.integrate(1.0, 2.0, |x| x * x),
            7.0 / 3.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_change_degree() {
        let mut rule = Trapezoid::new(1000).unwrap();
        assert!(rule.change_degree(0).is_err());
        assert!(rule.change_degree(1).is_err());
        assert!(rule.change_degree(2).is_ok());

        assert_abs_diff_eq!(
            rule.integrate(1.0, 2.0, |x| x * x),
            7.0 / 3.0,
            epsilon = 1e-1
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_integration() {
        let rule = Trapezoid::new(1000).unwrap();
        assert_abs_diff_eq!(
            rule.par_integrate(1.0, 2.0, |x| x * x),
            7.0 / 3.0,
            epsilon = 1e-6
        );
    }
}
