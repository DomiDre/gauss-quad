//! Numerical integration using the uniform [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
//!
//! This rule can integrate functions on finite intervals, [a, b].

use core::{iter::FusedIterator, num::NonZeroU32};

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
/// # use approx::assert_abs_diff_eq;
///
/// // initialize a trapezoid rule with 1000 grid points.
/// let rule = Trapezoid::new(1000.try_into()?);
///
/// // numerically integrate a function from -1.0 to 1.0 using the rule.
/// let integral = rule.integrate(-1.0, 1.0, |x| x * x - 1.0);
///
/// assert_abs_diff_eq!(integral, -4.0 / 3.0, epsilon = 1e-5);
/// # Ok::<(), core::num::TryFromIntError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// let rule = Trapezoid::new(1000.try_into()?);
    ///
    /// assert_abs_diff_eq!(rule.integrate(1.0, 2.0, |x| x * x), 7.0 / 3.0, epsilon = 1e-6);
    ///
    /// # Ok::<(), core::num::TryFromIntError>(())
    /// ```
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let delta_x = (b - a) / f64::from(self.degree.get());
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..self.degree.get())
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
        let delta_x = (b - a) / f64::from(self.degree.get());
        let edge_points = (integrand(a) + integrand(b)) / 2.0;
        let sum: f64 = (1..self.degree.get())
            .into_par_iter()
            .map(|x| integrand(a + f64::from(x) * delta_x))
            .sum();
        (edge_points + sum) * delta_x
    }

    /// Returns the degree of the rule.
    pub const fn degree(&self) -> NonZeroU32 {
        self.degree
    }

    /// Returns an iterator over the nodes of the rule.
    pub fn iter(&self) -> TrapezoidIter {
        TrapezoidIter::new(self.degree)
    }
}

impl IntoIterator for Trapezoid {
    type Item = Node;
    type IntoIter = TrapezoidIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        TrapezoidIter::new(self.degree())
    }
}

impl IntoIterator for &Trapezoid {
    type Item = Node;
    type IntoIter = TrapezoidIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        TrapezoidIter::new(self.degree())
    }
}

/// An iterator over the nodes of the trapezoid rule.
#[derive(Debug, Clone)]
pub struct TrapezoidIter(core::iter::Map<core::ops::RangeInclusive<u32>, fn(u32) -> f64>);

impl TrapezoidIter {
    pub(crate) fn new(degree: NonZeroU32) -> Self {
        Self((0..=degree.get()).map(f64::from))
    }
}

impl Iterator for TrapezoidIter {
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

impl DoubleEndedIterator for TrapezoidIter {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n)
    }
}

impl FusedIterator for TrapezoidIter {}

#[cfg(test)]
mod test {
    use super::*;

    use core::num::NonZeroU32;

    use approx::assert_abs_diff_eq;

    #[test]
    fn integrate_parabola() {
        let rule = Trapezoid::new(1000.try_into().unwrap());
        assert_eq!(rule.degree().get(), 1000);
        assert_abs_diff_eq!(
            rule.integrate(1.0, 2.0, |x| x * x),
            7.0 / 3.0,
            epsilon = 1e-6
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_par_integration() {
        let rule = Trapezoid::new(1000.try_into().unwrap());
        assert_abs_diff_eq!(
            rule.par_integrate(1.0, 2.0, |x| x * x),
            7.0 / 3.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_iter() {
        let rule = Trapezoid::new(1000.try_into().unwrap());
        assert_eq!(rule.iter().size_hint(), (1001, Some(1001)));
        assert_eq!(rule.iter().next(), Some(0.0));
        assert_eq!(rule.iter().nth(999), Some(999.0));
        assert_eq!(rule.iter().last(), Some(1000.0));
        assert_eq!(rule.iter().count(), 1001);
        assert_eq!(rule.iter().next_back(), Some(1000.0));
        assert_eq!(rule.iter().nth_back(999), Some(1.0));
    }

    #[test]
    fn test_into_iter() {
        const DEGREE: NonZeroU32 = NonZeroU32::new(1000).unwrap();

        let rule = Trapezoid::new(DEGREE);

        for (node, ans) in (&rule).into_iter().zip(0..=DEGREE.get()) {
            assert_eq!(node, ans as f64);
        }

        for (node, ans) in rule.into_iter().zip(0..=DEGREE.get()) {
            assert_eq!(node, ans as f64);
        }
    }

    #[test]
    fn test_single_node() {
        let rule = Trapezoid::new(1.try_into().unwrap());
        assert_eq!(rule.integrate(0.0, 1.0, |x| x), 0.5);
    }
}
