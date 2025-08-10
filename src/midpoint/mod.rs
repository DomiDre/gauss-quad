//! Numerical integration using the midpoint rule.
//!
//! This is one of the simplest integration schemes.
//!
//! 1. Divide the domain into equally sized sections.
//! 2. Find the function value at the midpoint of each section.
//! 3. The section's integral is approximated as a rectangle as wide as the section and as tall as the function
//!    value at the midpoint.
//!
//! ```
//! use gauss_quad::midpoint::Midpoint;
//! use approx::assert_abs_diff_eq;
//!
//! use core::f64::consts::PI;
//!
//! let eps = 0.001;
//!
//! let quad = Midpoint::new(30.try_into().unwrap());
//!
//! // integrate some functions
//! let two_thirds = quad.integrate(-1.0, 1.0, |x| x * x);
//! assert_abs_diff_eq!(two_thirds, 0.66666, epsilon = eps);
//!
//! let estimate_sin = quad.integrate(-PI, PI, |x| x.sin());
//! assert_abs_diff_eq!(estimate_sin, 0.0, epsilon = eps);
//!
//! // some functions need more steps than others
//! let better_quad = Midpoint::new(100.try_into().unwrap());
//!
//! let piecewise = better_quad.integrate(-5.0, 5.0, |x|
//!     if x > 1.0 && x < 2.0 {
//!         (-x * x).exp()
//!     } else {
//!         0.0
//!     }
//! );
//!
//! assert_abs_diff_eq!(0.135257, piecewise, epsilon = eps);
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::__impl_node_rule;

use core::num::NonZeroU32;

/// A midpoint rule.
/// ```
/// # use gauss_quad::midpoint::Midpoint;
/// // initialize a midpoint rule with 100 cells
/// let quad: Midpoint = Midpoint::new(100.try_into().unwrap());
///
/// // numerically integrate a function from -1.0 to 1.0 using the midpoint rule
/// let approx = quad.integrate(-1.0, 1.0, |x| x * x);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Midpoint {
    degree: NonZeroU32,
}

impl Midpoint {
    /// Initialize a new midpoint rule with `degree` number of cells. The nodes are evenly spaced.
    // -- code based on Luca Palmieri's "Scientific computing: a Rust adventure [Part 2 - Array1]"
    //    <https://www.lpalmieri.com/posts/2019-04-07-scientific-computing-a-rust-adventure-part-2-array1/>
    pub const fn new(degree: NonZeroU32) -> Self {
        Self { degree }
    }

    /// Integrate over the domain [a, b].
    pub fn integrate<F>(&self, a: f64, b: f64, mut integrand: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        let rect_width = (b - a) / f64::from(self.degree().get());

        let sum: f64 = self
            .iter()
            .map(|node| integrand(a + rect_width * (0.5 + node)))
            .sum();

        sum * rect_width
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Midpoint::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let rect_width = (b - a) / f64::from(self.degree().get());

        let sum: f64 = (0..self.degree().get())
            .into_par_iter()
            .map(|node| integrand(a + rect_width * (0.5 + f64::from(node))))
            .sum();

        sum * rect_width
    }

    /// Sets the degree of the midpoint rule to a new value.
    pub fn change_degree(&mut self, new_degree: NonZeroU32) {
        self.degree = new_degree;
    }
}

__impl_node_rule! {Midpoint, MidpointIter}

impl From<NonZeroU32> for Midpoint {
    #[inline]
    fn from(degree: NonZeroU32) -> Self {
        Self { degree }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn check_midpoint_integration() {
        let quad = Midpoint::new(100.try_into().unwrap());
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_check_midpoint_integration() {
        let quad = Midpoint::new(100.try_into().unwrap());
        let integral = quad.par_integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[test]
    fn check_derives() {
        let quad = Midpoint::new(10.try_into().unwrap());
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = Midpoint::new(3.try_into().unwrap());
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_iterators() {
        let rule = Midpoint::new(100.try_into().unwrap());
        let a = 0.0;
        let b = 1.0;
        let ans = 1.0 / 3.0;
        let rect_width = (b - a) / rule.degree().get() as f64;

        assert_abs_diff_eq!(
            ans,
            rule.iter().fold(0.0, |tot, n| {
                let x = a + rect_width * (0.5 + n);
                tot + x * x
            }) * rect_width,
            epsilon = 1e-4
        );

        assert_abs_diff_eq!(
            ans,
            rule.into_iter().fold(0.0, |tot, n| {
                let x = a + rect_width * (0.5 + n);
                tot + x * x
            }) * rect_width,
            epsilon = 1e-4
        );
    }

    #[test]
    fn verify_from_equivalence() {
        let new = Midpoint::new(100.try_into().unwrap());
        let from = Midpoint::from(NonZeroU32::new(100).unwrap());
        assert_eq!(new, from);
    }

    #[test]
    fn test_change_degree() {
        let mut quad = Midpoint::new(1000.try_into().unwrap());
        quad.change_degree(2000.try_into().unwrap());
        assert_eq!(quad.degree(), 2000.try_into().unwrap());
    }
}
