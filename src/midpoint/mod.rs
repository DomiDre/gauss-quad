//! Numerical integration using the [midpoint rule](https://en.wikipedia.org/wiki/Riemann_sum#Midpoint_rule).
//!
//! This is one of the simplest integration schemes.
//!
//! 1. Divide the domain into equally sized sections.
//! 2. Find the function value at the midpoint of each section.
//! 3. The section's integral is approximated as a rectangle as wide as the section and as tall as the function
//!    value at the midpoint.
//!
//! ```
//! use gauss_quad::midpoint::{Midpoint, MidpointError};
//! use approx::assert_abs_diff_eq;
//!
//! use core::f64::consts::PI;
//!
//! let eps = 0.001;
//!
//! let n = 30;
//! let quad = Midpoint::new(n)?;
//!
//! // integrate some functions
//! let two_thirds = quad.integrate(-1.0, 1.0, |x| x * x);
//! assert_abs_diff_eq!(two_thirds, 0.66666, epsilon = eps);
//!
//! let estimate_sin = quad.integrate(-PI, PI, |x| x.sin());
//! assert_abs_diff_eq!(estimate_sin, 0.0, epsilon = eps);
//!
//! // some functions need more steps than others
//! let m = 100;
//! let better_quad = Midpoint::new(m)?;
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
//! # Ok::<(), MidpointError>(())
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{Node, __impl_node_rule};

use std::backtrace::Backtrace;

/// A midpoint rule.
/// ```
/// # use gauss_quad::midpoint::{Midpoint, MidpointError};
/// // initialize a midpoint rule with 100 cells
/// let quad: Midpoint = Midpoint::new(100)?;
///
/// // numerically integrate a function from -1.0 to 1.0 using the midpoint rule
/// let approx = quad.integrate(-1.0, 1.0, |x| x * x);
/// # Ok::<(), MidpointError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Midpoint {
    /// The dimensionless midpoints
    nodes: Vec<Node>,
}

impl Midpoint {
    /// Initialize a new midpoint rule with `degree` number of cells. The nodes are evenly spaced.
    // -- code based on Luca Palmieri's "Scientific computing: a Rust adventure [Part 2 - Array1]"
    //    <https://www.lpalmieri.com/posts/2019-04-07-scientific-computing-a-rust-adventure-part-2-array1/>
    ///
    /// # Errors
    ///
    /// Returns an error if `degree` is less than 1.
    pub fn new(degree: usize) -> Result<Self, MidpointError> {
        if degree > 0 {
            Ok(Self {
                nodes: (0..degree).map(|d| d as f64).collect(),
            })
        } else {
            Err(MidpointError::new())
        }
    }

    /// Integrate over the domain [a, b].
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let rect_width = (b - a) / self.nodes.len() as f64;

        let sum: f64 = self
            .nodes
            .iter()
            .map(|&node| integrand(a + rect_width * (0.5 + node)))
            .sum();

        sum * rect_width
    }

    #[cfg(feature = "rayon")]
    /// Same as [`integrate`](Midpoint::integrate) but runs in parallel.
    pub fn par_integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64 + Sync,
    {
        let rect_width = (b - a) / self.nodes.len() as f64;

        let sum: f64 = self
            .nodes
            .par_iter()
            .map(|&node| integrand(a + rect_width * (0.5 + node)))
            .sum();

        sum * rect_width
    }
}

__impl_node_rule! {Midpoint, MidpointIter, MidpointIntoIter}

/// The error returned by [`Midpoint::new`] if given a degree of 0.
#[derive(Debug)]
pub struct MidpointError(Backtrace);

use core::fmt;
impl fmt::Display for MidpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree of the midpoint rule needs to be at least 1")
    }
}

impl MidpointError {
    /// Calls [`Backtrace::capture`] and wraps the result in a `MidpointError` struct.
    fn new() -> Self {
        Self(Backtrace::capture())
    }

    /// Returns a [`Backtrace`] to where the error was created.
    ///
    /// This backtrace is captured with [`Backtrace::capture`], see it for more information about how to make it display information when printed.
    #[inline]
    pub fn backtrace(&self) -> &Backtrace {
        &self.0
    }
}

impl std::error::Error for MidpointError {}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn check_midpoint_integration() {
        let quad = Midpoint::new(100).unwrap();
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn par_check_midpoint_integration() {
        let quad = Midpoint::new(100).unwrap();
        let integral = quad.par_integrate(0.0, 1.0, |x| x * x);
        assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[test]
    fn check_midpoint_error() {
        let midpoint_rule = Midpoint::new(0);
        assert!(midpoint_rule.is_err());
        assert_eq!(
            format!("{}", midpoint_rule.err().unwrap()),
            "the degree of the midpoint rule needs to be at least 1"
        );
    }

    #[test]
    fn check_derives() {
        let quad = Midpoint::new(10).unwrap();
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = Midpoint::new(3).unwrap();
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_iterators() {
        let rule = Midpoint::new(100).unwrap();
        let a = 0.0;
        let b = 1.0;
        let ans = 1.0 / 3.0;
        let rect_width = (b - a) / rule.degree() as f64;

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
}
