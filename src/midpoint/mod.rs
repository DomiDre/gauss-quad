//! Numerical integration using the midpoint rule.
//!
//! This is one of the simplest integration schemes.
//!
//! 1. Divide the domain into equally sized sections.
//! 2. Find the function value at the midpoint of each section.
//! 3. The section's integral is approximated as a rectangle as wide as the section and as tall as the function
//!  value at the midpoint.
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

pub mod iterators;

use crate::Node;
use iterators::MidpointIter;

/// A midpoint rule quadrature scheme.
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
    /// # Panics
    /// Panics if degree is less than 1
    pub fn new(degree: usize) -> Result<Self, MidpointError> {
        if degree > 0 {
            Ok(Self {
                nodes: (0..degree).map(|d| d as f64).collect(),
            })
        } else {
            Err(MidpointError())
        }
    }

    /// Returns an iterator over the nodes of the midpoint rule.
    #[inline]
    pub fn iter(&self) -> MidpointIter<'_> {
        MidpointIter::new(self.nodes.iter())
    }

    /// Returns the nodes of the rule as a slice.
    #[inline]
    pub fn as_nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Converts `self` into a vector of nodes.
    ///
    /// Simply returns the underlying vector with no computation or allocation.
    #[inline]
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes
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
}

impl IntoIterator for Midpoint {
    type IntoIter = std::vec::IntoIter<Node>;
    type Item = Node;
    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MidpointError();

use core::fmt;
impl fmt::Display for MidpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree of the midpoint rule needs to be at least 1")
    }
}

impl std::error::Error for MidpointError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_midpoint_integration() {
        let quad = Midpoint::new(100).unwrap();
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        approx::assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[test]
    fn check_derives() {
        let quad = Midpoint::new(10);
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = Midpoint::new(3);
        assert_ne!(quad, other_quad);
    }
}
