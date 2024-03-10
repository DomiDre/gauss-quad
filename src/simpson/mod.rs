//! Numerical integration using the Simpson rule.
//!
//! A popular quadrature rule (also known as Kepler's barrel rule). It can be derived
//! in the simplest case by replacing the integrand with a parabola that has the same
//! function values at the end points a & b, as well as the Simpson m=(a+b)/2, which
//! results in the integral formula
//! S(f) = (b-a)/6 * [ f(a) + 4f(m) + f(b) ]
//!
//! Dividing the interval \[a, b\] into N neighboring intervals of length h = (b-a)/N and
//! applying the Simpson rule to each subinterval, the integral is given by
//!
//! S(f) = h/6 * [ f(a) + f(b) + 2*Sum_{k=1..N-1} f(x_k) + 4*Sum_{k=1..N} f( (x_{k-1} + x_k)/2 )]
//!
//! with x_k = a + k*h.
//!
//! ```
//! use gauss_quad::simpson::Simpson;
//! # use gauss_quad::simpson::SimpsonError;
//! use approx::assert_abs_diff_eq;
//!
//! use core::f64::consts::PI;
//!
//! let eps = 0.001;
//!
//! let n = 10;
//! let quad = Simpson::new(n)?;
//!
//! // integrate some functions
//! let integrate_euler = quad.integrate(0.0, 1.0, |x| x.exp());
//! assert_abs_diff_eq!(integrate_euler, 1.0_f64.exp() - 1.0, epsilon = eps);
//!
//! let integrate_sin = quad.integrate(-PI, PI, |x| x.sin());
//! assert_abs_diff_eq!(integrate_sin, 0.0, epsilon = eps);
//! # Ok::<(), SimpsonError>(())
//! ```

pub mod iterators;

use crate::{impl_node_rule, Node};
use iterators::{SimpsonIntoIter, SimpsonIter};

/// A Simpson rule quadrature scheme.
/// ```
/// # use gauss_quad::simpson::{Simpson, SimpsonError};
/// // initialize a Simpson rule with 100 subintervals
/// let quad: Simpson = Simpson::new(100)?;
///
/// // numerically integrate a function from -1.0 to 1.0 using the Simpson rule
/// let approx = quad.integrate(-1.0, 1.0, |x| x * x);
/// # Ok::<(), SimpsonError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simpson {
    /// The dimensionless Simpsons nodes.
    nodes: Vec<Node>,
}

impl Simpson {
    /// Initialize a new Simpson rule with `degree` being the number of intervals.
    ///
    /// # Errors
    ///
    /// Returns an error if given a degree of zero.
    pub fn new(degree: usize) -> Result<Self, SimpsonError> {
        if degree >= 1 {
            Ok(Self {
                nodes: (0..degree).map(|d| d as f64).collect(),
            })
        } else {
            Err(SimpsonError)
        }
    }

    /// Integrate over the domain [a, b].
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let n = self.nodes.len() as f64;

        let h = (b - a) / n;

        // first sum over the interval edges. Skips first index to sum 1..n-1
        let sum_over_interval_edges: f64 = self
            .nodes
            .iter()
            .skip(1)
            .map(|&node| integrand(a + node * h))
            .sum();

        // sum over the midpoints f( (x_{k-1} + x_k)/2 ), as node N is not included,
        // add it in the final result
        let sum_over_midpoints: f64 = self
            .nodes
            .iter()
            .skip(1)
            .map(|&node| integrand(a + (2.0 * node - 1.0) * h / 2.0))
            .sum();

        h / 6.0
            * (2.0 * sum_over_interval_edges
                + 4.0 * sum_over_midpoints
                + 4.0 * integrand(a + (2.0 * n - 1.0) * h / 2.0)
                + integrand(a)
                + integrand(b))
    }
}

impl_node_rule! {Simpson, SimpsonIter, SimpsonIntoIter}

/// The error returned by [`Simpson::new`] if given a degree of 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimpsonError;

use core::fmt;
impl fmt::Display for SimpsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the degree of the Simpson rule must be at least 1.")
    }
}

impl std::error::Error for SimpsonError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_simpson_integration() {
        let quad = Simpson::new(2).unwrap();
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        approx::assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }

    #[test]
    fn check_simpson_error() {
        assert!(Simpson::new(0).is_err());
    }

    #[test]
    fn check_derives() {
        let quad = Simpson::new(10);
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = Simpson::new(3);
        assert_ne!(quad, other_quad);
    }
}
