/*!
Numerical integration using the midpoint rule.

This is one of the simplest integration schemes.

1. Divide the domain into equally sized sections.
2. Find the function value at the midpoint of each section.
3. The section's integral is approximated as a rectangle as wide as the section and as tall as the function
 value at the midpoint.

```
use gauss_quad::{Midpoint};
use approx::assert_abs_diff_eq;

use std::f64::consts::PI;

let eps = 0.001;

let n = 30;
let quad = Midpoint::init(n);

// integrate some functions
let two_thirds = quad.integrate(-1.0, 1.0, |x| x * x);
assert_abs_diff_eq!(two_thirds, 0.66666, epsilon = eps);

let estimate_sin = quad.integrate(-PI, PI, |x| x.sin());
assert_abs_diff_eq!(estimate_sin, 0.0, epsilon = eps);

// some functions need more steps than others
let m = 100;
let better_quad = Midpoint::init(m);

let piecewise = better_quad.integrate(-5.0, 5.0,
                    |x| if x > 1.0 && x < 2.0 {
                        (-x * x).exp()
                    } else { 0.0 });

assert_abs_diff_eq!(0.135257, piecewise, epsilon = eps);
```

!*/

#[derive(Debug, Clone)]
/// A midpoint rule quadrature scheme.
/// ```
/// # extern crate gauss_quad;
/// # use gauss_quad::Midpoint;
/// # fn main() {
/// #
/// // initialize a midpoint rule with 100 cells
/// let quad: Midpoint = Midpoint::init(100);
///
/// // numerically integrate a function from -1.0 to 1.0 using the midpoint rule
/// let approx = quad.integrate(-1.0, 1.0, |x| x * x);
/// # }
/// ```
pub struct Midpoint {
    /// The dimensionless midpoints
    nodes: Vec<f64>,
}

impl Midpoint {
    /// Initialize a new midpoint rule with `degree` number of cells.
    // -- code based on Luca Palmieri's "Scientific computing: a Rust adventure [Part 2 - Array1]"
    //    https://www.lpalmieri.com/posts/2019-04-07-scientific-computing-a-rust-adventure-part-2-array1/
    pub fn init(degree: usize) -> Self {
        assert!(degree >= 1, "Degree of Midpoint rule needs to be >= 1");
        Self {
            nodes: Self::nodes(degree),
        }
    }

    /// Make a set of evenly spaced nodes
    fn nodes(degree: usize) -> Vec<f64> {
        let mut nodes = Vec::new();
        nodes.reserve(degree);
        for idx in 0..degree {
            nodes.push(idx as f64);
        }

        nodes
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_midpoint_integration() {
        let quad = Midpoint::init(100);
        let integral = quad.integrate(0.0, 1.0, |x| x * x);
        approx::assert_abs_diff_eq!(integral, 1.0 / 3.0, epsilon = 0.0001);
    }
}
