use crate::hermite;
use crate::legendre;

pub trait QuadratureRule {
    fn nodes_and_weights(deg: usize) -> (Vec<f64>, Vec<f64>);
}

pub trait DefiniteIntegral {
    fn argument_transformation(x: f64, a: f64, b: f64) -> f64;
    fn scale_factor(a: f64, b: f64) -> f64;
    fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64;
}

pub trait ImproperIntegral {
    fn integrate<F>(&self, integrand: F) -> f64
    where
        F: Fn(f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GaussHermite, GaussLegendre, PI};

    #[test]
    fn integrate_linear_legendre() {
        let quad = GaussLegendre::init(5);
        let integral = quad.integrate(0.0, 1.0, |x| x);
        assert_float_absolute_eq!(integral, 0.5);
    }
    #[test]
    fn integrate_parabola_legendre() {
        let quad = GaussLegendre::init(5);
        let integral = quad.integrate(0.0, 3.0, |x| x.powi(2));
        assert_float_absolute_eq!(integral, 9.0);
    }
    #[test]
    fn integrate_one_hermite() {
        let quad = GaussHermite::init(5);
        let integral = quad.integrate(|_x| 1.0);
        assert_float_absolute_eq!(integral, PI.sqrt());
    }
}
