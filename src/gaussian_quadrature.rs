use crate::hermite;
use crate::legendre;

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
