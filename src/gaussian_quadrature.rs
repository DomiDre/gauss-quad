#[cfg(test)]
mod tests {
    use crate::{GaussHermite, GaussLegendre, PI};

    #[test]
    fn integrate_linear_legendre() {
        let quad = GaussLegendre::new(5).unwrap();
        let integral = quad.integrate(0.0, 1.0, |x| x);
        approx::assert_abs_diff_eq!(integral, 0.5, epsilon = 1e-15);
    }
    #[test]
    fn integrate_parabola_legendre() {
        let quad = GaussLegendre::new(5).unwrap();
        let integral = quad.integrate(0.0, 3.0, |x| x.powi(2));
        approx::assert_abs_diff_eq!(integral, 9.0, epsilon = 1e-13);
    }
    #[test]
    fn integrate_one_hermite() {
        let quad = GaussHermite::new(5).unwrap();
        let integral = quad.integrate(|_x| 1.0);
        approx::assert_abs_diff_eq!(integral, PI.sqrt(), epsilon = 1e-15);
    }
}
