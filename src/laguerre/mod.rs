//! Numerical integration using the generalized Gauss-Laguerre quadrature rule.
//!
//! A Gauss-Laguerre rule of degree `n` has nodes and weights chosen such that it
//! can integrate polynomials of degree `2n - 1` exactly
//! with the weighing function `w(x, alpha) = x^alpha * e^(-x)` over the domain `[0, ∞)`.
//!
//! # Examples
//! ```
//! use gauss_quad::GaussLaguerre;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLaguerre::init(10, 1.0);
//! let integral = quad.integrate(|x| x.powi(2));
//! assert_abs_diff_eq!(integral, 6.0, epsilon = 1e-14);
//! ```

use crate::gamma::gamma;
use crate::DMatrixf64;

/// A Gauss-Laguerre quadrature scheme.
///
/// These rules can perform integrals with integrands of the form x^alpha * e^(-x) * f(x) over the domain [0, ∞).
/// # Example
/// Compute the factorial of 5:
/// ```
/// # use gauss_quad::GaussLaguerre;
/// # use approx::assert_abs_diff_eq;
/// let quad = GaussLaguerre::init(10, 0.0);
/// let fact_5 = quad.integrate(|x| x.powi(5));
/// assert_abs_diff_eq!(fact_5, 1.0 * 2.0 * 3.0 * 4.0 * 5.0, epsilon = 1e-11);
/// ```
pub struct GaussLaguerre {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussLaguerre {
    pub fn init(deg: usize, alpha: f64) -> GaussLaguerre {
        let (nodes, weights) = GaussLaguerre::nodes_and_weights(deg, alpha);

        GaussLaguerre { nodes, weights }
    }

    /// Apply Golub-Welsch algorithm to determine Gauss-Laguerre nodes & weights
    /// construct companion matrix A for the Laguerre Polynomial using the relation:
    /// -n L_{n-1} + (2n+1) L_{n} -(n+1) L_{n+1} = x L_n
    /// The constructed matrix is symmetric and tridiagonal with
    /// (2n+1) on the diagonal & -(n+1) on the off-diagonal (n = row number).
    /// Root & weight finding are equivalent to eigenvalue problem.
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn nodes_and_weights(deg: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
        if alpha < -1.0 {
            panic!("Gauss-Laguerre quadrature needs alpha > -1.0");
        }
        if deg < 2 {
            panic!("Degree of Gauss-Quadrature needs to be >= 2");
        }

        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);

        let mut diag = alpha + 1.0;
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let off_diag = (idx_f64 * (idx_f64 + alpha)).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx)) = diag;
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = off_diag;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = off_diag;
            }
            diag += 2.0;
        }
        unsafe {
            *companion_matrix.get_unchecked_mut((deg - 1, deg - 1)) = diag;
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        let scale_factor = gamma(alpha + 1.0);
        // return nodes and weights as Vec<f64>
        let nodes = eigen.eigenvalues.data.as_vec().clone();
        let weights = (eigen.eigenvectors.row(0).map(|x| x.powi(2)) * scale_factor)
            .data
            .as_vec()
            .clone();
        let mut both: Vec<_> = nodes.iter().zip(weights.iter()).collect();
        both.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
        let (nodes, weights): (Vec<f64>, Vec<f64>) = both.iter().cloned().unzip();
        (nodes, weights)
    }

    /// Perform quadrature of  
    /// x^`alpha` * e^(-x) * `integrand`  
    /// over the domain `[0, ∞)`, where `alpha` was given in the call to [`init`](Self::init).
    pub fn integrate<F>(&self, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&x_val, w_val)| integrand(x_val) * w_val)
            .sum();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golub_welsch_2_alpha_5() {
        let (x, w) = GaussLaguerre::nodes_and_weights(2, 5.0);
        let x_should = [4.354_248_688_935_409, 9.645_751_311_064_59];
        let w_should = [82.677_868_380_553_63, 37.322_131_619_446_37];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-12);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_0() {
        let (x, w) = GaussLaguerre::nodes_and_weights(3, 0.0);
        let x_should = [
            0.415_774_556_783_479_1,
            2.294_280_360_279_042,
            6.289_945_082_937_479_4,
        ];
        let w_should = [
            0.711_093_009_929_173,
            0.278_517_733_569_240_87,
            0.010_389_256_501_586_135,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_1_5() {
        let (x, w) = GaussLaguerre::nodes_and_weights(3, 1.5);
        let x_should = [
            1.220_402_317_558_883_8,
            3.808_880_721_467_068,
            8.470_716_960_974_048,
        ];
        let w_should = [
            0.730_637_894_350_016,
            0.566_249_100_686_605_7,
            0.032_453_393_142_515_25,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_negative() {
        let (x, w) = GaussLaguerre::nodes_and_weights(5, -0.9);
        let x_should = [
            0.020_777_151_319_288_104,
            0.808_997_536_134_602_1,
            2.674_900_020_624_07,
            5.869_026_089_963_398,
            11.126_299_201_958_641,
        ];
        let w_should = [
            8.738_289_241_242_436,
            0.702_782_353_089_744_5,
            0.070_111_720_632_849_48,
            0.002_312_760_116_115_564,
            1.162_358_758_613_074_8E-5,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }
}
