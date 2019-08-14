use crate::DMatrixf64;
use statrs::function::gamma::gamma;

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

    /// Perform quadrature of integrand using given nodes x and weights w
    pub fn integrate<F>(&self, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(x_val, w_val)| (integrand)(x_val.clone()) * w_val)
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
        let x_should = [4.354248688935409409498, 9.645751311064590590502];
        let w_should = [82.67786838055363363287, 37.32213161944636636713];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_0() {
        let (x, w) = GaussLaguerre::nodes_and_weights(3, 0.0);
        let x_should = [
            0.4157745567834790833115,
            2.29428036027904171982,
            6.289945082937479196866,
        ];
        let w_should = [
            0.71109300992917301545,
            0.2785177335692408488014,
            0.01038925650158613574897,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_3_alpha_1_5() {
        let (x, w) = GaussLaguerre::nodes_and_weights(3, 1.5);
        let x_should = [
            1.220402317558883850334,
            3.808880721467068127299,
            8.470716960974048022367,
        ];
        let w_should = [
            0.730637894350016062262,
            0.566249100686605712588,
            0.03245339314251524562338,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_negative() {
        let (x, w) = GaussLaguerre::nodes_and_weights(5, -0.9);
        let x_should = [
            0.02077715131928810402902,
            0.8089975361346021299738,
            2.674900020624070245538,
            5.869026089963398759102,
            11.12629920195864076136,
        ];
        let w_should = [
            8.7382892412424362192,
            0.7027823530897444422573,
            0.07011172063284948044208,
            0.00231276011611556364796,
            1.16235875861307471126E-5,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }
}
