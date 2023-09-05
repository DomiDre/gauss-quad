use crate::{DMatrixf64, PI};

#[derive(Debug, Clone, PartialEq)]
pub struct GaussHermite {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussHermite {
    pub fn init(deg: usize) -> GaussHermite {
        let (nodes, weights) = GaussHermite::nodes_and_weights(deg);

        GaussHermite { nodes, weights }
    }

    /// Apply Golub-Welsch algorithm to determine Gauss-Hermite nodes & weights
    /// construct companion matrix A for the Hermite Polynomial using the relation:
    /// 1/2 H_{n+1} + n H_{n-1} = x H_n
    /// A similar matrix that is symmetrized is constructed by D A D^{-1}
    /// Resulting in a symmetric tridiagonal matrix with
    /// 0 on the diagonal & sqrt(n/2) on the off-diagonal
    /// root & weight finding are equivalent to eigenvalue problem
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn nodes_and_weights(deg: usize) -> (Vec<f64>, Vec<f64>) {
        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let element = (idx_f64 * 0.5).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = element;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = element;
            }
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        // return nodes and weights as Vec<f64>
        let nodes = eigen.eigenvalues.data.as_vec().clone();
        let weights = (eigen.eigenvectors.row(0).map(|x| x.powi(2)) * PI.sqrt())
            .data
            .as_vec()
            .clone();
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
            .map(|(&x_val, w_val)| integrand(x_val) * w_val)
            .sum();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golub_welsch_3() {
        let (x, w) = GaussHermite::nodes_and_weights(3);
        let x_should = [1.224_744_871_391_589, 0.0, -1.224_744_871_391_589];
        let w_should = [
            0.295_408_975_150_919_35,
            1.181_635_900_603_677_4,
            0.295_408_975_150_919_35,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-15);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-15);
        }
    }
}
