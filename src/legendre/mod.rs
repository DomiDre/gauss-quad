use crate::DMatrixf64;

pub struct GaussLegendre {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussLegendre {
    pub fn init(deg: usize) -> GaussLegendre {
        let (nodes, weights) = GaussLegendre::nodes_and_weights(deg);

        GaussLegendre { nodes, weights }
    }

    /// Apply Golub-Welsch algorithm to determine Gauss-Legendre nodes & weights
    /// construct companion matrix A for the Hermite Polynomial using the relation:
    /// (n+1)/(2n+1) P_{n+1} + n/(2n+1) P_{n-1} = x P_n
    /// A similar matrix that is symmetrized is constructed by C = D A D^{-1}
    /// where D is a diagonal matrix and det(C-t*1) = det(A-t*1)
    /// Resulting in a symmetric tridiagonal matrix with
    /// 0 on the diagonal & n/sqrt(4n^2 - 1) on the off-diagonal.
    /// Root & weight finding are equivalent to eigenvalue problem.
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn nodes_and_weights(deg: usize) -> (Vec<f64>, Vec<f64>) {
        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = 1.0 + idx as f64;
            let element = idx_f64 / (4.0 * idx_f64 * idx_f64 - 1.0).sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = element;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = element;
            }
        }
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        // return nodes and weights as Vec<f64>
        let nodes = eigen.eigenvalues.data.as_vec().clone();
        let weights = (eigen.eigenvectors.row(0).map(|x| x.powi(2)) * 2.0)
            .data
            .as_vec()
            .clone();
        (nodes, weights)
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    /// Perform quadrature of integrand using given nodes x and weights w
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&x_val, w_val)| {
                integrand(
                    GaussLegendre::argument_transformation(x_val, a, b,)
                ) * w_val
            })
            .sum();
        GaussLegendre::scale_factor(a, b) * result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golub_welsch_3() {
        let (x, w) = GaussLegendre::nodes_and_weights(3);
        println!("{:?}", x);

        let x_should = [0.7745966692414834, 0.0000000000000000, -0.7745966692414834];
        let w_should = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }
}
