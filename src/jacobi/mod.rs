use crate::DMatrixf64;
use statrs::function::gamma::gamma;

pub struct GaussJacobi {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl GaussJacobi {
    pub fn init(deg: usize, alpha: f64, beta: f64) -> GaussJacobi {
        let (nodes, weights) = GaussJacobi::nodes_and_weights(deg, alpha, beta);

        GaussJacobi { nodes, weights }
    }

    /// Apply Golub-Welsch algorithm to determine Gauss-Jacobi nodes & weights
    /// see Gil, Segura, Temme - Numerical Methods for Special Functions
    pub fn nodes_and_weights(deg: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
        if alpha < -1.0 || beta < -1.0 {
            panic!("Gauss-Laguerre quadrature needs alpha > -1.0 and beta > -1.0");
        }
        if deg < 2 {
            panic!("Degree of Gauss-Quadrature needs to be >= 2");
        }

        let mut companion_matrix = DMatrixf64::from_element(deg, deg, 0.0);

        let mut diag = (beta - alpha) / (2.0 + beta + alpha);
        // Initialize symmetric companion matrix
        for idx in 0..deg - 1 {
            let idx_f64 = idx as f64;
            let idx_p1 = idx_f64 + 1.0;
            let denom_sum = 2.0 * idx_p1 + alpha + beta;
            let off_diag = 2.0 / denom_sum
                * (idx_p1 * (idx_p1 + alpha) * (idx_p1 + beta) * (idx_p1 + alpha + beta)
                    / ((denom_sum + 1.0) * (denom_sum - 1.0)))
                    .sqrt();
            unsafe {
                *companion_matrix.get_unchecked_mut((idx, idx)) = diag;
                *companion_matrix.get_unchecked_mut((idx, idx + 1)) = off_diag;
                *companion_matrix.get_unchecked_mut((idx + 1, idx)) = off_diag;
            }
            diag = (beta * beta - alpha * alpha) / (denom_sum * (denom_sum + 2.0));
        }
        unsafe {
            *companion_matrix.get_unchecked_mut((deg - 1, deg - 1)) = diag;
        }
        println!("{}", companion_matrix);
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        let scale_factor =
            (2.0f64).powf(alpha + beta + 1.0) * gamma(alpha + 1.0) * gamma(beta + 1.0)
                / gamma(alpha + beta + 1.0)
                / (alpha + beta + 1.0);
        // return nodes and weights as Vec<f64>
        let nodes = eigen.eigenvalues.data.as_vec().clone();
        let weights = (eigen.eigenvectors.row(0).map(|x| x.powi(2)) * scale_factor)
            .data
            .as_vec()
            .clone();
        let mut both: Vec<_> = nodes.iter().zip(weights.iter()).collect();
        both.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
        let (mut nodes, weights): (Vec<f64>, Vec<f64>) = both.iter().cloned().unzip();

        // TO FIX: implement correction
        // eigenvalue algorithm has problem to get the zero eigenvalue for odd degrees 
        // for now... manual correction seems to do the trick
        if deg & 1 == 1 {
            nodes[deg / 2] = 0.0;
        }
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
    fn golub_welsch_5_alpha_0_beta_0() {
        let (x, w) = GaussJacobi::nodes_and_weights(5, 0.0, 0.0);
        println!("{:?}, {:?}", x, w);
        let x_should = [
            -0.9061798459386639927976,
            -0.5384693101056830910363,
            0.0,
            0.5384693101056830910363,
            0.9061798459386639927976,
        ];
        let w_should = [
            0.2369268850561890875143,
            0.4786286704993664680413,
            0.5688888888888888888889,
            0.4786286704993664680413,
            0.2369268850561890875143,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_2_alpha_1_beta_0() {
        let (x, w) = GaussJacobi::nodes_and_weights(2, 1.0, 0.0);
        println!("{:?}, {:?}", x, w);
        let x_should = [-0.6898979485566356196395, 0.2898979485566356196395];
        let w_should = [1.272165526975908677578, 0.7278344730240913224225];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_1_beta_0() {
        let (x, w) = GaussJacobi::nodes_and_weights(5, 1.0, 0.0);
        println!("{:?}", x);
        println!("{:?}", w);
        let x_should = [
            -0.9203802858970625153184,
            -0.6039731642527836549284,
            0.0,
            0.3909285467072721890292,
            0.802929828402347147753,
        ];
        let w_should = [
            0.3871263609066067170974,
            0.6686985523774782619667,
            0.5855479483386792347922,
            0.2956354802904666814025,
            0.06299165808676910474117,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_0_beta_1() {
        let (x, w) = GaussJacobi::nodes_and_weights(5, 0.0, 1.0);
        println!("{:?}, {:?}", x, w);
        let x_should = [
            -0.802929828402347147753,
            -0.3909285467072721890292,
            0.0,
            0.6039731642527836549284,
            0.9203802858970625153184,
        ];
        let w_should = [
            0.06299165808676910474117,
            0.2956354802904666814025,
            0.5855479483386792347922,
            0.6686985523774782619667,
            0.3871263609066067170974,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }

    #[test]
    fn golub_welsch_50_alpha_42_beta_23() {
        let (x, w) = GaussJacobi::nodes_and_weights(50, 42.0, 23.0);
        let x_should = [
            -0.9365282331525411838663,
            -0.914340864546088583509,
            -0.8921599049727096724447,
            -0.8692169092212256532025,
            -0.8452772287692255936742,
            -0.8202527663480567638827,
            -0.7941135404985295060866,
            -0.766857786572463477347,
            -0.738499459607423389405,
            -0.7090622355144468371192,
            -0.6785763279056292625059,
            -0.6470766611816353500936,
            -0.6146017510276355831637,
            -0.5811929774585084004922,
            -0.546894086695451968942,
            -0.5117508318261052529877,
            -0.4758107003474938611877,
            -0.4391226974604178665677,
            -0.4017371657777085159884,
            -0.3637056290465180460026,
            -0.3250806516861351027124,
            -0.2859157085442329125656,
            -0.2462650609067338566445,
            -0.2061836358194088532463,
            -0.1657269064017096241122,
            -0.1249507711761477874171,
            -0.0839114305668714202679,
            -0.042665258670068654318,
            -0.00126866817019554953778,
            0.0402220341515399794636,
            0.081750804545872012533,
            0.1232620363011974665587,
            0.164700756351269239516,
            0.206012852393607163488,
            0.247145341670134957966,
            0.288046697452241011081,
            0.3286672567960525119536,
            0.3689597449831742046189,
            0.4088799712411144097709,
            0.4483877823727348289103,
            0.4874484164193912633858,
            0.5260344987981807926701,
            0.5641291140461261515919,
            0.6017307713882077559259,
            0.63886191986089739786,
            0.6755846687520414251946,
            0.7120327664554348876942,
            0.7484861314364706848162,
            0.7855851847775175445611,
            0.8252413421023552639732,
        ];
        let w_should = [
            7.48575322545471E-18,
            4.368160045795394E-15,
            5.4750922260937406E-13,
            2.88380289400016428E-11,
            8.37597440094303407E-10,
            1.551169281097026691E-8,
            2.002752126655059966E-7,
            1.914052885645138047E-6,
            1.41297397768079788844E-5,
            8.3152815809485827148E-5,
            3.99634976967242873E-4,
            0.00159844229039337849757,
            0.00540148446249289198905,
            0.01560951595196132493331,
            0.0389608598947761436648,
            0.08467599281535783864636,
            0.1613202720417803687556,
            0.270895707022141992679,
            0.4027660521441900194,
            0.532134840644357203309,
            0.6265618503964772341889,
            0.658939504140677532012,
            0.619968794555102026424,
            0.5223926348726763884,
            0.39441880692372075308,
            0.266845588852137266064,
            0.1616939432973513952064,
            0.087665230931323018033,
            0.0424621462429458192666,
            0.0183366105888594795662,
            0.0070408225241987007192,
            0.00239595351575043630831,
            7.19670969124877088955E-4,
            1.89882258226640087571E-4,
            4.3753525829371834142E-5,
            8.7442188734473804877E-6,
            1.50325570891327034203E-6,
            2.20126341718083405992E-7,
            2.7132693744791163385E-8,
            2.7749216815329958754E-9,
            2.31354608559198414896E-10,
            1.53822055920499431196E-11,
            7.9310125450026199762E-13,
            3.05766621818573885364E-14,
            8.3930769860264491044E-16,
            1.53118007263038902526E-17,
            1.675381720821777583E-19,
            9.3009618579336619391E-22,
            1.91253819440849937704E-24,
            6.6457767585162111226E-28,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_float_absolute_eq!(x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_float_absolute_eq!(w_val, w[i]);
        }
    }
}
