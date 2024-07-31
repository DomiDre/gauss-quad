//! This algorithm is based on an expansion of Legendre polynomials in terms of Bessel functions
//! where for large degrees only the first terms in the expansion matter. This means that
//! the zeros of the Legendre polynomials (the nodes in the quadrature rules) can be approximated
//! as the zeros of an expression of Bessel functions. The weights can be computed from the nodes.
//! The zeros of the Bessel functions are computed in terms of an expression involving only the
//! zeros of the zero:th order Bessel function and the first order Bessel function at those zeros.
//! For large enough degrees these expansions are accurate to within machine precision,
//! and lookup tables with exact values are used below those degrees.
//! For an exact derivation see the paper: `<https://doi.org/10.1137/140954969>`.

#[rustfmt::skip]
mod data;

use core::{cmp::Ordering, f64::consts::PI};
use data::{CL, EVEN_THETA_ZEROS, EVEN_WEIGHTS, J1, JZ, ODD_THETA_ZEROS, ODD_WEIGHTS};

/// This function computes the `k`th zero of Bessel function j_0.
/// 
/// # Panic
/// 
/// Panics if `k == 0`.
#[rustfmt::skip]
// Inlined because the function is only used in a single location
#[inline]
#[must_use]
fn bessel_j0_zero(k: usize) -> f64 {
    if k > 20 {
        let z: f64 = PI * (k as f64 - 0.25);
        let r = 1.0 / z;
        let r2 = r * r;
        z + r * (0.125 + r2 * (-8.072_916_666_666_667e-2 + r2 * (0.246_028_645_833_333_34 + r2 * (-1.824_438_767_206_101 + r2 * (25.336_414_797_343_906 + r2 * (-567.644_412_135_183_4 + r2 * (18_690.476_528_232_066  + r2 * (-8.493_535_802_991_488e5 + 5.092_254_624_022_268e7 * r2))))))))
    } else {
        JZ[k - 1]
    }
}

/// This function computes the square of Bessel function j_1
/// evaluated at the `k`th zero of Bessel function j_0.
/// 
/// # Panic
/// 
/// Panics if `k == 0`.
#[rustfmt::skip]
// Inlined because the function is only used in a single location
#[inline]
#[must_use]
fn bessel_j1_squared(k: usize) -> f64 {
    if k > 21 {
        let x: f64 = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202_642_367_284_675_55 + x2 * x2 * (-3.033_804_297_112_902_7e-4 + x2 * (1.989_243_642_459_693e-4 + x2 * (-2.289_699_027_721_116_6e-4 + x2 * (4.337_107_191_307_463e-4 + x2 * (-1.236_323_497_271_754e-3 + x2 * (4.961_014_232_688_831_4e-3 + x2 * (-2.668_373_937_023_237_7e-2 + 0.185_395_398_206_345_62 * x2))))))))
    } else {
        J1[k - 1]
    }
}

/// A struct containing a Gauss-Legendre node and its associated weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeWeightPair {
    node: f64,
    weight: f64,
}

impl NodeWeightPair {
    /// Compute a new `NodeWeightPair`.
    pub fn new(n: usize, k: usize) -> Self {
        ThetaWeightPair::new(n, k).into()
    }

    /// Destructure `self` into a tuple containing the node and weight.  
    /// The first number is the node, and the second is the weight.
    // Inlined because the function is trivial
    #[inline]
    pub const fn into_tuple(self) -> (f64, f64) {
        (self.node, self.weight)
    }
}

impl From<ThetaWeightPair> for NodeWeightPair {
    // Inlined because the function is only used in a single location
    #[inline]
    fn from(value: ThetaWeightPair) -> Self {
        Self {
            node: value.theta.cos(),
            weight: value.weight,
        }
    }
}

/// A Gauss-Legendre node-weight pair in theta-space.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ThetaWeightPair {
    theta: f64,
    weight: f64,
}

impl ThetaWeightPair {
    /// Compute a new ThetaWeightPair.
    ///
    /// # Panic
    ///
    /// Panics if `k == 0` or `k > n`.
    #[must_use]
    fn new(n: usize, k: usize) -> Self {
        if n <= 100 {
            // If n is small enough we can just use tabulated values.
            Self::tabulated_pair(n, k - 1)
        } else if 2 * k - 1 > n {
            let mut p = Self::compute_pair(n, n - k + 1);
            p.theta = PI - p.theta;
            p
        } else {
            Self::compute_pair(n, k)
        }
    }

    /// Compute a node-weight pair, with k limited to half the range
    ///
    /// # Panic
    ///
    /// Panics if `k == 0`.
    #[rustfmt::skip]
    #[must_use]
    fn compute_pair(n: usize, k: usize) -> Self {
        // First get the j_0 zero
        let w: f64 = 1.0 / (n as f64 + 0.5);
        let nu = bessel_j0_zero(k);
        let theta = w * nu;
        let x = theta * theta;

        // Get the asymptotic j_1(nu) squared
        let b = bessel_j1_squared(k);

        // Get the Chebyshev interpolants for the nodes...
        let sf1t = (((((-1.290_529_962_742_805_1e-12 * x + 2.407_246_858_643_301_3e-10) * x - 3.131_486_546_359_920_4e-8) * x + 2.755_731_689_620_612_4e-6) * x - 1.488_095_237_139_091_4e-4) * x + 4.166_666_666_651_934e-3) * x - 4.166_666_666_666_63e-2;
        let sf2t = (((((2.206_394_217_818_71e-9 * x - 7.530_367_713_737_693e-8) * x + 1.619_692_594_538_362_7e-6) * x - 2.533_003_260_082_32e-5) * x + 2.821_168_860_575_604_5e-4) * x - 2.090_222_483_878_529e-3) * x + 8.159_722_217_729_322e-3;
        let sf3t = (((((-2.970_582_253_755_262_3e-8 * x + 5.558_453_302_237_962e-7) * x - 5.677_978_413_568_331e-6) * x + 4.184_981_003_295_046e-5) * x - 2.513_952_932_839_659e-4) * x + 1.286_541_985_428_451_3e-3) * x - 4.160_121_656_202_043e-3;

        // ...and weights
        let wsf1t = ((((((((-2.209_028_610_446_166_4e-14 * x + 2.303_657_268_603_773_8e-12) * x - 1.752_577_007_354_238e-10) * x + 1.037_560_669_279_168e-8) * x - 4.639_686_475_532_213e-7) * x + 1.496_445_936_250_286_4e-5) * x - 3.262_786_595_944_122e-4) * x + 4.365_079_365_075_981e-3) * x - 3.055_555_555_555_53e-2) * x + 8.333_333_333_333_333e-2;
        let wsf2t = (((((((3.631_174_121_526_548e-12 * x + 7.676_435_450_698_932e-11) * x - 7.129_128_572_336_422e-9) * x + 2.114_838_806_859_471_6e-7) * x - 3.818_179_186_800_454e-6) * x + 4.659_695_306_949_684e-5) * x - 4.072_971_856_113_357_5e-4) * x + 2.689_594_356_947_297e-3) * x - 1.111_111_111_112_149_2e-2;
        let wsf3t = (((((((2.018_267_912_567_033e-9 * x - 4.386_471_225_202_067e-8) * x + 5.088_983_472_886_716e-7) * x - 3.979_333_165_191_352_5e-6) * x + 2.005_593_263_964_583_4e-5) * x - 4.228_880_592_829_212e-5) * x - 1.056_460_502_540_761_4e-4) * x - 9.479_693_089_585_773e-5) * x + 6.569_664_899_264_848e-3;

        // Then refine with the expansions from the paper
        let nu_o_sin = nu / theta.sin();
        let b_nu_o_sin = b * nu_o_sin;
        let w_inv_sinc = w * w * nu_o_sin;
        let wis2 = w_inv_sinc * w_inv_sinc;

        // Compute the theta-node and the weight
        let theta = w * (nu + theta * w_inv_sinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)));
        let weight = 2.0 * w / (b_nu_o_sin + b_nu_o_sin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t)));

        Self { theta, weight }
    }

    /// Returns tabulated theta and weight values, valid for n <= 100
    ///
    /// # Panic
    ///
    /// Panics if `n > 100`.
    #[must_use]
    fn tabulated_pair(n: usize, k: usize) -> Self {
        // Odd Legendre degree
        let (theta, weight) = if n % 2 == 1 {
            let n2 = (n - 1) / 2;
            match k.cmp(&n2) {
                Ordering::Equal => (PI / 2.0, 2.0 / (CL[n] * CL[n])),
                Ordering::Less => (
                    ODD_THETA_ZEROS[n2 - 1][n2 - k - 1],
                    ODD_WEIGHTS[n2 - 1][n2 - k - 1],
                ),
                Ordering::Greater => (
                    PI - ODD_THETA_ZEROS[n2 - 1][k - n2 - 1],
                    ODD_WEIGHTS[n2 - 1][k - n2 - 1],
                ),
            }
        // Even Legendre degree
        } else {
            let n2 = n / 2;
            match k.cmp(&n2) {
                Ordering::Less => (
                    EVEN_THETA_ZEROS[n2 - 1][n2 - k - 1],
                    EVEN_WEIGHTS[n2 - 1][n2 - k - 1],
                ),
                Ordering::Equal | Ordering::Greater => (
                    PI - EVEN_THETA_ZEROS[n2 - 1][k - n2],
                    EVEN_WEIGHTS[n2 - 1][k - n2],
                ),
            }
        };
        Self { theta, weight }
    }
}

#[cfg(test)]
mod test {
    use super::{bessel_j0_zero, bessel_j1_squared};
    use approx::assert_abs_diff_eq;

    #[test]
    fn check_bessel_j0_zero() {
        // check bessel zeros with values from table in A Treatise on the Theory of Bessel Functions
        assert_abs_diff_eq!(bessel_j0_zero(1), 2.404_825_6, epsilon = 0.000_000_1);
        assert_abs_diff_eq!(bessel_j0_zero(2), 5.520_078_1, epsilon = 0.000_000_1);
        assert_abs_diff_eq!(bessel_j0_zero(20), 62.048_469_2, epsilon = 0.000_000_1);
        assert_abs_diff_eq!(bessel_j0_zero(30), 93.463_718_8, epsilon = 0.000_000_1);
        assert_abs_diff_eq!(bessel_j0_zero(40), 124.879_308_9, epsilon = 0.000_000_1);
    }

    #[test]
    fn check_bessel_j1_squared() {
        // check bessel j_1 squared values evaluated at zeros of j_0
        // reference values calculated using implementation from scipy.special.j1
        assert_abs_diff_eq!(bessel_j1_squared(1), 0.269_514_1, epsilon = 0.000_000_1);
        assert_abs_diff_eq!(bessel_j1_squared(30), 0.006_811_5, epsilon = 0.000_000_1);
    }
}
