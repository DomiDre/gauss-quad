//! Numerical integration using the Gauss-Jacobi quadrature rule.
//!
//! This rule can integrate expressions of the form (1 - x)^alpha * (1 + x)^beta * f(x),
//! where f(x) is a smooth function on a finite domain, alpha > -1 and beta > -1, and where f(x) is transformed from the domain [a, b] to the domain [-1, 1].
//! This enables the approximation of integrals with singularities at the end points of the domain.
//!
//! # Example
//! ```
//! use gauss_quad::jacobi::GaussJacobi;
//! # use gauss_quad::jacobi::GaussJacobiError;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussJacobi::new(10, 0.0, -1.0 / 3.0)?;
//!
//! // numerically integrate sin(x) / (1 + x)^(1/3), a function with a singularity at x = -1.
//! let integral = quad.integrate(-1.0, 1.0, |x| x.sin());
//!
//! assert_abs_diff_eq!(integral, -0.4207987746500829, epsilon = 1e-14);
//! # Ok::<(), GaussJacobiError>(())
//! ```

use crate::gamma::gamma;
use crate::{DMatrixf64, Node, Weight, __impl_node_weight_rule};

use std::backtrace::Backtrace;

/// A Gauss-Jacobi quadrature scheme.
///
/// This rule can integrate expressions of the form (1 - x)^alpha * (1 + x)^beta * f(x),
/// where f(x) is a smooth function on a finite domain, alpha > -1 and beta > -1,
/// and where f(x) is transformed from the domain [a, b] to the domain [-1, 1].
/// This enables the approximation of integrals with singularities at the end points of the domain.
///
/// # Examples
/// ```
/// # use gauss_quad::jacobi::{GaussJacobi, GaussJacobiError};
/// # use approx::assert_abs_diff_eq;
/// # use core::f64::consts::E;
/// // initialize the quadrature rule.
/// let quad = GaussJacobi::new(10, -0.5, 0.0)?;
///
/// let integral = quad.integrate(0.0, 2.0, |x| (-x).exp());
///
/// assert_abs_diff_eq!(integral, 0.9050798148074449, epsilon = 1e-14);
/// # Ok::<(), GaussJacobiError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussJacobi {
    node_weight_pairs: Vec<(Node, Weight)>,
    alpha: f64,
    beta: f64,
}

impl GaussJacobi {
    /// Initializes Gauss-Jacobi quadrature rule of the given degree by computing the nodes and weights
    /// needed for the given parameters. `alpha` is the exponent of the (1 - x) factor and `beta` is the
    /// exponent of the (1 + x) factor.
    ///
    /// Applies the Golub-Welsch algorithm to determine Gauss-Jacobi nodes & weights.
    /// See Gil, Segura, Temme - Numerical Methods for Special Functions
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2, and/or if `alpha` and/or `beta` are smaller than or equal to -1.
    pub fn new(deg: usize, alpha: f64, beta: f64) -> Result<Self, GaussJacobiError> {
        match (
            deg >= 2,
            (alpha.is_finite() && alpha > -1.0),
            (beta.is_finite() && beta > -1.0),
        ) {
            (true, true, true) => (),
            (false, true, true) => {
                return Err(GaussJacobiError::new(GaussJacobiErrorReason::Degree))
            }
            (true, false, true) => {
                return Err(GaussJacobiError::new(GaussJacobiErrorReason::Alpha))
            }
            (true, true, false) => return Err(GaussJacobiError::new(GaussJacobiErrorReason::Beta)),
            (true, false, false) => {
                return Err(GaussJacobiError::new(GaussJacobiErrorReason::AlphaBeta))
            }
            (false, false, true) => {
                return Err(GaussJacobiError::new(GaussJacobiErrorReason::DegreeAlpha))
            }
            (false, true, false) => {
                return Err(GaussJacobiError::new(GaussJacobiErrorReason::DegreeBeta))
            }
            (false, false, false) => {
                return Err(GaussJacobiError::new(
                    GaussJacobiErrorReason::DegreeAlphaBeta,
                ))
            }
        };

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
        // calculate eigenvalues & vectors
        let eigen = companion_matrix.symmetric_eigen();

        let scale_factor =
            (2.0f64).powf(alpha + beta + 1.0) * gamma(alpha + 1.0) * gamma(beta + 1.0)
                / gamma(alpha + beta + 1.0)
                / (alpha + beta + 1.0);

        // zip together the iterator over nodes with the one over weights and return as Vec<(f64, f64)>
        let mut node_weight_pairs: Vec<(f64, f64)> = eigen
            .eigenvalues
            .iter()
            .copied()
            .zip(
                eigen
                    .eigenvectors
                    .row(0)
                    .iter()
                    .map(|x| x * x * scale_factor),
            )
            .collect();

        node_weight_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // TO FIX: implement correction
        // eigenvalue algorithm has problem to get the zero eigenvalue for odd degrees
        // for now... manual correction seems to do the trick
        if deg % 2 == 1 {
            node_weight_pairs[deg / 2].0 = 0.0;
        }

        Ok(Self {
            node_weight_pairs,
            alpha,
            beta,
        })
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    /// Perform quadrature of integrand from `a` to `b`. This will integrate  
    /// (1 - x)^`alpha` * (1 + x)^`beta` * `integrand`(x)  
    /// where `alpha` and `beta` were given in the call to [`new`](Self::new), and the integrand is transformed from the domain [a, b] to the domain [-1, 1].
    pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let result: f64 = self
            .node_weight_pairs
            .iter()
            .map(|(x_val, w_val)| integrand(Self::argument_transformation(*x_val, a, b)) * w_val)
            .sum();
        Self::scale_factor(a, b) * result
    }

    /// Returns the value of the `alpha` parameter.
    #[inline]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the value of the `beta` parameter.
    #[inline]
    pub const fn beta(&self) -> f64 {
        self.beta
    }
}

__impl_node_weight_rule! {GaussJacobi, GaussJacobiNodes, GaussJacobiWeights, GaussJacobiIter, GaussJacobiIntoIter}

/// The error returned by [`GaussJacobi::new`] if given a degree, `deg`, less than 2
/// and/or an `alpha` and/or `beta` less than or equal to -1.
#[derive(Debug)]
pub struct GaussJacobiError {
    reason: GaussJacobiErrorReason,
    backtrace: Backtrace,
}

impl GaussJacobiError {
    /// Captures a backtrace and creates a new GaussJacobiError with the given reason.
    #[inline]
    pub(crate) fn new(reason: GaussJacobiErrorReason) -> Self {
        Self {
            reason,
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns the reason for the error.
    #[inline]
    pub fn reason(&self) -> GaussJacobiErrorReason {
        self.reason
    }

    /// Returns a [`Backtrace`] to where the error was created.
    ///
    /// See [`Backtrace::capture`] for more information about how to make this display information when printed.
    #[inline]
    pub fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for GaussJacobiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason())
    }
}

impl std::error::Error for GaussJacobiError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GaussJacobiErrorReason {
    /// The degree was less than 2.
    Degree,
    /// The `alpha` exponent was less than or equal to -1.
    Alpha,
    /// The `beta` exponent was less than or equal to -1.
    Beta,
    /// Both the `alpha` and `beta` exponents were less than or equal to -1.
    AlphaBeta,
    /// The degree was less than 2 and the `alpha` exponent was less than or equal to -1.
    DegreeAlpha,
    /// The degree was less than 2 and the `beta` exponent was less than or equal to -1.
    DegreeBeta,
    /// The degree was less than 2 and both the `alpha` and `beta` exponents were less than or equal to -1.
    DegreeAlphaBeta,
}

impl GaussJacobiErrorReason {
    /// Returns true if the given degree, `deg`, was bad.
    #[inline]
    pub fn was_bad_degree(&self) -> bool {
        matches!(
            self,
            Self::Degree | Self::DegreeAlpha | Self::DegreeBeta | Self::DegreeAlphaBeta
        )
    }

    /// Returns true if the given `alpha` exponent was bad.
    #[inline]
    pub fn was_bad_alpha(&self) -> bool {
        matches!(
            self,
            Self::Alpha | Self::DegreeAlpha | Self::AlphaBeta | Self::DegreeAlphaBeta
        )
    }

    /// Returns true if the given `beta` exponent was bad.
    #[inline]
    pub fn was_bad_beta(&self) -> bool {
        matches!(
            self,
            Self::Beta | Self::DegreeBeta | Self::AlphaBeta | Self::DegreeAlphaBeta
        )
    }
}

use core::fmt;
impl fmt::Display for GaussJacobiErrorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DEGREE_LIMIT: &str = "must be at least 2";
        const EXPONENT_LIMIT: &str = "must be finite and larger than -1.0";
        match self {
            Self::Degree => write!(f, "degree {DEGREE_LIMIT}"),
            Self::Alpha => write!(f, "alpha {EXPONENT_LIMIT}"),
            Self::Beta => write!(f, "beta {EXPONENT_LIMIT}"),
            Self::AlphaBeta => write!(f, "alpha and beta {EXPONENT_LIMIT}"),
            Self::DegreeAlpha => write!(f, "degree {DEGREE_LIMIT} and alpha {EXPONENT_LIMIT}"),
            Self::DegreeBeta => write!(f, "degree {DEGREE_LIMIT} and beta {EXPONENT_LIMIT}"),
            Self::DegreeAlphaBeta => {
                write!(f, "degree {DEGREE_LIMIT}, alpha and beta {EXPONENT_LIMIT}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn check_alpha_beta_bounds() {
        assert!(GaussJacobi::new(10, -1.0, -1.0).is_err());
    }

    #[test]
    fn golub_welsch_5_alpha_0_beta_0() {
        let (x, w): (Vec<_>, Vec<_>) = GaussJacobi::new(5, 0.0, 0.0).unwrap().into_iter().unzip();
        let x_should = [
            -0.906_179_845_938_664,
            -0.538_469_310_105_683_1,
            0.0,
            0.538_469_310_105_683_1,
            0.906_179_845_938_664,
        ];
        let w_should = [
            0.236_926_885_056_189_08,
            0.478_628_670_499_366_47,
            0.568_888_888_888_888_9,
            0.478_628_670_499_366_47,
            0.236_926_885_056_189_08,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-15);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-15);
        }
    }

    #[test]
    fn golub_welsch_2_alpha_1_beta_0() {
        let (x, w): (Vec<_>, Vec<_>) = GaussJacobi::new(2, 1.0, 0.0).unwrap().into_iter().unzip();
        let x_should = [-0.689_897_948_556_635_7, 0.289_897_948_556_635_64];
        let w_should = [1.272_165_526_975_908_7, 0.727_834_473_024_091_3];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_1_beta_0() {
        let (x, w): (Vec<_>, Vec<_>) = GaussJacobi::new(5, 1.0, 0.0).unwrap().into_iter().unzip();
        let x_should = [
            -0.920_380_285_897_062_6,
            -0.603_973_164_252_783_7,
            0.0,
            0.390_928_546_707_272_2,
            0.802_929_828_402_347_2,
        ];
        let w_should = [
            0.387_126_360_906_606_74,
            0.668_698_552_377_478_2,
            0.585_547_948_338_679_2,
            0.295_635_480_290_466_66,
            0.062_991_658_086_769_1,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_5_alpha_0_beta_1() {
        let (x, w): (Vec<_>, Vec<_>) = GaussJacobi::new(5, 0.0, 1.0).unwrap().into_iter().unzip();
        let x_should = [
            -0.802_929_828_402_347_2,
            -0.390_928_546_707_272_2,
            0.0,
            0.603_973_164_252_783_7,
            0.920_380_285_897_062_6,
        ];
        let w_should = [
            0.062_991_658_086_769_1,
            0.295_635_480_290_466_66,
            0.585_547_948_338_679_2,
            0.668_698_552_377_478_2,
            0.387_126_360_906_606_74,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-14);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn golub_welsch_50_alpha_42_beta_23() {
        let (x, w): (Vec<_>, Vec<_>) = GaussJacobi::new(50, 42.0, 23.0)
            .unwrap()
            .into_iter()
            .unzip();
        let x_should = [
            -0.936_528_233_152_541_2,
            -0.914_340_864_546_088_5,
            -0.892_159_904_972_709_7,
            -0.869_216_909_221_225_6,
            -0.845_277_228_769_225_6,
            -0.820_252_766_348_056_8,
            -0.794_113_540_498_529_6,
            -0.766_857_786_572_463_5,
            -0.738_499_459_607_423_4,
            -0.709_062_235_514_446_8,
            -0.678_576_327_905_629_3,
            -0.647_076_661_181_635_3,
            -0.614_601_751_027_635_6,
            -0.581_192_977_458_508_4,
            -0.546_894_086_695_451_9,
            -0.511_750_831_826_105_3,
            -0.475_810_700_347_493_84,
            -0.439_122_697_460_417_9,
            -0.401_737_165_777_708_5,
            -0.363_705_629_046_518_04,
            -0.325_080_651_686_135_1,
            -0.285_915_708_544_232_9,
            -0.246_265_060_906_733_86,
            -0.206_183_635_819_408_85,
            -0.165_726_906_401_709_62,
            -0.124_950_771_176_147_79,
            -0.083_911_430_566_871_42,
            -0.042_665_258_670_068_65,
            -0.001_268_668_170_195_549_6,
            0.040_222_034_151_539_98,
            0.081_750_804_545_872_01,
            0.123_262_036_301_197_46,
            0.164_700_756_351_269_24,
            0.206_012_852_393_607_17,
            0.247_145_341_670_134_97,
            0.288_046_697_452_241,
            0.328_667_256_796_052_5,
            0.368_959_744_983_174_2,
            0.408_879_971_241_114_4,
            0.448_387_782_372_734_86,
            0.487_448_416_419_391_24,
            0.526_034_498_798_180_8,
            0.564_129_114_046_126_2,
            0.601_730_771_388_207_7,
            0.638_861_919_860_897_4,
            0.675_584_668_752_041_4,
            0.712_032_766_455_434_9,
            0.748_486_131_436_470_7,
            0.785_585_184_777_517_6,
            0.825_241_342_102_355_2,
        ];
        let w_should = [
            7.48575322545471E-18,
            4.368160045795394E-15,
            5.475_092_226_093_74E-13,
            2.883_802_894_000_164_4E-11,
            8.375_974_400_943_034E-10,
            1.551_169_281_097_026_6E-8,
            2.002_752_126_655_06E-7,
            1.914_052_885_645_138E-6,
            1.412_973_977_680_798E-5,
            8.315_281_580_948_582E-5,
            3.996_349_769_672_429E-4,
            0.001_598_442_290_393_378_4,
            0.005_401_484_462_492_892,
            0.015_609_515_951_961_325,
            0.038_960_859_894_776_14,
            0.084_675_992_815_357_84,
            0.161_320_272_041_780_37,
            0.270_895_707_022_142,
            0.402_766_052_144_190_03,
            0.532_134_840_644_357_2,
            0.626_561_850_396_477_3,
            0.658_939_504_140_677_5,
            0.619_968_794_555_102,
            0.522_392_634_872_676_4,
            0.394_418_806_923_720_8,
            0.266_845_588_852_137_27,
            0.161_693_943_297_351_4,
            0.087_665_230_931_323_02,
            0.042_462_146_242_945_82,
            0.018_336_610_588_859_478,
            0.007_040_822_524_198_700_5,
            0.002_395_953_515_750_436_4,
            7.196_709_691_248_771E-4,
            1.898_822_582_266_401E-4,
            4.375_352_582_937_183E-5,
            8.744_218_873_447_381E-6,
            1.503_255_708_913_270_4E-6,
            2.201_263_417_180_834_2E-7,
            2.713_269_374_479_116_4E-8,
            2.774_921_681_532_996E-9,
            2.313_546_085_591_984_2E-10,
            1.538_220_559_204_994_4E-11,
            7.931_012_545_002_62E-13,
            3.057_666_218_185_739E-14,
            8.393_076_986_026_449E-16,
            1.531_180_072_630_389E-17,
            1.675_381_720_821_777_5E-19,
            9.300_961_857_933_663E-22,
            1.912_538_194_408_499_4E-24,
            6.645_776_758_516_211E-28,
        ];
        for (i, x_val) in x_should.iter().enumerate() {
            assert_abs_diff_eq!(*x_val, x[i], epsilon = 1e-10);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            assert_abs_diff_eq!(*w_val, w[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn check_derives() {
        let quad = GaussJacobi::new(10, 0.0, 1.0).unwrap();
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussJacobi::new(10, 1.0, 0.0).unwrap();
        assert_ne!(quad, other_quad);
    }

    #[test]
    fn check_jacobi_error() {
        let jacobi_rule = GaussJacobi::new(3, -2.0, 1.0);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::Alpha));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "alpha must be finite and larger than -1.0"
        );

        let jacobi_rule = GaussJacobi::new(3, 1.0, -2.0);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::Beta));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "beta must be finite and larger than -1.0"
        );

        let jacobi_rule = GaussJacobi::new(3, -2.0, -2.0);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::AlphaBeta));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "alpha and beta must be finite and larger than -1.0"
        );
        assert_eq!(
            GaussJacobi::new(3, -2.0, 1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Alpha)
        );

        assert_eq!(
            GaussJacobi::new(3, -1.0, 1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Alpha)
        );

        assert_eq!(
            GaussJacobi::new(3, 1.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Beta)
        );

        assert_eq!(
            GaussJacobi::new(3, -1.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::AlphaBeta)
        );

        let jacobi_rule = GaussJacobi::new(0, 0.5, 0.5);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::Degree));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "degree must be at least 2"
        );
        assert_eq!(
            GaussJacobi::new(1, 0.5, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Degree)
        );

        let jacobi_rule = GaussJacobi::new(0, -1.0, 0.5);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::DegreeAlpha));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "degree must be at least 2 and alpha must be finite and larger than -1.0"
        );
        assert_eq!(
            GaussJacobi::new(0, -2.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussJacobi::new(1, -1.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );

        let jacobi_rule = GaussJacobi::new(0, 0.5, -1.0);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::DegreeBeta));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "degree must be at least 2 and beta must be finite and larger than -1.0"
        );
        assert_eq!(
            GaussJacobi::new(3, -2.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::AlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(3, -1.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::AlphaBeta)
        );

        let jacobi_rule = GaussJacobi::new(0, -1.0, -1.0);
        assert!(jacobi_rule
            .as_ref()
            .is_err_and(|x| x.reason() == GaussJacobiErrorReason::DegreeAlphaBeta));
        assert_eq!(
            format!("{}", jacobi_rule.err().unwrap()),
            "degree must be at least 2, alpha and beta must be finite and larger than -1.0"
        );
        assert_eq!(
            GaussJacobi::new(3, -2.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::AlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(3, -1.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::AlphaBeta)
        );

        assert_eq!(
            GaussJacobi::new(0, 0.5, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Degree)
        );
        assert_eq!(
            GaussJacobi::new(1, 0.5, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::Degree)
        );

        assert_eq!(
            GaussJacobi::new(0, -1.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussJacobi::new(0, -2.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussJacobi::new(1, -1.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );
        assert_eq!(
            GaussJacobi::new(1, -2.0, 0.5).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlpha)
        );

        assert_eq!(
            GaussJacobi::new(0, 0.5, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeBeta)
        );
        assert_eq!(
            GaussJacobi::new(0, 0.5, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, 0.5, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, 0.5, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeBeta)
        );

        assert_eq!(
            GaussJacobi::new(0, -1.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(0, -2.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, -1.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, -2.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(0, -1.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(0, -2.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, -1.0, -2.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
        assert_eq!(
            GaussJacobi::new(1, -2.0, -1.0).map_err(|e| e.reason()),
            Err(GaussJacobiErrorReason::DegreeAlphaBeta)
        );
    }

    #[test]
    fn check_iterators() {
        let rule = GaussJacobi::new(2, -0.25, -0.5).unwrap();
        // Answer taken from Wolfram Alpha <https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BDivide%5BPower%5Bx%2C2%5D%2CPower%5B%5C%2840%291-x%5C%2841%29%2CDivide%5B1%2C4%5D%5DPower%5B%5C%2840%291%2Bx%5C%2841%29%2CDivide%5B1%2C2%5D%5D%5D%2C%7Bx%2C-1%2C1%7D%5D>
        let ans = 1.3298477657906902;

        assert_abs_diff_eq!(
            ans,
            rule.iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            ans,
            rule.nodes()
                .zip(rule.weights())
                .fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            ans,
            rule.into_iter().fold(0.0, |tot, (n, w)| tot + n * n * w),
            epsilon = 1e-14
        );
    }

    #[test]
    fn check_some_integrals() {
        let rule = GaussJacobi::new(10, -0.5, -0.25).unwrap();

        assert_abs_diff_eq!(
            rule.integrate(-1.0, 1.0, |x| x * x),
            1.3298477657906902,
            epsilon = 1e-14
        );

        assert_abs_diff_eq!(
            rule.integrate(-1.0, 1.0, |x| x.cos()),
            2.2239,
            epsilon = 1e-5
        );
    }
}
