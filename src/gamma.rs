//! The gamma function provided in this module is copied directly from stats.rs:
//! <https://docs.rs/statrs/latest/src/statrs/function/gamma.rs.html>
//!
//! The reason for this is the reduction of dependencies.

use core::f64::consts::{E, PI};

use crate::elementary::{pow, sin};

/// Constant value for `2 * sqrt(e / pi)`
const TWO_SQRT_E_OVER_PI: f64 = 1.860_382_734_205_265_7;

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

/// Polynomial coefficients for approximating the `gamma_ln` function
static GAMMA_DK: &[f64] = &[
    2.485_740_891_387_535_5e-5,
    1.051_423_785_817_219_7,
    -3.456_870_972_220_162_5,
    4.512_277_094_668_948,
    -2.982_852_253_235_766_4,
    1.056_397_115_771_267,
    -1.954_287_731_916_458_7e-1,
    1.709_705_434_044_412e-2,
    -5.719_261_174_043_057e-4,
    4.633_994_733_599_057e-6,
    -2.719_949_084_886_077_2e-9,
];

/// Computes the gamma function with an accuracy
/// of 16 floating point digits. The implementation
/// is derived from "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub(crate) fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        PI / (sin(PI * x) * s * TWO_SQRT_E_OVER_PI * pow((0.5 - x + GAMMA_R) / E, 0.5 - x))
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s * TWO_SQRT_E_OVER_PI * pow((x - 0.5 + GAMMA_R) / E, x - 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::gamma;
    use approx::assert_abs_diff_eq;
    use core::f64::{self, consts};

    #[test]
    fn test_gamma() {
        assert!(gamma(f64::NAN).is_nan());
        assert_abs_diff_eq!(gamma(1.000001e-35), 9.999_990_000_01e34, epsilon = 1e20);
        assert_abs_diff_eq!(gamma(1.000001e-10), 9.999_989_999_432_785e9, epsilon = 1e-5);
        assert_abs_diff_eq!(gamma(1.000001e-5), 99_999.322_794_325_58, epsilon = 1e-10);
        assert_abs_diff_eq!(gamma(1.000001e-2), 99.432_485_128_962_57, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(-4.8), -0.062_423_361_354_759_55, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(-1.5), 2.363_271_801_207_355, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(-0.5), -3.544_907_701_811_032, epsilon = 1e-13);
        assert_abs_diff_eq!(
            gamma(1.0e-5 + 1.0e-16),
            99_999.422_793_225_57,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(gamma(0.1), 9.513_507_698_668_732, epsilon = 1e-14);
        assert_eq!(gamma(1.0 - 1.0e-14), 1.000_000_000_000_005_8);
        assert_abs_diff_eq!(gamma(1.0), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            gamma(1.0 + 1.0e-14),
            0.999_999_999_999_994_2,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(gamma(1.5), 0.886_226_925_452_758, epsilon = 1e-14);
        assert_abs_diff_eq!(
            gamma(consts::PI / 2.0),
            0.890_560_890_381_539_3,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(gamma(2.0), 1.0);
        assert_abs_diff_eq!(gamma(2.5), 1.329_340_388_179_137, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(3.0), 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(gamma(consts::PI), 2.288_037_795_340_032_6, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(3.5), 3.323_350_970_447_842_6, epsilon = 1e-14);
        assert_abs_diff_eq!(gamma(4.0), 6.0, epsilon = 1e-13);
        assert_abs_diff_eq!(gamma(4.5), 11.631_728_396_567_448, epsilon = 1e-12);
        assert_abs_diff_eq!(
            gamma(5.0 - 1.0e-14),
            23.999_999_999_999_638,
            epsilon = 1e-13
        );
        assert_abs_diff_eq!(gamma(5.0), 24.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            gamma(5.0 + 1.0e-14),
            24.000_000_000_000_362,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(gamma(5.5), 52.342_777_784_553_52, epsilon = 1e-12);
        assert_abs_diff_eq!(gamma(10.1), 454_760.751_441_585_95, epsilon = 1e-7);
        assert_abs_diff_eq!(
            gamma(150.0 + 1.0e-12),
            3.808_922_637_649_642e260,
            epsilon = 1e248
        );
    }
}
