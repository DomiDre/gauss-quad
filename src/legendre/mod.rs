//! Numerical integration using the Gauss-Legendre quadrature rule.
//!
//! A Gauss-Legendre quadrature rule of degree `n` can integrate
//! degree 2`n`-1 polynomials exactly.
//!
//! Evaluation point x_i of a degree n rule is the i:th root
//! of Legendre polynomial P_n and its weight is  
//! w = 2 / ((1 - x_i)(P'_n(x_i))^2).
//!
//!
//! # Example
//! 
//! ```
//! use gauss_quad::legendre::GaussLegendre;
//! # use gauss_quad::legendre::GaussLegendreError;
//! use approx::assert_abs_diff_eq;
//!
//! let quad = GaussLegendre::new(10)?;
//! let integral = quad.integrate(-1.0, 1.0,
//!     |x| 0.125 * (63.0 * x.powi(5) - 70.0 * x.powi(3) + 15.0 * x)
//! );
//! assert_abs_diff_eq!(integral, 0.0);
//! # Ok::<(), GaussLegendreError>(())
//! ```

mod bogaert;

use bogaert::NodeWeightPair;

use crate::{impl_node_weight_rule, impl_node_weight_rule_iterators, Node, Weight};

/// A Gauss-Legendre quadrature scheme.
///
/// These rules can integrate functions on the domain [a, b].
///
/// # Examples
///
/// Basic usage:
/// ```
/// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
/// # use approx::assert_abs_diff_eq;
/// // initialize a Gauss-Legendre rule with 2 nodes
/// let quad = GaussLegendre::new(2)?;
///
/// // numerically integrate x^2 - 1/3 over the domain [0, 1]
/// let integral = quad.integrate(0.0, 1.0, |x| x * x - 1.0 / 3.0);
///
/// assert_abs_diff_eq!(integral, 0.0);
/// # Ok::<(), GaussLegendreError>(())
/// ```
/// The nodes and weights are computed in O(n) time,
/// so large quadrature rules are feasible:
/// ```
/// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
/// # use approx::assert_abs_diff_eq;
/// let quad = GaussLegendre::new(1_000_000)?;
///
/// let integral = quad.integrate(-3.0, 3.0, |x| x.sin());
///
/// assert_abs_diff_eq!(integral, 0.0);
/// # Ok::<(), GaussLegendreError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLegendre {
    node_weight_pairs: Vec<(Node, Weight)>,
}

impl GaussLegendre {
    /// Initializes a Gauss-Legendre quadrature rule of the given degree by computing the needed nodes and weights.
    ///
    /// Uses the [algorithm by Ignace Bogaert](https://doi.org/10.1137/140954969), which has linear time
    /// complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if `deg` is smaller than 2.
    pub fn new(deg: usize) -> Result<Self, GaussLegendreError> {
        if deg < 2 {
            return Err(GaussLegendreError);
        }

        Ok(Self {
            node_weight_pairs: (1..deg + 1)
                .map(|k: usize| NodeWeightPair::new(deg, k).into_tuple())
                .collect(),
        })
    }

    fn argument_transformation(x: f64, a: f64, b: f64) -> f64 {
        0.5 * ((b - a) * x + (b + a))
    }

    fn scale_factor(a: f64, b: f64) -> f64 {
        0.5 * (b - a)
    }

    /// Perform quadrature integration of given integrand from `a` to `b`.
    ///
    /// # Example
    ///
    /// Basic usage
    /// ```
    /// # use gauss_quad::legendre::{GaussLegendre, GaussLegendreError};
    /// # use approx::assert_abs_diff_eq;
    /// let glq_rule = GaussLegendre::new(3)?;
    ///
    /// assert_abs_diff_eq!(glq_rule.integrate(-1.0, 1.0, |x| x.powi(5)), 0.0);
    ///
    /// # Ok::<(), GaussLegendreError>(())
    /// ```
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
}

impl_node_weight_rule! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

impl_node_weight_rule_iterators! {GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

/// The error returned by [`GaussLegendre::new`] if it's given a degree of 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussLegendreError;

use core::fmt;
impl fmt::Display for GaussLegendreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the degree of the Gauss-Legendre quadrature rule must be at least 2"
        )
    }
}

impl std::error::Error for GaussLegendreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_degree_3() {
        let (x, w): (Vec<_>, Vec<_>) = GaussLegendre::new(3).unwrap().into_iter().unzip();

        let x_should = [0.7745966692414834, 0.0000000000000000, -0.7745966692414834];
        let w_should = [0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
        for (i, x_val) in x_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i]);
        }
        for (i, w_val) in w_should.iter().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i]);
        }
    }

    #[test]
    fn check_degree_128() {
        // A Legendre quadrature rule with degree > 100 to test calculation with non tabulated values
        let (x, w): (Vec<_>, Vec<_>) = GaussLegendre::new(128).unwrap().into_iter().unzip();

        // comparison values copied from http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/#gauss_quadrature_abscissas_table
        #[rustfmt::skip]
        let x_should = [0.0122236989606157641980521,0.0366637909687334933302153,0.0610819696041395681037870,0.0854636405045154986364980,0.1097942311276437466729747,0.1340591994611877851175753,0.1582440427142249339974755,0.1823343059853371824103826,0.2063155909020792171540580,0.2301735642266599864109866,0.2538939664226943208556180,0.2774626201779044028062316,0.3008654388776772026671541,0.3240884350244133751832523,0.3471177285976355084261628,0.3699395553498590266165917,0.3925402750332674427356482,0.4149063795522750154922739,0.4370245010371041629370429,0.4588814198335521954490891,0.4804640724041720258582757,0.5017595591361444642896063,0.5227551520511754784539479,0.5434383024128103634441936,0.5637966482266180839144308,0.5838180216287630895500389,0.6034904561585486242035732,0.6228021939105849107615396,0.6417416925623075571535249,0.6602976322726460521059468,0.6784589224477192593677557,0.6962147083695143323850866,0.7135543776835874133438599,0.7304675667419088064717369,0.7469441667970619811698824,0.7629743300440947227797691,0.7785484755064119668504941,0.7936572947621932902433329,0.8082917575079136601196422,0.8224431169556438424645942,0.8361029150609068471168753,0.8492629875779689691636001,0.8619154689395484605906323,0.8740527969580317986954180,0.8856677173453972174082924,0.8967532880491581843864474,0.9073028834017568139214859,0.9173101980809605370364836,0.9267692508789478433346245,0.9356743882779163757831268,0.9440202878302201821211114,0.9518019613412643862177963,0.9590147578536999280989185,0.9656543664319652686458290,0.9717168187471365809043384,0.9771984914639073871653744,0.9820961084357185360247656,0.9864067427245862088712355,0.9901278184917343833379303,0.9932571129002129353034372,0.9957927585349811868641612,0.9977332486255140198821574,0.9990774599773758950119878,0.9998248879471319144736081];

        #[rustfmt::skip]
        let w_should = [0.0244461801962625182113259,0.0244315690978500450548486,0.0244023556338495820932980,0.0243585572646906258532685,0.0243002001679718653234426,0.0242273192228152481200933,0.0241399579890192849977167,0.0240381686810240526375873,0.0239220121367034556724504,0.0237915577810034006387807,0.0236468835844476151436514,0.0234880760165359131530253,0.0233152299940627601224157,0.0231284488243870278792979,0.0229278441436868469204110,0.0227135358502364613097126,0.0224856520327449668718246,0.0222443288937997651046291,0.0219897106684604914341221,0.0217219495380520753752610,0.0214412055392084601371119,0.0211476464682213485370195,0.0208414477807511491135839,0.0205227924869600694322850,0.0201918710421300411806732,0.0198488812328308622199444,0.0194940280587066028230219,0.0191275236099509454865185,0.0187495869405447086509195,0.0183604439373313432212893,0.0179603271850086859401969,0.0175494758271177046487069,0.0171281354231113768306810,0.0166965578015892045890915,0.0162550009097851870516575,0.0158037286593993468589656,0.0153430107688651440859909,0.0148731226021473142523855,0.0143943450041668461768239,0.0139069641329519852442880,0.0134112712886163323144890,0.0129075627392673472204428,0.0123961395439509229688217,0.0118773073727402795758911,0.0113513763240804166932817,0.0108186607395030762476596,0.0102794790158321571332153,0.0097341534150068058635483,0.0091830098716608743344787,0.0086263777986167497049788,0.0080645898904860579729286,0.0074979819256347286876720,0.0069268925668988135634267,0.0063516631617071887872143,0.0057726375428656985893346,0.0051901618326763302050708,0.0046045842567029551182905,0.0040162549837386423131943,0.0034255260409102157743378,0.0028327514714579910952857,0.0022382884309626187436221,0.0016425030186690295387909,0.0010458126793403487793129,0.0004493809602920903763943];

        for (i, x_val) in x_should.iter().rev().enumerate() {
            approx::assert_abs_diff_eq!(*x_val, x[i], epsilon = 0.000_000_1);
        }
        for (i, w_val) in w_should.iter().rev().enumerate() {
            approx::assert_abs_diff_eq!(*w_val, w[i], epsilon = 0.000_000_1);
        }
    }

    #[test]
    fn check_legendre_error() {
        assert!(GaussLegendre::new(0).is_err());
        assert!(GaussLegendre::new(1).is_err());
    }

    #[test]
    fn check_derives() {
        let quad = GaussLegendre::new(10);
        let quad_clone = quad.clone();
        assert_eq!(quad, quad_clone);
        let other_quad = GaussLegendre::new(3);
        assert_ne!(quad, other_quad);
    }
}
