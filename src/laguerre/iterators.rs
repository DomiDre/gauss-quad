//! This module contains the iterators produced by some of the functions on [`GaussLaguerre`].

use super::GaussLaguerre;

crate::impl_iterators! {GaussLaguerre, GaussLaguerreNodes, GaussLaguerreWeights, GaussLaguerreIter, GaussLaguerreIntoIter}

#[cfg(test)]
mod test {
    use super::GaussLaguerre;
    use approx::assert_abs_diff_eq;

    #[test]
    fn iterator_sanity_check() {
        for deg in (10..=100).step_by(10) {
            let rule = GaussLaguerre::new(deg, -0.5);
            assert_eq!(rule.degree(), deg);
            for ((ni, wi), (nn, ww)) in rule.iter().zip(rule.nodes().zip(rule.weights())) {
                assert_abs_diff_eq!(ni, nn);
                assert_eq!(wi, ww);
            }
            for ((ni, wi), (nn, ww)) in rule
                .as_node_weight_pairs()
                .iter()
                .zip(rule.nodes().zip(rule.weights()))
            {
                assert_abs_diff_eq!(ni, nn);
                assert_eq!(wi, ww);
            }
        }
    }
}
