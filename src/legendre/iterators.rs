//! This module contains the iterators produced by some of the functions on [`GaussLegendre`].

use super::GaussLegendre;

crate::impl_iterators! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}

#[cfg(test)]
mod test {
    use super::GaussLegendre;
    use approx::assert_abs_diff_eq;

    #[test]
    fn iterator_sanity_check() {
        for deg in (10..=100).step_by(10) {
            let rule = GaussLegendre::new(deg);
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
