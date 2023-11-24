//! This module contains the iterators produced by some of the functions on [`GaussHermite`].

use super::GaussHermite;

crate::impl_iterators! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}

#[cfg(test)]
mod test {
    use super::GaussHermite;
    use approx::assert_abs_diff_eq;

    #[test]
    fn iterator_sanity_check() {
        for deg in (10..=100).step_by(10) {
            let rule = GaussHermite::new(deg);
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
