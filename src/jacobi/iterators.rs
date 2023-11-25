//! This module contains the iterators produced by some of the functions on [`GaussJacobi`].

use super::GaussJacobi;

crate::impl_node_weight_rule_iterators! {GaussJacobi, GaussJacobiNodes, GaussJacobiWeights, GaussJacobiIter, GaussJacobiIntoIter}
