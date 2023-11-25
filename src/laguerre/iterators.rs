//! This module contains the iterators produced by some of the functions on [`GaussLaguerre`].

use super::GaussLaguerre;

crate::impl_node_weight_rule_iterators! {GaussLaguerre, GaussLaguerreNodes, GaussLaguerreWeights, GaussLaguerreIter, GaussLaguerreIntoIter}
