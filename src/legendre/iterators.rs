//! This module contains the iterators produced by some of the functions on [`GaussLegendre`].

use super::GaussLegendre;

crate::impl_iterators! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}
