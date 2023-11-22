//! This module contains the iterators produced by some of the functions on [`GaussLegendre`].

use super::GaussLegendre;
use crate::{impl_iterators, slice_iter_impl};

impl_iterators! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}
