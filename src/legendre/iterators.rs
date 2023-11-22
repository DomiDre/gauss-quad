//! This module contains the iterators produced by some of the functions on [`GaussLegendre`].

use super::GaussLegendre;
use crate::{impl_data_api, slice_iter_impl};

impl_data_api! {GaussLegendre, GaussLegendreNodes, GaussLegendreWeights, GaussLegendreIter, GaussLegendreIntoIter}
