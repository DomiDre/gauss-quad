//! This module contains the iterators produced by some of the functions on [`GaussLaguerre`].

use super::GaussLaguerre;
use crate::{impl_data_api, slice_iter_impl};

impl_data_api! {GaussLaguerre, GaussLaguerreNodes, GaussLaguerreWeights, GaussLaguerreIter, GaussLaguerreIntoIter}
