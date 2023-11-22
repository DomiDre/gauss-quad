//! This module contains the iterators produced by some of the functions on [`GaussJacobi`].

use super::GaussJacobi;
use crate::{impl_data_api, slice_iter_impl};

impl_data_api!{GaussJacobi, GaussJacobiNodes, GaussJacobiWeights, GaussJacobiIter, GaussJacobiIntoIter}