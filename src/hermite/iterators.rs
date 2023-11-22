//! This module contains the iterators produced by some of the functions on [`GaussHermite`].

use super::GaussHermite;
use crate::{impl_data_api, slice_iter_impl};

impl_data_api! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}
