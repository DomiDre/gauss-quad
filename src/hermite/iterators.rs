//! This module contains the iterators produced by some of the functions on [`GaussHermite`].

use super::GaussHermite;
use crate::{impl_iterators, slice_map_iter_impl, Node, Weight};

impl_iterators! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}
