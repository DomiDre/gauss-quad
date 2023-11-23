//! This module contains the iterators produced by some of the functions on [`GaussHermite`].

use super::GaussHermite;

crate::impl_iterators! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}
