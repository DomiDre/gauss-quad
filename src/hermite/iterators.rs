//! This module contains the iterators produced by some of the functions on [`GaussHermite`].

use super::GaussHermite;
use crate::{impl_iterators, Node, Weight};

impl_iterators! {GaussHermite, GaussHermiteNodes, GaussHermiteWeights, GaussHermiteIter, GaussHermiteIntoIter}
