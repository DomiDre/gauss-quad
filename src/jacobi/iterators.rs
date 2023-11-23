//! This module contains the iterators produced by some of the functions on [`GaussJacobi`].

use super::GaussJacobi;
use crate::{impl_iterators, Node, Weight};

impl_iterators! {GaussJacobi, GaussJacobiNodes, GaussJacobiWeights, GaussJacobiIter, GaussJacobiIntoIter}