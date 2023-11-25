//! This crate contains the implementation of the iterators that
//! can be returned by method calls on a [`Midpoint`] instance.

use super::Midpoint;
use crate::impl_node_rule_iterators;

impl_node_rule_iterators! {Midpoint, MidpointIter, MidpointIntoIter}
