//! This crate contains the implementation of the iterators that
//! can be returned by method calls on a [`Midpoint`](super::Midpoint) instance.

use crate::impl_node_rule_iterators;

impl_node_rule_iterators! {MidpointIter, MidpointIntoIter}
