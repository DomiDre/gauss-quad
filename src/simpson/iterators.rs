//! This module contains the implementation of the iterators that
//! can be returned by method calls on a [`Simpson`](super::Simpson) instance.

use crate::impl_node_rule_iterators;

impl_node_rule_iterators! {SimpsonIter, SimpsonIntoIter}
