//! This crate contains the implementation of the iterators that
//! can be returned by method calls on a [`Simpson`] instance.

use super::Simpson;
use crate::impl_node_rule_iterators;

impl_node_rule_iterators! { Simpson, SimpsonIter, SimpsonIntoIter}
