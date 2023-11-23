//! This crate contains the implementation of the iterators that
//! can be returned by method calls on a [`Midpoint`](super::Midpoint) instance.

use crate::Node;
use core::iter::FusedIterator;

/// An iterator of the nodes of a [`Midpoint`](super::Midpoint) instance created by
/// [`Midpoint::iter`](super::Midpoint::iter).
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct MidpointIter<'a>(core::slice::Iter<'a, f64>);

impl<'a> MidpointIter<'a> {
    pub(super) fn new(iter: core::slice::Iter<'a, f64>) -> Self {
        Self(iter)
    }
}

impl<'a> Iterator for MidpointIter<'a> {
    type Item = &'a Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a> DoubleEndedIterator for MidpointIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a> ExactSizeIterator for MidpointIter<'a> {}
impl<'a> FusedIterator for MidpointIter<'a> {}
