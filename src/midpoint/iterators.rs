//! This crate contains the implementation of the iterators that
//! can be returned by method calls on a [`Midpoint`](super::Midpoint) instance.

use crate::Node;
use core::iter::FusedIterator;
use core::slice::Iter;

/// An iterator of the nodes of a [`Midpoint`](super::Midpoint) instance.
/// Created by the [`Midpoint::iter`](super::Midpoint::iter) function, see it for more information.
#[derive(Debug, Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct MidpointIter<'a>(Iter<'a, f64>);

impl<'a> MidpointIter<'a> {
    #[inline]
    pub(super) fn new(iter: Iter<'a, f64>) -> Self {
        Self(iter)
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// See [`core::slice::Iter::as_slice`] for more information.
    #[inline]
    pub fn as_slice(&self) -> &'a [Node] {
        self.0.as_slice()
    }
}

impl<'a> AsRef<[Node]> for MidpointIter<'a> {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        self.0.as_ref()
    }
}

impl<'a> Iterator for MidpointIter<'a> {
    type Item = &'a Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
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
