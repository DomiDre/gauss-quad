//! This module contains the implementation of the iterators that
//! can be returned by method calls on a [`Simpson`](super::Simpson) instance.

use crate::Node;
use core::{iter::FusedIterator, slice::Iter};

/// An iterator of the nodes of a [`Simpson`](super::Simpson) instance.
/// Created by the [`Simpson::iter`](super::Simpson::iter) function, see it for more information.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Debug, Clone)]
pub struct SimpsonIter<'a>(Iter<'a, Node>);

impl<'a> SimpsonIter<'a> {
    #[inline]
    pub(super) fn new(iter: Iter<'a, Node>) -> Self {
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

impl<'a> AsRef<[Node]> for SimpsonIter<'a> {
    #[inline]
    fn as_ref(&self) -> &[Node] {
        self.0.as_ref()
    }
}

impl<'a> Iterator for SimpsonIter<'a> {
    type Item = &'a Node;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a> DoubleEndedIterator for SimpsonIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a> ExactSizeIterator for SimpsonIter<'a> {}
impl<'a> FusedIterator for SimpsonIter<'a> {}
