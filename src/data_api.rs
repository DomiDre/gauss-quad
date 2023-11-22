/// This macro implements the data access API for the given quadrature rule struct.
/// It takes in the name of the quadrature rule as well as the names of the iterators
/// over its nodes, weights, both (optional), and IntoIterator implementation.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_data_api {
    (
        // The name of the quadrature rule struct, e.g. GaussLegendre.
        $quadrature_rule:ident,
        // The name that the iterator over the nodes should have, e.g. GaussLegendreNodes.
        $quadrature_rule_nodes:ident,
        // The name that the iterator over the weights should have, e.g. GaussLegendreWeights.
        $quadrature_rule_weights:ident,
        // The name that the iterator returned when calling the `iter` function should have,
        // e.g. GaussLegendreIter.
        $quadrature_rule_iter:ident,
        // The name that the iterator returned by the `into_iter` function of the IntoIterator
        // trait should have, e.g. GaussLegendreIntoIter.
        $quadrature_rule_into_iter:ident
    ) => {
        impl $quadrature_rule {
            /// Returns an iterator over the nodes of the quadrature rule.
            #[inline]
            pub fn nodes(&self) -> $quadrature_rule_nodes<'_> {
                $quadrature_rule_nodes {
                    slice: &self.nodes,
                    index: 0,
                    back_index: self.nodes.len(),
                }
            }

            /// Returns a slice of the nodes of the quadrature rule.
            #[inline]
            pub fn as_nodes(&self) -> &[f64] {
                &self.nodes
            }

            /// Returns an iterator over the weights of the quadrature rule.
            #[inline]
            pub fn weights(&self) -> $quadrature_rule_weights<'_> {
                $quadrature_rule_weights {
                    slice: &self.weights,
                    index: 0,
                    back_index: self.weights.len(),
                }
            }

            /// Returns a slice of the weights of the quadrature rule.
            #[inline]
            pub fn as_weights(&self) -> &[f64] {
                &self.weights
            }

            /// Returns an iterator over pairs of nodes and the corresponding weights of the quadrature rule.
            #[inline]
            pub fn iter(&self) -> $quadrature_rule_iter<'_> {
                $quadrature_rule_iter {
                    node_iter: self.nodes(),
                    weight_iter: self.weights(),
                }
            }

            /// Converts the quadrature rule into a tuple of vectors.
            ///
            /// Element `.0` is the nodes of the rule and element `.1` is the weights.
            ///
            /// This function just returns the underlying data and does no
            /// computation or cloning.
            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_nodes_and_weights(self) -> (Vec<f64>, Vec<f64>) {
                (self.nodes, self.weights)
            }
        }

        slice_iter_impl! {$quadrature_rule_nodes}
        slice_iter_impl! {$quadrature_rule_weights}

        /// An iterator over the quadrature rule's nodes and weights.
        ///
        /// Created by the `iter` function on the quadrature rule.
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use = "iterators do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a> {
            node_iter: $quadrature_rule_nodes<'a>,
            weight_iter: $quadrature_rule_weights<'a>,
        }

        impl<'a> ::core::iter::Iterator for $quadrature_rule_iter<'a> {
            type Item = (&'a f64, &'a f64);
            fn next(&mut self) -> Option<Self::Item> {
                match (self.node_iter.next(), self.weight_iter.next()) {
                    (Some(x), Some(w)) => Some((x, w)),
                    _ => None,
                }
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_iter<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                match (self.node_iter.next_back(), self.weight_iter.next_back()) {
                    (Some(x), Some(w)) => Some((x, w)),
                    _ => None,
                }
            }
        }

        /// An owning iterator over the nodes and weights of the quadrature rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the quadrature rule struct.
        #[derive(Debug, Clone, PartialEq)]
        #[must_use = "iterators do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter {
            nodes: Vec<f64>,
            weights: Vec<f64>,
            index: usize,
            back_index: usize,
        }

        impl ::core::iter::Iterator for $quadrature_rule_into_iter {
            type Item = (f64, f64);
            fn next(&mut self) -> Option<Self::Item> {
                if self.index < self.back_index {
                    let out = Some((self.nodes[self.index], self.weights[self.index]));
                    self.index += 1;
                    out
                } else {
                    None
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                assert_eq!(self.nodes.len(), self.weights.len());
                let len = self.nodes.len();
                (len, Some(len))
            }
        }

        impl ::core::iter::DoubleEndedIterator for $quadrature_rule_into_iter {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index < self.back_index {
                    let out = Some((self.nodes[self.back_index], self.weights[self.back_index]));
                    self.back_index -= 1;
                    out
                } else {
                    None
                }
            }
        }

        impl ::core::iter::ExactSizeIterator for $quadrature_rule_into_iter {}
        impl ::core::iter::FusedIterator for $quadrature_rule_into_iter {}

        impl $quadrature_rule_into_iter {
            /// Returns a view into the underlying data as a tuple of slices.
            ///
            /// Element `.0` is a slice of nodes and element `.1` is a slice
            /// of their corresponding weights.
            #[inline]
            pub fn as_slices(&self) -> (&[f64], &[f64]) {
                (
                    &self.nodes[self.index..self.back_index],
                    &self.weights[self.index..self.back_index],
                )
            }
        }
    };
}

/// This macro defines a struct with the given name that contains a slice and two
/// indices, one from the front and one from the back.
/// It then implements the [`Iterator`] trait for it, and the convenience method
/// `as_slice`. Kind of a rename of [`core::slice::Iter`].
#[doc(hidden)]
#[macro_export]
macro_rules! slice_iter_impl {
    ($slice_iter:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[must_use = "iterators do nothing unless consumed"]
        pub struct $slice_iter<'a> {
            slice: &'a [f64],
            index: usize,
            back_index: usize,
        }

        impl<'a> ::core::iter::Iterator for $slice_iter<'a> {
            type Item = &'a f64;
            fn next(&mut self) -> Option<Self::Item> {
                if self.index < self.back_index {
                    let out = Some(&self.slice[self.index]);
                    self.index += 1;
                    out
                } else {
                    None
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.back_index - self.index;
                (len, Some(len))
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $slice_iter<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index < self.back_index {
                    let out = Some(&self.slice[self.back_index]);
                    self.back_index -= 1;
                    out
                } else {
                    None
                }
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $slice_iter<'a> {}
        impl<'a> ::core::iter::FusedIterator for $slice_iter<'a> {}

        impl<'a> $slice_iter<'a> {
            /// Returns a view of the underlying data as a slice.
            #[inline]
            pub fn as_slice(&self) -> &[f64] {
                &self.slice[self.index..self.back_index]
            }
        }
    };
}
