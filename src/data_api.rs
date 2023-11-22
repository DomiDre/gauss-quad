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
        // The functions in this impl block all have an #[inline] directive because they are trivial.
        impl $quadrature_rule {
            /// Returns an iterator over the nodes of the quadrature rule.
            #[inline]
            pub fn nodes(&self) -> $quadrature_rule_nodes<'_> {
                $quadrature_rule_nodes {
                    iter: self.nodes.iter(),
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
                    iter: self.weights.iter(),
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
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
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
        #[must_use = "iterators are lazy and do nothing unless consumed"]
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
                // Only need to check one, the lengths
                // were asserted to be equal upon the creation of the struct.
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

        impl ::core::iter::IntoIterator for $quadrature_rule {
            type IntoIter = $quadrature_rule_into_iter;
            type Item = (f64, f64);
            fn into_iter(self) -> Self::IntoIter {
                assert_eq!(self.nodes.len(), self.weights.len(), "internal error, please file an issue at <https://github.com/DomiDre/gauss-quad>");
                let l = self.nodes.len();
                $quadrature_rule_into_iter {
                    nodes: self.nodes,
                    weights: self.weights,
                    index: 0,
                    back_index: l,
                }
            }
        }
    };
}

/// This macro defines a struct with the given name that contains a [`core::slice::Iter`].
/// It then implements the [`Iterator`], [`DoubleEndedIterator`], [`ExactSizeIterator`], and [`FusedIterator`]
///  traits for it, and the convenience method `as_slice`.
#[doc(hidden)]
#[macro_export]
macro_rules! slice_iter_impl {
    ($slice_iter:ident) => {
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $slice_iter<'a> {
            iter: ::core::slice::Iter<'a, f64>,
        }

        impl<'a> ::core::iter::Iterator for $slice_iter<'a> {
            type Item = &'a f64;
            fn next(&mut self) -> Option<Self::Item> {
                self.iter.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter.size_hint()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $slice_iter<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.iter.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $slice_iter<'a> {}
        impl<'a> ::core::iter::FusedIterator for $slice_iter<'a> {}

        impl<'a> $slice_iter<'a> {
            /// Returns a view of the underlying data as a slice.
            #[inline]
            pub fn as_slice(&self) -> &'a [f64] {
                self.iter.as_slice()
            }
        }
    };
}
