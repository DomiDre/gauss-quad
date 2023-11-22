/// This macro implements the data access API for the given quadrature rule struct.
/// It takes in the name of the quadrature rule as well as the names of the iterators
/// over its nodes, weights, both, and IntoIterator implementation.
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
        $quadrature_rule_iter:ident
    ) => {
        // The functions in this impl block all have an #[inline] directive because they are trivial.
        impl $quadrature_rule {
            /// Returns an iterator over the nodes of the quadrature rule.
            #[inline]
            pub fn nodes(&self) -> $quadrature_rule_nodes<'_> {
                $quadrature_rule_nodes::new(self.node_weight_pairs.iter().map(|p| &p.0))
            }

            /// Returns an iterator over the weights of the quadrature rule.
            #[inline]
            pub fn weights(&self) -> $quadrature_rule_weights<'_> {
                $quadrature_rule_weights::new(self.node_weight_pairs.iter().map(|p| &p.1))
            }

            /// Returns an iterator over the node-weight-pairs of the quadrature rule.
            #[inline]
            pub fn iter(&self) -> $quadrature_rule_iter<'_> {
                $quadrature_rule_iter::new(self.node_weight_pairs.iter())
            }

            /// Returns a slice of the node-weight-pairs of the quadrature rule.
            #[inline]
            pub fn as_node_weight_pairs(&self) -> &[(f64, f64)] {
                &self.node_weight_pairs
            }

            /// Converts the quadrature rule into a vector of node-weight-pairs.
            ///
            /// Element `.0` is the nodes of the rule and element `.1` is the weights.
            ///
            /// This function just returns the underlying data and does no
            /// computation or cloning.
            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_node_weight_pairs(self) -> Vec<(f64, f64)> {
                self.node_weight_pairs
            }

            /// Returns the degree of the quadrature rule.
            #[inline]
            pub fn degree(&self) -> usize {
                self.node_weight_pairs.len()
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_iterators {
    ($quadrature_rule:ident, $quadrature_rule_nodes:ident, $quadrature_rule_weights:ident, $quadrature_rule_iter:ident, $quadrature_rule_into_iter:ident) => {
        slice_iter_map_impl! {$quadrature_rule_nodes}
        slice_iter_map_impl! {$quadrature_rule_weights}

        /// An iterator over node-weight-pairs of the quadrature rule.
        ///
        /// Created by the `iter` function on the quadrature rule struct.
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a> {
            node_weight_pairs: ::core::slice::Iter<'a, (f64, f64)>,
        }

        impl<'a> $quadrature_rule_iter<'a> {
            pub(super) fn new(node_weight_pairs: ::core::slice::Iter<'a, (f64, f64)>) -> Self {
                Self { node_weight_pairs }
            }
        }

        impl<'a> $quadrature_rule_iter<'a> {
            /// Returns a view of the underlying data.
            pub fn as_slice(&self) -> &'a [(f64, f64)] {
                self.node_weight_pairs.as_slice()
            }
        }

        impl<'a> ::core::iter::Iterator for $quadrature_rule_iter<'a> {
            type Item = &'a (f64, f64);
            fn next(&mut self) -> Option<Self::Item> {
                self.node_weight_pairs.next()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_iter<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.node_weight_pairs.next_back()
            }
        }

        /// An owning iterator over the nodes and weights of the quadrature rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the quadrature rule struct.
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter {
            node_weight_pairs: ::std::vec::IntoIter<(f64, f64)>,
        }

        impl ::core::iter::Iterator for $quadrature_rule_into_iter {
            type Item = (f64, f64);
            fn next(&mut self) -> Option<Self::Item> {
                self.node_weight_pairs.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.node_weight_pairs.size_hint()
            }
        }

        impl ::core::iter::DoubleEndedIterator for $quadrature_rule_into_iter {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.node_weight_pairs.next_back()
            }
        }

        impl ::core::iter::ExactSizeIterator for $quadrature_rule_into_iter {}
        impl ::core::iter::FusedIterator for $quadrature_rule_into_iter {}

        impl $quadrature_rule_into_iter {
            pub(super) fn new(node_weight_pairs: ::std::vec::IntoIter<(f64, f64)>) -> Self {
                Self { node_weight_pairs }
            }

            /// Returns a view into the underlying data as a slice of tuples.
            ///
            /// Element `.0` of the tuples is the node and element `.1` its corresponding weight.
            #[inline]
            pub fn as_slice(&self) -> &[(f64, f64)] {
                self.node_weight_pairs.as_slice()
            }
        }

        impl ::core::iter::IntoIterator for $quadrature_rule {
            type IntoIter = $quadrature_rule_into_iter;
            type Item = (f64, f64);
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_into_iter::new(self.node_weight_pairs.into_iter())
            }
        }
    };
}

/// This macro defines a struct with the given name that contains a [`core::slice::Iter`].
/// It then implements the [`Iterator`], [`DoubleEndedIterator`], [`ExactSizeIterator`], and [`FusedIterator`]
///  traits for it, and the convenience method `as_slice`.
#[doc(hidden)]
#[macro_export]
macro_rules! slice_iter_map_impl {
    ($slice_iter:ident) => {
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $slice_iter<'a> {
            iter_map: ::std::iter::Map<
                ::core::slice::Iter<'a, (f64, f64)>,
                fn(&'a (f64, f64)) -> &'a f64,
            >,
        }

        impl<'a> $slice_iter<'a> {
            pub(super) fn new(
                iter_map: ::std::iter::Map<
                    ::core::slice::Iter<'a, (f64, f64)>,
                    fn(&'a (f64, f64)) -> &'a f64,
                >,
            ) -> Self {
                Self { iter_map }
            }
        }

        impl<'a> ::core::iter::Iterator for $slice_iter<'a> {
            type Item = &'a f64;
            fn next(&mut self) -> Option<Self::Item> {
                self.iter_map.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.iter_map.size_hint()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $slice_iter<'a> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.iter_map.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $slice_iter<'a> {}
        impl<'a> ::core::iter::FusedIterator for $slice_iter<'a> {}
    };
}
