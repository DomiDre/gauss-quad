//! This module contains two macros: [`impl_data_api!`] and [`impl_iterators!`].
//! The first takes in the name of a quadrature rule struct that has a field named `node_weight_pairs`
//! of the type `Vec<(f64, f64)>`. It also needs the names that it should give the various structs
//! that iterate over that data. It should be called in the module that defines the quadrature rule struct.
//! The second macro defines the iterators that the first returns. It should be called somewhere it makes sense
//! for the iterators to be defined, e.g. a sub-module.

// The code in the macros uses fully qualified paths for every type, so it is quite verbose.
// That is, instead of `usize` it uses `::core::primitive::usize` and so on. This makes it so that
// the caller of the macro doesn't have to import anything into the module in order for the macro to compile.

/// This macro implements the data access API for the given quadrature rule struct.
/// It takes in the name of the quadrature rule struct as well as the names of the iterators
/// over its nodes, weights, and both.
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
        // Implements functions for accessing the underlying data of the quadrature rule struct
        // in a way the adheres to the API guidelines: <https://rust-lang.github.io/api-guidelines/naming.html>.
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
            pub fn as_node_weight_pairs(&self) -> &[(Node, Weight)] {
                &self.node_weight_pairs
            }

            /// Converts the quadrature rule into a vector of node-weight-pairs.
            ///
            /// This function just returns the underlying data and does no
            /// computation or cloning.
            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_node_weight_pairs(self) -> ::std::vec::Vec<(Node, Weight)> {
                self.node_weight_pairs
            }

            /// Returns the degree of the quadrature rule.
            #[inline]
            pub fn degree(&self) -> ::core::primitive::usize {
                self.node_weight_pairs.len()
            }
        }
    };
}

/// This macro defines the iterators used by the functions defined in the macro [`impl_data_api!`].
/// It takes in the names of the same structs as that macro,
/// plus the name of the iterator that should be returned by the [`IntoIterator`] implementation.
/// These iterators can only be created in the module where the macro is called
/// or the module above it (due to the `pub(super)` marker on the constructors).
#[doc(hidden)]
#[macro_export]
macro_rules! impl_iterators {
    (
        $quadrature_rule:ident,
        $quadrature_rule_nodes:ident,
        $quadrature_rule_weights:ident,
        $quadrature_rule_iter:ident,
        // The name of the iterator that should be returned by the IntoIterator trait.
        $quadrature_rule_into_iter:ident
    ) => {
        // region: QuadratureRuleNodes

        /// An iterator over the nodes of the quadrature rule.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_nodes<'a>(
            ::std::iter::Map<
                ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Node,
            >,
        );

        impl<'a> $quadrature_rule_nodes<'a> {
            #[inline]
            pub(super) fn new(
                iter_map: ::std::iter::Map<
                    ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                    fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Node,
                >,
            ) -> Self {
                Self(iter_map)
            }
        }

        impl<'a> ::core::iter::Iterator for $quadrature_rule_nodes<'a> {
            type Item = &'a $crate::Node;
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(
                &self,
            ) -> (
                ::core::primitive::usize,
                ::core::option::Option<::core::primitive::usize>,
            ) {
                self.0.size_hint()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_nodes<'a> {
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $quadrature_rule_nodes<'a> {}
        impl<'a> ::core::iter::FusedIterator for $quadrature_rule_nodes<'a> {}

        // endregion: QuadratureRuleNodes

        // region: QuadratureRuleWeights

        /// An iterator over the weights of the quadrature rule.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_weights<'a>(
            ::std::iter::Map<
                ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Weight,
            >,
        );

        impl<'a> $quadrature_rule_weights<'a> {
            #[inline]
            pub(super) fn new(
                iter_map: ::std::iter::Map<
                    ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                    fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Weight,
                >,
            ) -> Self {
                Self(iter_map)
            }
        }

        impl<'a> ::core::iter::Iterator for $quadrature_rule_weights<'a> {
            type Item = &'a $crate::Weight;
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(
                &self,
            ) -> (
                ::core::primitive::usize,
                ::core::option::Option<::core::primitive::usize>,
            ) {
                self.0.size_hint()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_weights<'a> {
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $quadrature_rule_weights<'a> {}
        impl<'a> ::core::iter::FusedIterator for $quadrature_rule_weights<'a> {}

        // endregion: QuadratureRuleWeights

        // region: QuadratureRuleIter

        /// An iterator over node-weight-pairs of the quadrature rule.
        ///
        /// Created by the `iter` function on the quadrature rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a>(
            ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
        );

        impl<'a> $quadrature_rule_iter<'a> {
            #[inline]
            pub(super) fn new(
                node_weight_pairs: ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
            ) -> Self {
                Self(node_weight_pairs)
            }

            /// Views the underlying data as a subslice of the original data.
            ///
            /// See [`core::slice::Iter::as_slice`] for more information.
            #[inline]
            pub fn as_slice(&self) -> &'a [($crate::Node, $crate::Weight)] {
                self.0.as_slice()
            }
        }

        impl<'a> ::core::iter::Iterator for $quadrature_rule_iter<'a> {
            /// Element `.0` is the node and element `.1` the corresponding weight.
            type Item = &'a ($crate::Node, $crate::Weight);
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_iter<'a> {
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $quadrature_rule_iter<'a> {}
        impl<'a> ::core::iter::FusedIterator for $quadrature_rule_iter<'a> {}

        impl<'a> ::core::convert::AsRef<[($crate::Node, $crate::Weight)]>
            for $quadrature_rule_iter<'a>
        {
            #[inline]
            fn as_ref(&self) -> &[($crate::Node, $crate::Weight)] {
                self.0.as_ref()
            }
        }

        // endregion: QuadratureRuleIter

        // region: QuadratureRuleIntoIter

        /// An owning iterator over the node-weight-pairs of the quadrature rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the quadrature rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter(::std::vec::IntoIter<($crate::Node, $crate::Weight)>);

        impl ::core::iter::Iterator for $quadrature_rule_into_iter {
            /// Element `.0` is the node and element `.1` the corresponding weight.
            type Item = ($crate::Node, $crate::Weight);
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(
                &self,
            ) -> (
                ::core::primitive::usize,
                ::core::option::Option<::core::primitive::usize>,
            ) {
                self.0.size_hint()
            }
        }

        impl ::core::iter::DoubleEndedIterator for $quadrature_rule_into_iter {
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl ::core::iter::ExactSizeIterator for $quadrature_rule_into_iter {}
        impl ::core::iter::FusedIterator for $quadrature_rule_into_iter {}

        impl $quadrature_rule_into_iter {
            #[inline]
            pub(super) fn new(
                node_weight_pairs: ::std::vec::IntoIter<($crate::Node, $crate::Weight)>,
            ) -> Self {
                Self(node_weight_pairs)
            }

            /// Views the underlying data as a subslice of the original data.
            ///
            /// See [`core::slice::Iter::as_slice`] for more information.
            #[inline]
            pub fn as_slice(&self) -> &[($crate::Node, $crate::Weight)] {
                self.0.as_slice()
            }
        }

        impl<'a> ::core::convert::AsRef<[($crate::Node, $crate::Weight)]>
            for $quadrature_rule_into_iter
        {
            #[inline]
            fn as_ref(&self) -> &[($crate::Node, $crate::Weight)] {
                self.0.as_ref()
            }
        }

        impl ::core::iter::IntoIterator for $quadrature_rule {
            type IntoIter = $quadrature_rule_into_iter;
            type Item = ($crate::Node, $crate::Weight);
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_into_iter::new(self.node_weight_pairs.into_iter())
            }
        }

        // endregion: QuadratureRuleIntoIter
    };
}
