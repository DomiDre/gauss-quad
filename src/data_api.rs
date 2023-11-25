//! This module contains the traits [`NodeRule`] and [`NodeWeightRule`] as well as
//! macros that can be used to implement them.
//! The traits define the common API for accessing the data that underlies the quadrature rules.
//! The [`impl_node_weight_rule_trait!`] macro implements the [`NodeWeightRule`] trait for a struct. It takes in the name of
//! a quadrature rule struct that has a field named `node_weight_pairs`
//! of the type `Vec<(Node, Weight)>`. It also needs the names that it should give the various structs
//! that iterate over that data. It should be called in the module that defines the quadrature rule struct.
//! The [`impl_node_weight_rule_iterators!`] macro defines the iterators that the trait returns. It should be called somewhere it makes sense
//! for the iterators to be defined, e.g. a sub-module.
//! The [`impl_node_rule_trait!`] and [`impl_node_rule_iterators!`] do the same thing as the previous macros but for the [`NodeRule`] trait.

// The code in the macros uses fully qualified paths for every type, so it is quite verbose.
// That is, instead of `usize` it uses `::core::primitive::usize` and so on. This makes it so that
// the caller of the macro doesn't have to import anything into the module in order for the macro to compile.

/// A node in a quadrature rule.
pub type Node = f64;
/// A weight in a quadrature rule.
pub type Weight = f64;

/// This trait defines the API for reading the underlying data in quadrature rules that have
/// both nodes and weights.
pub trait NodeWeightRule
where
    Self: IntoIterator<Item = (Node, Weight)>,
{
    /// The type of the nodes.
    type Node;
    /// The type of the weights.
    type Weight;
    /// An iterator over node-weight-pairs of the quadrature rule.
    type Iter<'a>: Iterator<Item = &'a (Self::Node, Self::Weight)>
    where
        Self: 'a;
    /// An iterator over the nodes of the quadrature rule.
    type Nodes<'a>: Iterator<Item = &'a Self::Node>
    where
        Self: 'a;
    /// An iterator over the weights of the quadrature rule.
    type Weights<'a>: Iterator<Item = &'a Self::Weight>
    where
        Self: 'a;
    /// Returns an iterator over the node-weight-pairs of the quadrature rule.
    fn iter(&self) -> Self::Iter<'_>;
    /// Returns an iterator over the nodes of the quadrature rule.
    fn nodes(&self) -> Self::Nodes<'_>;
    /// Returns an iterator over the weights of the quadrature rule.
    fn weights(&self) -> Self::Weights<'_>;
    /// Returns a slice of the node-weight-pairs of the quadrature rule.
    fn as_node_weight_pairs(&self) -> &[(Self::Node, Self::Weight)];
    /// Converts the quadrature rule into a vector of node-weight-pairs.
    ///
    /// This function just returns the underlying data and does no
    /// computation or cloning.
    fn into_node_weight_pairs(self) -> Vec<(Self::Node, Self::Weight)>;
    /// Returns the degree of the quadrature rule.
    fn degree(&self) -> usize;
}

/// This trait defines the API for accessing the underlying nodes
/// in rules that do not have weights.
pub trait NodeRule
where
    Self: IntoIterator<Item = Node>,
{
    /// The type of the nodes.
    type Node;
    /// An iterator over the nodes.
    type Iter<'a>: Iterator<Item = &'a Self::Node>
    where
        Self: 'a;
    /// Returns an iterator over the nodes.
    fn iter(&self) -> Self::Iter<'_>;
    /// Returns a slice of the nodes.
    fn as_nodes(&self) -> &[Self::Node];
    /// Converts the rule into a vector of nodes.
    ///
    /// This function just returns the underlying data and does no
    /// computation or cloning.
    fn into_nodes(self) -> Vec<Self::Node>;
    /// Returns the number of nodes of the rule.
    fn degree(&self) -> usize;
}

/// This macro implements the data access API for the given quadrature rule struct.
/// It takes in the name of the quadrature rule struct as well as the names of the iterators
/// over its nodes, weights, and both.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_node_weight_rule_trait {
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
        impl $crate::NodeWeightRule for $quadrature_rule {
            type Node = f64;
            type Weight = f64;
            type Nodes<'a> = $quadrature_rule_nodes<'a>;
            type Weights<'a> = $quadrature_rule_weights<'a>;
            type Iter<'a> = $quadrature_rule_iter<'a>;
            #[inline]
            fn nodes(&self) -> Self::Nodes<'_> {
                $quadrature_rule_nodes::new(self.node_weight_pairs.iter().map(|p| &p.0))
            }

            #[inline]
            fn weights(&self) -> Self::Weights<'_> {
                $quadrature_rule_weights::new(self.node_weight_pairs.iter().map(|p| &p.1))
            }

            #[inline]
            fn iter(&self) -> Self::Iter<'_> {
                $quadrature_rule_iter::new(self.node_weight_pairs.iter())
            }

            #[inline]
            fn as_node_weight_pairs(&self) -> &[(Self::Node, Self::Weight)] {
                &self.node_weight_pairs
            }

            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            fn into_node_weight_pairs(self) -> ::std::vec::Vec<(Self::Node, Self::Weight)> {
                self.node_weight_pairs
            }

            #[inline]
            fn degree(&self) -> ::core::primitive::usize {
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
macro_rules! impl_node_weight_rule_iterators {
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
            /// See [`std::vec::IntoIter::as_slice`] for more information.
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

/// This macro implements the functions of the [`NodeRule`] trait for
/// the given quadrature struct that contans a field named `nodes`
/// of the type `Vec<Node>`.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_node_rule_trait {
    ($quadrature_rule:ident, $quadrature_rule_iter:ident) => {
        impl $crate::NodeRule for $quadrature_rule {
            type Node = $crate::Node;
            type Iter<'a> = $quadrature_rule_iter<'a>;
            #[inline]
            fn iter(&self) -> Self::Iter<'_> {
                $quadrature_rule_iter::new(self.nodes.iter())
            }

            #[inline]
            fn as_nodes(&self) -> &[Self::Node] {
                &self.nodes
            }
            #[inline]
            fn into_nodes(self) -> Vec<Self::Node> {
                self.nodes
            }

            #[inline]
            fn degree(&self) -> usize {
                self.nodes.len()
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_node_rule_iterators {
    ($quadrature_rule:ident, $quadrature_rule_iter:ident, $quadrature_rule_into_iter:ident) => {
        /// An iterator of the nodes of the rule.
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a>(::core::slice::Iter<'a, $crate::Node>);

        impl<'a> $quadrature_rule_iter<'a> {
            #[inline]
            pub(super) fn new(iter: ::core::slice::Iter<'a, $crate::Node>) -> Self {
                Self(iter)
            }

            /// Views the underlying data as a subslice of the original data.
            ///
            /// See [`core::slice::Iter::as_slice`] for more information.
            #[inline]
            pub fn as_slice(&self) -> &'a [$crate::Node] {
                self.0.as_slice()
            }
        }

        impl<'a> AsRef<[$crate::Node]> for $quadrature_rule_iter<'a> {
            #[inline]
            fn as_ref(&self) -> &[$crate::Node] {
                self.0.as_ref()
            }
        }

        impl<'a> Iterator for $quadrature_rule_iter<'a> {
            type Item = &'a $crate::Node;
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(&self) -> (::core::primitive::usize, Option<::core::primitive::usize>) {
                self.0.size_hint()
            }
        }

        impl<'a> ::core::iter::DoubleEndedIterator for $quadrature_rule_iter<'a> {
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<'a> ::core::iter::ExactSizeIterator for $quadrature_rule_iter<'a> {}
        impl<'a> ::core::iter::FusedIterator for $quadrature_rule_iter<'a> {}

        // region: QuadratureRuleIntoIter

        /// An owning iterator over the nodes of the rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter(::std::vec::IntoIter<$crate::Node>);

        impl $quadrature_rule_into_iter {
            #[inline]
            pub(super) fn new(iter: ::std::vec::IntoIter<$crate::Node>) -> Self {
                Self(iter)
            }

            /// Views the underlying data as a subslice of the original data.
            ///
            /// See [`std::vec::IntoIter::as_slice`] for more information.
            #[inline]
            pub fn as_slice(&self) -> &[$crate::Node] {
                self.0.as_slice()
            }
        }

        impl ::core::iter::Iterator for $quadrature_rule_into_iter {
            type Item = $crate::Node;
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

        impl<'a> ::core::convert::AsRef<[$crate::Node]> for $quadrature_rule_into_iter {
            #[inline]
            fn as_ref(&self) -> &[$crate::Node] {
                self.0.as_ref()
            }
        }

        impl ::core::iter::IntoIterator for $quadrature_rule {
            type Item = $crate::Node;
            type IntoIter = $quadrature_rule_into_iter;
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_into_iter::new(self.nodes.into_iter())
            }
        }

        // endregion: QuadratureRuleIntoIter
    };
}
