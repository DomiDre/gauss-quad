//! The macros in this module define the common API for accessing the data that underlies the quadrature rules.
//! The [`impl_node_weight_rule!`] macro implements the API for a for a struct with both nodes and weights.
//! It should be called in the module that defines the quadrature rule struct.
//! The [`impl_node_weight_rule_iterators!`] macro defines the iterators that somce of the functions return.
//! It should be called somewhere it makes sense for the iterators to be defined, e.g. a sub-module.
//! The [`impl_node_rule!`] and [`impl_node_rule_iterators!`] do the same thing as the previous
//! macros but for a struct with only nodes and no weights.

// The code in the macros uses fully qualified paths for every type, so it is quite verbose.
// That is, instead of `usize` it uses `::core::primitive::usize` and so on. This makes it so that
// the caller of the macro doesn't have to import anything into the module in order for the macro to compile.

/// A node in a quadrature rule.
pub type Node = f64;
/// A weight in a quadrature rule.
pub type Weight = f64;

/// This macro implements the data access API for the given quadrature rule struct that contains
/// a field named `node_weight_pairs` of the type `Vec<Node, Weight>`.
/// It takes in the name of the quadrature rule struct as well as the names if should give the iterators
/// over its nodes, weights, and both, as well as the iterator returned by the [`IntoIterator`] trait.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_node_weight_rule {
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
        // The name of the iterator returned by the by the IntoIterator trait.
        $quadrature_rule_into_iter:ident
    ) => {
        // Implements functions for accessing the underlying data of the quadrature rule struct
        // in a way the adheres to the API guidelines: <https://rust-lang.github.io/api-guidelines/naming.html>.
        // The functions in these impl blocks all have an #[inline] directive because they are trivial.

        // Lets the user do
        // for (node, weight) in QuadratuleRule::new(...) {
        //     ...
        // }
        impl ::core::iter::IntoIterator for $quadrature_rule {
            type IntoIter = $quadrature_rule_into_iter;
            type Item = ($crate::Node, $crate::Weight);
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_into_iter::new(self.node_weight_pairs.into_iter())
            }
        }

        // Lets the user do
        // let rule = QuadratureRule::new(...);
        // for &(node, weight) in &rule {
        //     ...
        // }
        // rule.integrate(...) // <-- still available
        impl<'a> ::core::iter::IntoIterator for &'a $quadrature_rule {
            type IntoIter = $quadrature_rule_iter<'a>;
            type Item = &'a ($crate::Node, $crate::Weight);
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_iter::new(self.node_weight_pairs.iter())
            }
        }

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

            /// Returns an iterator over the node-weight pairs of the quadrature rule.
            #[inline]
            pub fn iter(&self) -> $quadrature_rule_iter<'_> {
                $quadrature_rule_iter::new(self.node_weight_pairs.iter())
            }

            /// Returns a slice of all the node-weight pairs of the quadrature rule.
            #[inline]
            pub fn as_node_weight_pairs(&self) -> &[($crate::Node, $crate::Weight)] {
                &self.node_weight_pairs
            }

            /// Converts the quadrature rule into a vector of node-weight pairs.
            ///
            /// This function just returns the underlying vector without any computation or cloning.
            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_node_weight_pairs(self) -> ::std::vec::Vec<($crate::Node, $crate::Weight)> {
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

/// Implements the Iterator, DoubleEndedIterator, ExactSizeIterator and FusedIterator traits for a struct
/// that wraps an iterator that has those traits. Takes in the name of the struct and optionally its lifetime
/// as well as the type returned by the iterator.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_slice_iterator_newtype_traits {
    ($iterator:ident$(<$a:lifetime>)?, $item:ty) => {
        impl$(<$a>)? ::core::iter::Iterator for $iterator<$($a)?> {
            type Item = $item;
            #[inline]
            fn next(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next()
            }

            #[inline]
            fn size_hint(&self) -> (::core::primitive::usize, ::core::option::Option<::core::primitive::usize>) {
                self.0.size_hint()
            }

            // These methods by default call the `next` method a lot to access data.
            // This isn't needed in our case, since the data underlying the iterator is
            // a slice, which has O(1) access to any element. As a result we reimplement them
            // and just delegate to the inbuilt method.

            #[inline]
            fn nth(&mut self, index: ::core::primitive::usize) -> ::core::option::Option<Self::Item> {
                self.0.nth(index)
            }

            #[inline]
            fn count(self) -> ::core::primitive::usize {
                self.0.count()
            }

            #[inline]
            fn last(self) -> ::core::option::Option<Self::Item> {
                self.0.last()
            }
        }

        impl$(<$a>)? ::core::iter::DoubleEndedIterator for $iterator$(<$a>)? {
            #[inline]
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl$(<$a>)? ::core::iter::ExactSizeIterator for $iterator$(<$a>)? {
            #[inline]
            fn len(&self) -> ::core::primitive::usize {
                self.0.len()
            }
        }
        impl$(<$a>)? ::core::iter::FusedIterator for $iterator$(<$a>)? {}
    };
}

/// This macro defines the iterators used by the functions defined in the [`impl_node_weight_rule!`] macro.
/// It takes in the names of the same structs as that macro,
/// plus the name it should give the iterator that is returned by the [`IntoIterator`] implementation.
/// These iterators can only be created in the module where the macro is called.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_node_weight_rule_iterators {
    (
        $quadrature_rule_nodes:ident,
        $quadrature_rule_weights:ident,
        $quadrature_rule_iter:ident,
        $quadrature_rule_into_iter:ident
    ) => {
        // region: QuadratureRuleNodes

        /// An iterator over the nodes of the quadrature rule.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_nodes<'a>(
            // This horrible type is just the fully qualified path of the type returned
            // by `slice.iter().map(|(x, _)| x)`.
            ::std::iter::Map<
                ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Node,
            >,
        );

        impl<'a> $quadrature_rule_nodes<'a> {
            #[inline]
            const fn new(
                iter_map: ::std::iter::Map<
                    ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                    fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Node,
                >,
            ) -> Self {
                Self(iter_map)
            }
        }

        $crate::impl_slice_iterator_newtype_traits!{$quadrature_rule_nodes<'a>, &'a $crate::Node}

        // endregion: QuadratureRuleNodes

        // region: QuadratureRuleWeights

        /// An iterator over the weights of the quadrature rule.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_weights<'a>(
            // Same as the previous horrible type, but maps out the weight instead of the node.
            ::std::iter::Map<
                ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Weight,
            >,
        );

        impl<'a> $quadrature_rule_weights<'a> {
            #[inline]
            const fn new(
                iter_map: ::std::iter::Map<
                    ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
                    fn(&'a ($crate::Node, $crate::Weight)) -> &'a $crate::Weight,
                >,
            ) -> Self {
                Self(iter_map)
            }
        }

        $crate::impl_slice_iterator_newtype_traits!{$quadrature_rule_weights<'a>, &'a $crate::Weight}

        // endregion: QuadratureRuleWeights

        // region: QuadratureRuleIter

        /// An iterator over the node-weight-pairs of the quadrature rule.
        ///
        /// Created by the `iter` function on the quadrature rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a>(
            ::core::slice::Iter<'a, ($crate::Node, $crate::Weight)>,
        );

        impl<'a> $quadrature_rule_iter<'a> {
            #[inline]
            const fn new(
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

        impl<'a> ::core::convert::AsRef<[($crate::Node, $crate::Weight)]>
            for $quadrature_rule_iter<'a>
        {
            #[inline]
            fn as_ref(&self) -> &[($crate::Node, $crate::Weight)] {
                self.0.as_ref()
            }
        }

        $crate::impl_slice_iterator_newtype_traits!{$quadrature_rule_iter<'a>, &'a ($crate::Node, $crate::Weight)}

        // endregion: QuadratureRuleIter

        // region: QuadratureRuleIntoIter

        /// An owning iterator over the node-weight-pairs of the quadrature rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the quadrature rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter(::std::vec::IntoIter<($crate::Node, $crate::Weight)>);

        impl $quadrature_rule_into_iter {
            #[inline]
            const fn new(
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

        impl ::core::convert::AsRef<[($crate::Node, $crate::Weight)]>
            for $quadrature_rule_into_iter
        {
            #[inline]
            fn as_ref(&self) -> &[($crate::Node, $crate::Weight)] {
                self.0.as_ref()
            }
        }

        $crate::impl_slice_iterator_newtype_traits!{$quadrature_rule_into_iter, ($crate::Node, $crate::Weight)}

        // endregion: QuadratureRuleIntoIter
    };
}

/// This macro implements the data access API for rules that have only nodes and no weights.
/// It takes in the name of the a rule struct that contans a field with the name `nodes`
/// of the type `Vec<Node>`. As well as the names it should give the iterator over its
/// nodes and the iterator returned by the [`IntoIterator`] trait.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_node_rule {
    ($quadrature_rule:ident, $quadrature_rule_iter:ident, $quadrature_rule_into_iter:ident) => {
        // Lets the user do
        // for node in QuadratureRule::new(...) {
        //    ...
        // }
        impl ::core::iter::IntoIterator for $quadrature_rule {
            type Item = $crate::Node;
            type IntoIter = $quadrature_rule_into_iter;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_into_iter::new(self.nodes.into_iter())
            }
        }

        // Lets the user do
        // let rule = QuadratureRule::new(...);
        // for &node in &rule {
        //     ...
        // }
        // rule.integrate(...) // <--- still available
        impl<'a> ::core::iter::IntoIterator for &'a $quadrature_rule {
            type IntoIter = $quadrature_rule_iter<'a>;
            type Item = &'a $crate::Node;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                $quadrature_rule_iter::new(self.nodes.iter())
            }
        }

        impl $quadrature_rule {
            /// Returns an iterator over the nodes of the rule.
            #[inline]
            pub fn iter(&self) -> $quadrature_rule_iter<'_> {
                $quadrature_rule_iter::new(self.nodes.iter())
            }

            /// Returns a slice of all the nodes of the rule.
            #[inline]
            pub fn as_nodes(&self) -> &[$crate::Node] {
                &self.nodes
            }

            /// Converts the rule into a vector of nodes.
            ///
            /// This function just returns the underlying data without any computation or cloning.
            #[inline]
            pub fn into_nodes(self) -> Vec<$crate::Node> {
                self.nodes
            }

            /// Returns the degree of the rule.
            #[inline]
            pub fn degree(&self) -> usize {
                self.nodes.len()
            }
        }
    };
}

/// This macro defines the iterators used by the functions defined by the [`impl_node_rule`] macro.
/// It takes in the names of the same structs as that macro,
/// plus the name it should give the iterator that is returned by the [`IntoIterator`] implementation.
/// These iterators can only be created in the module where the macro is called.
#[macro_export]
#[doc(hidden)]
macro_rules! impl_node_rule_iterators {
    ($quadrature_rule_iter:ident, $quadrature_rule_into_iter:ident) => {
        // region: QuadratureRuleIter

        /// An iterator of the nodes of the rule.
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter<'a>(::core::slice::Iter<'a, $crate::Node>);

        impl<'a> $quadrature_rule_iter<'a> {
            #[inline]
            const fn new(iter: ::core::slice::Iter<'a, $crate::Node>) -> Self {
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

        impl<'a> ::core::convert::AsRef<[$crate::Node]> for $quadrature_rule_iter<'a> {
            #[inline]
            fn as_ref(&self) -> &[$crate::Node] {
                self.0.as_ref()
            }
        }

        $crate::impl_slice_iterator_newtype_traits! {$quadrature_rule_iter<'a>, &'a $crate::Node}

        // endregion: QuadratureRuleIter

        // region: QuadratureRuleIntoIter

        /// An owning iterator over the nodes of the rule.
        ///
        /// Created by the [`IntoIterator`] trait implementation of the rule struct.
        #[derive(::core::fmt::Debug, ::core::clone::Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_into_iter(::std::vec::IntoIter<$crate::Node>);

        impl $quadrature_rule_into_iter {
            #[inline]
            const fn new(iter: ::std::vec::IntoIter<$crate::Node>) -> Self {
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

        impl ::core::convert::AsRef<[$crate::Node]> for $quadrature_rule_into_iter {
            #[inline]
            fn as_ref(&self) -> &[$crate::Node] {
                self.0.as_ref()
            }
        }

        $crate::impl_slice_iterator_newtype_traits! {$quadrature_rule_into_iter, $crate::Node}

        // endregion: QuadratureRuleIntoIter
    };
}
