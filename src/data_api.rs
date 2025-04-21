//! The macros in this module define the common API for accessing the data that underlies the quadrature rules.
//! The [`__impl_node_weight_rule!`] macro implements the API for a struct with both nodes and weights.
//! It should be called in the module that defines the quadrature rule struct.
//! [`__impl_node_rule!`] does the same thing as the previous macro
//! but for a struct with only nodes and no weights.

// The code in the macros uses fully qualified paths for every type, so it is quite verbose.
// That is, instead of `usize` it uses `::core::primitive::usize` and so on. This makes it so that
// the caller of the macro doesn't have to import anything into the module in order for the macro to compile,
// and makes it compile even if the user has made custom types whose names shadow types used by the macro.

/// A node in a quadrature rule.
pub type Node = f64;
/// A weight in a quadrature rule.
pub type Weight = f64;

/// This macro implements the data access API for the given quadrature rule struct that contains
/// a field named `node_weight_pairs` of the type `Vec<(Node, Weight)>`.
/// It takes in the name of the quadrature rule struct as well as the names it should give the iterators
/// over its nodes, weights, and both, as well as the iterator returned by the [`IntoIterator`] trait.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_node_weight_rule {
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
                $quadrature_rule_into_iter::new(self.node_weight_pairs.into_vec().into_iter())
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
                self.node_weight_pairs.into_vec()
            }

            /// Returns the degree of the quadrature rule.
            #[inline]
            pub fn degree(&self) -> ::core::primitive::usize {
                self.node_weight_pairs.len()
            }
        }

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

        $crate::__impl_slice_iterator_newtype_traits!{$quadrature_rule_nodes<'a>, &'a $crate::Node}

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

        $crate::__impl_slice_iterator_newtype_traits!{$quadrature_rule_weights<'a>, &'a $crate::Weight}

        // endregion: QuadratureRuleWeights

        // region: QuadratureRuleIter

        /// An iterator over the node-weight pairs of the quadrature rule.
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

        $crate::__impl_slice_iterator_newtype_traits!{$quadrature_rule_iter<'a>, &'a ($crate::Node, $crate::Weight)}

        // endregion: QuadratureRuleIter

        // region: QuadratureRuleIntoIter

        /// An owning iterator over the node-weight pairs of the quadrature rule.
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

        $crate::__impl_slice_iterator_newtype_traits!{$quadrature_rule_into_iter, ($crate::Node, $crate::Weight)}

        // endregion: QuadratureRuleIntoIter
    };
}

/// Implements the [`Iterator`], [`DoubleEndedIterator`], [`ExactSizeIterator`] and [`FusedIterator`](core::iter::FusedIterator) traits for a struct
/// that wraps an iterator that has those traits. Takes in the name of the struct and optionally its lifetime
/// as well as the type returned by the iterator.
#[macro_export]
#[doc(hidden)]
macro_rules! __impl_slice_iterator_newtype_traits {
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
            // a slice, which has O(1) access to any element.
            // As a result we reimplement them and just delegate to the builtin slice iterator methods.

            #[inline]
            fn nth(&mut self, index: ::core::primitive::usize) -> ::core::option::Option<Self::Item> {
                self.0.nth(index)
            }

            #[inline]
            fn count(self) -> ::core::primitive::usize {
                self.0.count()
            }

            #[inline]
            fn last(mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl$(<$a>)? ::core::iter::DoubleEndedIterator for $iterator$(<$a>)? {
            #[inline]
            fn next_back(&mut self) -> ::core::option::Option<Self::Item> {
                self.0.next_back()
            }

            #[inline]
            fn nth_back(&mut self, n: ::core::primitive::usize) -> ::core::option::Option<Self::Item> {
                self.0.nth_back(n)
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

/// This macro implements the data access API for rules that have only nodes and no weights.
/// It takes in the name of the a rule struct that contans a field with the name `nodes`
/// of the type `Vec<Node>`. As well as the names it should give the iterator over its
/// nodes and the iterator returned by the [`IntoIterator`] trait.
#[macro_export]
#[doc(hidden)]
macro_rules! __impl_node_rule {
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
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_nodes(self) -> Vec<$crate::Node> {
                self.nodes
            }

            /// Returns the degree of the rule.
            #[inline]
            pub fn degree(&self) -> usize {
                self.nodes.len()
            }
        }

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

        $crate::__impl_slice_iterator_newtype_traits! {$quadrature_rule_iter<'a>, &'a $crate::Node}

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

        $crate::__impl_slice_iterator_newtype_traits! {$quadrature_rule_into_iter, $crate::Node}

        // endregion: QuadratureRuleIntoIter
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt;
    use std::backtrace::Backtrace;

    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct MockQuadrature {
        node_weight_pairs: Box<[(Node, Weight)]>,
    }

    #[derive(Debug)]
    pub struct MockQuadratureError(Backtrace);

    impl MockQuadratureError {
        pub fn backtrace(&self) -> &Backtrace {
            &self.0
        }
    }

    impl fmt::Display for MockQuadratureError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "wrong! bad! T_T")
        }
    }

    impl std::error::Error for MockQuadratureError {}

    impl MockQuadrature {
        pub fn new(deg: usize) -> Result<Self, MockQuadratureError> {
            if deg < 1 {
                return Err(MockQuadratureError(Backtrace::capture()));
            }

            Ok(Self {
                node_weight_pairs: (0..deg).map(|d| (d as f64, 1.0)).collect(),
            })
        }

        pub fn integrate<F>(&self, a: f64, b: f64, integrand: F) -> f64
        where
            F: Fn(f64) -> f64,
        {
            let rect_width = (b - a) / self.node_weight_pairs.len() as f64;
            let result: f64 = self
                .node_weight_pairs
                .iter()
                .map(|(x_val, w_val)| integrand(a + rect_width * (0.5 + x_val)) * w_val)
                .sum();
            result * rect_width
        }
    }

    __impl_node_weight_rule! {MockQuadrature, MockQuadratureNodes, MockQuadratureWeights, MockQuadratureIter, MockQuadratureIntoIter}

    #[test]
    fn test_macro_implementation() {
        let quad = MockQuadrature::new(5).unwrap();
        assert_eq!(quad.integrate(0.0, 1.0, |x| x), 0.5);

        // Test iterator implementations
        assert_eq!(quad.nodes().count(), 5);
        assert_eq!(quad.weights().count(), 5);
        assert_eq!(quad.iter().count(), 5);
        assert_eq!(quad.as_node_weight_pairs().len(), 5);

        // Test IntoIterator implementation
        let vec: Vec<(Node, Weight)> = quad.clone().into_iter().collect();
        assert_eq!(vec.len(), 5);

        // Test into_node_weight_pairs
        let pairs = quad.clone().into_node_weight_pairs();
        assert_eq!(pairs.len(), 5);

        // Test degree
        assert_eq!(quad.degree(), 5);

        // test iter functions
        let mut quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.next().unwrap().0, 0.0);
        assert_eq!(quad_iter.next().unwrap().0, 1.0);

        let quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.size_hint(), (5, Some(5)));

        let mut quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.nth(2).unwrap().0, 2.0);

        let quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.count(), 5);

        let quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.last().unwrap().0, 4.0);

        let mut quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.next_back().unwrap().0, 4.0);

        let mut quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.nth_back(1).unwrap().0, 3.0);

        let quad_iter = (&quad).into_iter();
        assert_eq!(quad_iter.len(), 5);

        // test slice
        let quad_slice = (&quad).into_iter().as_slice();
        assert_eq!(quad_slice.len(), 5);
        assert_eq!(quad_slice[2].0, 2.0);

        // test as_ref
        let quad_iter = (&quad).into_iter();
        let quad_ref = quad_iter.as_ref();
        assert_eq!(quad_ref.len(), 5);
        assert_eq!(quad_ref[2].0, 2.0);
    }
}
