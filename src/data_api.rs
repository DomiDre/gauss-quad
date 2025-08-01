//! The macros in this module define the common API for accessing the data that underlies the quadrature rules.
//! The [`__impl_node_weight_rule!`] macro implements the API for a struct with both nodes and weights.
//! It should be called in the module that defines the quadrature rule struct.
//! [`__impl_node_rule!`] does the same thing as the previous macro
//! but for a struct with only nodes and no weights.

// The code in the macros uses fully qualified paths for every type, so it is quite verbose.
// That is, instead of `usize` it uses `::core::primitive::usize` and so on. This makes it so that
// the caller of the macro doesn't have to import anything into the module in order for the macro to compile,
// and makes it compile even if the user has made custom types whose names shadow types used by the macro.

use core::{fmt, num::ParseFloatError, str::FromStr};

/// A node in a quadrature rule.
pub type Node = f64;
/// A weight in a quadrature rule.
pub type Weight = f64;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A wrapper around a [`f64`] value that ensures the value is greater than -1.0.
pub struct FiniteAboveNegOneF64(f64);

impl fmt::Display for FiniteAboveNegOneF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::LowerExp for FiniteAboveNegOneF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:e}", self.0)
    }
}

impl fmt::UpperExp for FiniteAboveNegOneF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:E}", self.0)
    }
}

impl FiniteAboveNegOneF64 {
    /// Creates a new `FiniteAboveNegOneF64` if the value is greater than -1.0.
    #[inline]
    pub const fn new(value: f64) -> Option<Self> {
        if value > -1.0 && value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Creates a new `FiniteAboveNegOneF64` without checking the value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value is finite and greater than -1.0.
    pub const unsafe fn new_unchecked(value: f64) -> Self {
        debug_assert!(value > -1.0, "value must be greater than -1.0");
        Self(value)
    }

    /// Returns the inner `f64` value of the `FiniteAboveNegOneF64`.
    #[inline]
    pub const fn get(&self) -> f64 {
        self.0
    }

    #[inline]
    pub fn checked_add(self, rhs: f64) -> Option<Self> {
        Self::new(self.0 + rhs)
    }

    #[inline]
    pub fn checked_sub(self, rhs: f64) -> Option<Self> {
        Self::new(self.0 - rhs)
    }

    #[inline]
    pub fn checked_mul(self, rhs: f64) -> Option<Self> {
        Self::new(self.0 * rhs)
    }

    #[inline]
    pub fn checked_div(self, rhs: f64) -> Option<Self> {
        Self::new(self.0 / rhs)
    }

    #[inline]
    pub fn checked_powi(self, exp: i32) -> Option<Self> {
        Self::new(self.0.powi(exp))
    }

    #[inline]
    pub fn checked_powf(self, exp: f64) -> Option<Self> {
        Self::new(self.0.powf(exp))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// The error that that is returned when trying to convert a [`f64`] value that is less than or equal to -1.0
/// into an [`FiniteAboveNegOneF64`] with the [`TryFrom`] trait.
pub struct InfNegOneOrLessError;

impl fmt::Display for InfNegOneOrLessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "attempted to convert a value that is infinite, NAN, or less than or equal to -1.0 to an `FiniteAboveNegOneF64`"
        )
    }
}

impl core::error::Error for InfNegOneOrLessError {}

impl TryFrom<f64> for FiniteAboveNegOneF64 {
    type Error = InfNegOneOrLessError;

    /// Tries to convert a `f64` into an `FiniteAboveNegOneF64`.
    ///
    /// Returns an error if the value is less than or equal to -1.0.
    #[inline]
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        FiniteAboveNegOneF64::new(value).ok_or(InfNegOneOrLessError)
    }
}

impl From<FiniteAboveNegOneF64> for f64 {
    #[inline]
    fn from(value: FiniteAboveNegOneF64) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An error that can occur when parsing a `&str` into an [`FiniteAboveNegOneF64`].
pub enum ParseFiniteAboveNegOneF64Error {
    ParseError(ParseFloatError),
    TooSmall(InfNegOneOrLessError),
}

impl fmt::Display for ParseFiniteAboveNegOneF64Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseFiniteAboveNegOneF64Error::ParseError(e) => write!(f, "{e}"),
            ParseFiniteAboveNegOneF64Error::TooSmall(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for ParseFiniteAboveNegOneF64Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            ParseFiniteAboveNegOneF64Error::ParseError(e) => Some(e),
            ParseFiniteAboveNegOneF64Error::TooSmall(e) => Some(e),
        }
    }
}

impl FromStr for FiniteAboveNegOneF64 {
    type Err = ParseFiniteAboveNegOneF64Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.parse::<f64>() {
            Ok(value) => value
                .try_into()
                .map_err(ParseFiniteAboveNegOneF64Error::TooSmall),
            Err(e) => Err(ParseFiniteAboveNegOneF64Error::ParseError(e)),
        }
    }
}

/// This macro implements the data access API for the given quadrature rule struct that contains
/// a field named `node_weight_pairs` of the type `Box<[(Node, Weight)]>`.
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

            /// Converts the quadrature rule into a boxed slice of node-weight pairs.
            ///
            /// This function just returns the underlying data without any computation or cloning.
            #[inline]
            #[must_use = "`self` will be dropped if the result is not used"]
            pub fn into_node_weight_pairs(self) -> ::std::boxed::Box<[($crate::Node, $crate::Weight)]> {
                self.node_weight_pairs
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
        impl$(<$a>)? ::core::iter::Iterator for $iterator$(<$a>)? {
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
/// It takes in the name of the a rule struct that contans a field with the name `degree`
/// of the type `NonZeroU32`. As well as the names it should give the iterator over its
/// nodes and the iterator returned by the [`IntoIterator`] trait.
#[macro_export]
#[doc(hidden)]
macro_rules! __impl_node_rule {
    ($quadrature_rule:ident, $quadrature_rule_iter:ident) => {
        // Lets the user do
        // for node in QuadratureRule::new(...) {
        //    ...
        // }
        impl ::core::iter::IntoIterator for $quadrature_rule {
            type Item = $crate::Node;
            type IntoIter = $quadrature_rule_iter;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        // Lets the user do
        // let rule = QuadratureRule::new(...);
        // for &node in &rule {
        //     ...
        // }
        // rule.integrate(...) // <--- still available
        impl<'a> ::core::iter::IntoIterator for &'a $quadrature_rule {
            type IntoIter = $quadrature_rule_iter;
            type Item = $crate::Node;
            #[inline]
            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl $quadrature_rule {
            /// Returns an iterator over the nodes of the rule.
            #[inline]
            pub fn iter(&self) -> $quadrature_rule_iter {
                $quadrature_rule_iter::new((0..self.degree.get()).map(|d| d as $crate::Node))
            }

            /// Returns the degree of the rule.
            #[inline]
            pub fn degree(&self) -> ::core::num::NonZeroU32 {
                self.degree
            }
        }

        // region: QuadratureRuleIter

        /// An iterator of the nodes of the rule.
        #[derive(Debug, Clone)]
        #[must_use = "iterators are lazy and do nothing unless consumed"]
        pub struct $quadrature_rule_iter(
            ::core::iter::Map<core::ops::Range<u32>, fn(u32) -> $crate::Node>,
        );

        impl $quadrature_rule_iter {
            #[inline]
            const fn new(
                iter: ::core::iter::Map<core::ops::Range<u32>, fn(u32) -> $crate::Node>,
            ) -> Self {
                Self(iter)
            }
        }

        $crate::__impl_slice_iterator_newtype_traits! {$quadrature_rule_iter, $crate::Node}

        // endregion: QuadratureRuleIter
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
