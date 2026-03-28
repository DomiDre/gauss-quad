# Changelog

This document contains all changes to the crate since version 0.1.8.

## [0.3.1] - 2026-03-28

- Internal code structure improvements.
- Add the `zerocopy` feature that derives the `KnownLayout` trait from the [`zerocopy`](https://crates.io/crates/zerocopy) crate for the quadrature rule structs.
- Add the `rkyv` feature that derives the (de)serialization traits from the [`rkyv`](https://crates.io/crates/rkyv) crate for the quadrature rule structs.
- Update transitive dependencies.

## [0.3.0] - 2026-03-15

### Breaking changes

- Made all quadrature rules store their nodes and weights sorted in ascending order by node (according to the `f64::total_cmp` function).
 This means that all functions that return some view of the nodes (and weights) now return them in this sorted order.
 This affects the Gauss-Legendre, Gauss-Hermite and Gauss-Chebyshev rules.
 The affected functions are `QuadratureRule::iter()`, `QuadratureRule::into_iter()`, `QuadratureRule::nodes()`,
 `QuadratureRule::weights()`, `QuadratureRule::as_node_weight_pairs()` and `QuadratureRule::into_node_weight_pairs()`.
- Made the `Simpson` and `Midpoint` rules not allocate any memory.
 This removes all the functions on those types that access some view of the nodes.
- Made the different rules capable of working also at a degree of 1.
 As a result their constructors take a `NonZeroUsize` for a degree.
- Made the constructors take types that ensure that the inputs uphold the correct invariants.
 As a result all constructors just return the rule itself, no `Result` or `Option`.
- Made the quadrature rule structs that store a `Vec` of nodes and weights instead store a boxed slice.
 This makes them take up less space on the stack.
 This affects the `QuadratureRule::into_node_weight_pairs()` functions.
 Call `.into_vec()` on the boxed slice to get it as a `Vec`.

### Other changes

- Made the constructors of `Simpson` and `Midpoint` into `const` functions.
- Added the trapezoidal rule.
- Update dependencies.
- Add dates to the releases in this log.
- Document the MSRV of the crate.
- Remove all usage of `unsafe`, add `#![forbid(unsafe_code)]` to the crate root.
- Make the crate `no_std` compatible. 
 The standard library can optionally be enabled through the `std` feature for a potential performance gain.

## [0.2.4] - 2025-08-23

- Upgrade nalgebra & rayon packages. 
- Add a CI job that mimics the doc generation on docs.rs. 
- Run the test job on multiple operating systems. 
- Add license info to readme. 
- Add a badge that takes you to the repo when clicked. 
- Update transitive dependencies. 


## [0.2.3] - 2025-04-20

- Make the `QuadratureRule::integrate` functions take a `FnMut` instead of a `Fn`.
- Documentation improvements.
- CI improvements.
- Updated dependencies.
- Implemented `Iterator::last` by calling `DoubleEndedIterator::next_back` where applicable.

## [0.2.2] - 2024-12-29

- Add Gauss-Chebyshev quadrature of the first and second kinds.

## [0.2.1] - 2024-08-24

- Add the `rayon` feature that enables certain calculations to be done in parallel.
- Add the function `par_integrate` to every quadrature rule struct which can be used when the `rayon` feature is enabled to perform integration in parallel.
- Add the function `par_new` to `GaussLegendre` to initialize it in parallel. This function is also behind the `rayon` feature.

## [0.2.0] - 2024-07-29

This update is mostly about changing the API to adhere to the [Rust API guidelines](https://rust-lang.github.io/api-guidelines/about.html).

### Breaking changes

- Changed the name of all constructors from `init` to `new`.  
- All constructors now return a `Result` that contains an error when the input is invalid, instead of panicking. Simply `unwrap()` it to recover the old behaviour.  
- The fields of the quadrature rule structs are now private to uphold the invariants needed for integration.  
- A set of functions have been implemented that access the node and weight data of the quadrature rule structs in various ways.  
- The `nodes_and_weights` functions have been removed. To achieve the same effect you can do `QuadratureRule::new(...)?.into_node_weight_pairs()` if you wish to have a `Vec<(f64, f64)>` of nodes and their corresponding weights, or you can do `QuadratureRule::new(...)?.into_iter().unzip()` if you wish to have the nodes and weights separate.
- The crate no longer exports the `DMatrixf64` type alias.
- The crate no longer re-exports the `core::f64::consts::PI` constant.

### Other changes

- Added the `serde` feature which implements the `Serialize` and `Deserialize` traits from [`serde`](https://crates.io/crates/serde) for all quadrature rule structs.
- The quadrature rule structs now store the nodes and weights together in a single allocation. This slightly speeds up integration, and removes one intermediate allocation during creation.
- Fixed a sign error in the documentation for `GaussJacobi`.

## [0.1.9] - 2024-06-30

- Update `nalgebra` dependency to 0.33.0.
