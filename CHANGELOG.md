# Changelog

This document contains all changes to the crate since version 0.1.8.

## 0.2.3

- Make the `QuadratureRule::integrate` functions take a `FnMut` instead of a `Fn`.
- Documentation improvements.

## 0.2.2

- Add Gauss-Chebyshev quadrature of the first and second kinds.

## 0.2.1

- Add the `rayon` feature that enables certain calculations to be done in parallel.
- Add the function `par_integrate` to every quadrature rule struct which can be used when the `rayon` feature is enabled to perform integration in parallel.
- Add the function `par_new` to `GaussLegendre` to initialize it in parallel. This function is also behind the `rayon` feature.

## 0.2.0

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

## 0.1.9

- Update `nalgebra` dependency to 0.33.0.
