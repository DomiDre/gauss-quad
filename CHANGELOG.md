# 0.2.0

This update is mostly about changing the API to adhere to the [Rust API guidelines](https://rust-lang.github.io/api-guidelines/about.html).

## Breaking changes

 - Changed the name of all constructors from `init` to `new`.
 - The fields of the structs are now private.
 - A set of functions have been implemented that access the node and weight data of the quadrature rule structs in various ways.
 - The crate no longer exports the `DMatrixf64` type alias.
 - The crate no longer re-exports the `core::f64::consts::PI` constant.

## Other changes

- Added the `serde` feature which implements the `Serialize` and `Deserialize` traits from [`serde`](https://crates.io/crates/serde) for all quadrature rule structs.
- The quadrature rule structs now store the nodes and weights together in a single allocation. This slightly speeds up integration.
 - Fixed a sign error in the documentation for `GaussJacobi`.