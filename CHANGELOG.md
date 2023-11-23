# 0.2.0

This update is mostly about changing the API to adhere to the [Rust API guidelines](https://rust-lang.github.io/api-guidelines/about.html).

 - Changed the name of all constructors from `init` to `new`.
 - Added the `serde` feature which implements the `Serialize` and `Deserialize` traits from [`serde`](https://crates.io/crates/serde) for all quadrature rule structs.

## Minor changes

 - Fixed a sign error in the documentation for `GaussJacobi`.