/*!
# gauss-quad

**gauss-quad** is a Gaussian quadrature library for numerical integration.

## Quadrature rules
**gauss-quad** implements the following quadrature rules:
* Gauss-Legendre
* Gauss-Jacobi
* Gauss-Laguerre
* Gauss-Hermite
* Midpoint
* Simpson

## Using **gauss-quad**
First, add **gauss-quad** to your `Cargo.toml`:

```text
[dependencies]
gauss-quad = "0.1.4"
```

Then, you can use any of the quadrature rules in your project:
```rust
extern crate gauss_quad;
use gauss_quad::GaussLegendre;

fn main() {

    // initialize the quadrature rule
    let degree = 10;
    let quad = GaussLegendre::init(degree);

    // use the rule to integrate a function
    let left_bound = 0.0;
    let right_bound = 1.0;
    let integral = quad.integrate(left_bound, right_bound, |x| x * x);
}
```

## Setting up a quadrature rule
Using a quadrature rule takes two steps:
1. Initialization
2. Integration

First, rules must be initialized using some specific input parameters.

Then, you can integrate functions using those rules:
```
# extern crate gauss_quad;
# use gauss_quad::*;
# fn main() {
#   let degree = 5;
#   let alpha = 1.2;
#   let beta = 1.2;
#   let a = 0.0;
#   let b = 1.0;
#   let c = -10.;
#   let d = 100.;
let gauss_legendre = GaussLegendre::init(degree);
// Integrate on the domain [a,b]
let x_cubed = gauss_legendre.integrate(a, b, |x| x * x * x);

let gauss_jacobi = GaussJacobi::init(degree, alpha, beta);
// Integrate on the domain [c,d]
let double_x = gauss_jacobi.integrate(a, b, |x| 2.0 * x);

let gauss_laguerre = GaussLaguerre::init(degree, alpha);
// no explicit domain, Gauss-Laguerre integration is done on the domain (-∞, ∞).
let piecewise = gauss_laguerre.integrate(|x| if x > 0.0 && x < 2.0 { x } else { 0.0 });

let gauss_hermite = GaussHermite::init(degree);
// again, no explicit domain since integration is done over the domain (-∞, ∞).
let constant = gauss_hermite.integrate(|x| if x > -1.0 && x < 1.0 { 2.0 } else { 1.0 });
# }
```

## Specific quadrature rules
Different rules may take different parameters.

For example, the `GaussLaguerre` rule requires both a `degree` and an `alpha`
parameter.

`GaussLaguerre` is also defined as an improper integral over the domain (-∞, ∞).
This means no domain bounds are needed in the `integrate` call.
```rust
extern crate gauss_quad;
use gauss_quad::GaussLaguerre as quad_rule;

fn main() {

    // initialize the quadrature rule
    let degree = 10;
    let alpha = 0.5;
    let quad = quad_rule::init(degree, alpha);

    // use the rule to integrate a function
    let integral = quad.integrate(|x| x * x);
}
```

## Panics and errors
Quadrature rules are only defined for a certain set of input values.
For example, every rule is only defined for degrees where `degree > 1`.
```should_panic
# extern crate gauss_quad;
# use gauss_quad::GaussLaguerre;
# fn main() {
    let degree = 1;
    let quad = GaussLaguerre::init(degree, 0.1); // panics!
# }
```

Specific rules may have other requirements.
`GaussJacobi` for example, requires alpha and beta parameters larger than -1.0.
```should_panic
# extern crate gauss_quad;
# use gauss_quad::GaussJacobi;
# fn main() {
    let degree = 10;
    let alpha = 0.1;
    let beta = -1.1;

    let quad = GaussJacobi::init(degree, alpha, beta); // panics!
# }
```
Make sure to read the specific quadrature rule's documentation before using it.

Error handling is very simple: bad input values will cause the program to panic
and abort with a short error message.

## Passing functions to quadrature rules
The `integrate` method expects functions of the form `Fn(f64) -> f64`, i.e. functions of
one parameter.

```rust
extern crate gauss_quad;
use gauss_quad::GaussLegendre;

fn main() {

    // initialize the quadrature rule
    let degree = 10;
    let quad = GaussLegendre::init(degree);

    // use the rule to integrate a function
    let left_bound = 0.0;
    let right_bound = 1.0;
    let integral = quad.integrate(left_bound, right_bound, |x| x * x);
}
```
!*/

#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate assert_float_eq;

use nalgebra::{Dynamic, Matrix, VecStorage};
#[doc(inline)]
pub type DMatrixf64 = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
#[doc(inline)]
pub use std::f64::consts::PI;

pub mod gaussian_quadrature;
pub mod hermite;
pub mod legendre;
pub mod laguerre;
pub mod jacobi;
pub mod midpoint;
pub mod simpson;

#[doc(inline)]
pub use hermite::GaussHermite;
#[doc(inline)]
pub use legendre::GaussLegendre;
#[doc(inline)]
pub use laguerre::GaussLaguerre;
#[doc(inline)]
pub use jacobi::GaussJacobi;
#[doc(inline)]
pub use midpoint::Midpoint;
#[doc(inline)]
pub use simpson::Simpson;
