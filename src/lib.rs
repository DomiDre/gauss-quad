#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate assert_float_eq;

use nalgebra::{Dynamic, Matrix, VecStorage};
pub type DMatrixf64 = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
pub use std::f64::consts::PI;

pub mod gaussian_quadrature;
pub use gaussian_quadrature::{QuadratureRule, DefiniteIntegral, ImproperIntegral};

pub mod hermite;
pub use hermite::GaussHermite;

pub mod legendre;
pub use legendre::GaussLegendre;
