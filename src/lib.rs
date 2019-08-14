#![allow(dead_code)]
#![allow(unused_imports)]

#[macro_use]
extern crate assert_float_eq;

use nalgebra::{Dynamic, Matrix, VecStorage};
pub type DMatrixf64 = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
pub use std::f64::consts::PI;

pub mod gaussian_quadrature;

pub mod hermite;
pub use hermite::GaussHermite;

pub mod legendre;
pub use legendre::GaussLegendre;

pub mod laguerre;
pub use laguerre::GaussLaguerre;

pub mod jacobi;
pub use jacobi::GaussJacobi;
