gauss-quad
=========
 [![Build Status](https://travis-ci.com/DomiDre/gauss-quad.svg?branch=master)](https://travis-ci.com/DomiDre/gauss-quad)
 [![](http://meritbadge.herokuapp.com/gauss-quad)](https://crates.io/crates/gauss-quad)
 
 The ``gauss-quad`` crate is a small library to calculate integrals of the type

 ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%20w%28x%29%20%5Cmathrm%7Bd%7Dx)
 
 using Gaussian quadrature.

 To use the crate, the desired quadrature rule  has to be included in the program, e.g. for a Gauss-Legendre rule
 
 ```
  use gauss_quad::GaussLegendre;
 ```
 
 The general call structure is to first initialize the n-point quadrature rule setting the degree n via

```
 let quad = QUADRATURE_RULE::init(n);
```

where QUADRATURE_RULE can currently be set to calculate either:

| QUADRATURE_RULE  | Integral      |
| -------------    | ------------- |
| Midpoint         | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%20%5Cmathrm%7Bd%7Dx)  |
| Simpson          | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%20%5Cmathrm%7Bd%7Dx)  |
| GaussLegendre    | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%20%5Cmathrm%7Bd%7Dx)  |
| GaussJacobi      | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%281-x%29%5E%5Calpha%20%281&plus;x%29%5E%5Cbeta%20%5Cmathrm%7Bd%7Dx)  |
| GaussLaguerre    | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_%7B-%5Cinfty%7D%5E%5Cinfty%20f%28x%29x%5E%5Calpha%20e%5E%7B-x%7D%20%5Cmathrm%7Bd%7Dx)  |
| GaussHermite     | ![equation](https://latex.codecogs.com/svg.latex?%5Cint_%7B-%5Cinfty%7D%5E%5Cinfty%20f%28x%29%20e%5E%7B-x%5E2%7D%20%5Cmathrm%7Bd%7Dx)  |

For the quadrature rules that take an additional parameter, such as Gauss-Laguerre and Gauss-Jacobi, the parameters have to be added to the initialization, e.g.

```
 let quad = GaussLaguerre::init(n, alpha);
```

Then to calculate the integral of a function call

```
let integral = quad.integrate(a, b, f(x));
```

where a and b (both f64) are the integral bounds and the f(x) the integrand (fn(f64) -> f64).
For example to integrate a parabola from 0..1 one can use a lambda expression as integrand and call:
```
let integral = quad.integrate(0.0, 1.0, |x| x*x);
```

If the integral is improper, as in the case of Gauss-Laguerre and Gauss-Hermite integrals, no integral bounds should be passed and the call simplifies to
```
let integral = quad.integrate(f(x));
```
