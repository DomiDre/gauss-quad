gauss-quad
=========

The ``gauss-quad`` crate is a small library to calculate integrals of the type

 ![equation](https://latex.codecogs.com/svg.latex?%5Cint_a%5Eb%20f%28x%29%20w%28x%29%20%5Cmathrm%7Bd%7Dx)

 using Gaussian quadrature.

 To use the crate, the desired quadrature rule and the trait whether it is a DefiniteIntegral or ImproperIntegral has to be included in the program, e.g. for a Gauss-Legendre rule
 
 ```
  use gauss_quad::{GaussLegendre, DefiniteIntegral};
 ```
 
 The general call structure is to first initialize the n-point quadrature rule setting the degree n via

```
 let quadrature = gauss_quad::QUADRATURE_RULE::init(n);
```

where QUADRATURE_RULE can currently be either of:

```
GaussLegendre
GaussHermite
```

Then to calculate the integral of a function can be calculated by calling

```
let integral = quad.integrate(a, b, f(x));
```

where a and b (both f64) are the integral bounds and the f(x) the integrand (fn(f64) -> f64).
For example to integrate a parabola from 0..1 one can use a lambda expression as integrand and call:
```
let integral = quad.integrate(0.0, 1.0, |x| x*x);
```

If the integral is improper, as in the case of Gauss Hermite integrals

![equation](https://latex.codecogs.com/svg.latex?%5Cint_%7B-%5Cinfty%7D%5E%5Cinfty%20f%28x%29%20e%5E%7B-x%5E2%7D%20%5Cmathrm%7Bd%7Dx)

no integral bounds should be passed and the call simplifies to
```
let integral = quad.integrate(f(x));
```
