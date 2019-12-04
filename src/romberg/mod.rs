/*!
Numerical integration using the Romberg rule.

Romberg's method allows to iteratively decrease the step width used in the
estimation of the integral value and subsequently uses Richardson extrapolation,
to estimate the limit for h->0.
With the step width for the n-th iteration given by h_n = (b-a)/2^n, the Romberg
integral is inductively determined by the scheme
R(0,0) = h_1 ( f(a) + f(b) )
R(n,0) = 1/2 * R(n-1,0) + h_n Sum_{k=1..2^(n-1)} f(a + (2k-1)*h_n)
R(n,m) = R(n, m-1) + 1/(4^m - 1)*( R(n,m-1) - R(n-1,m-1) )

R(n,0) corresponds to the trapezoidal rule of degree 2^n+1, R(n,1) is equivalent
to the Simpson rule 2^n+1 points, R(n,2) to Booles rule.
The calculated integral approximation is then given by R(n,n). 
The implemented algorithm always stores the last R(n,m) (with n fixed and m=1..n)
to be able to calculate the next values R(n+1,m) on demand.
The error of the calculated integral is estimated by R(n,n) - R(n-1,n-1)



!*/

#[derive(Debug, Clone)]
/// A Romberg rule quadrature scheme.
pub struct Romberg {
    /// the last calculated row of R(n,m), the integral is given by R(n,n)
    /// 
    /// TO DO: Check which values need to be stored
    romberg_row: Vec<f64>,
    initial_degree: usize,
    current_integrand: fn(f64) -> f64,
    degree: usize,
    h_n: f64,
    err: f64
}

impl Romberg {

    /// Initialize a new Romberg rule with `degree` being the number of bisections
    pub fn init(initial_degree: usize) -> Self {
        Self {   
            // TO DO: Initialize Romberg
        }
    }
    
    fn romberg_step(&mut self) {
        self.h_n = Some(self.h_n) / 2.0; // take half of previous h_n for next step

        // R_n_0 = 0.5*self.romberg_row[0] +
        //     self.h_n * (1..2.0_f64.powi(self.degree-1))
        //     .map(|k| integrand(a + (2.0*k-1.0)*h_n))
        //     .sum();
        // self.romberg_row.
        // new_romberg_row.reserve();

        self.degree = self.degree + 1;
    }

    /// Integrate over the domain [a, b].
    pub fn integrate<F>(&mut self, a: f64, b: f64, integrand: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        // initialize h_1 and R(0,0)
        self.h_n = (b - a) / 2.0; // calculate h_1
        self.romberg_row = Vec::new();
        self.romberg_row.push(self.h_n * (integrand(a) + integrand(b)));

        for idx in 1..self.initial_degree {
            self.romberg_step();
        }
        
        self.romberg_row[self.degree-1]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_Romberg_integration() {
        let quad = Romberg::init(2);
        let integral = quad.integrate(0.0, 1.0, |x| x*x);
        assert_float_absolute_eq!(integral, 1.0/3.0, 0.0001);
    }
}

