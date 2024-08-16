use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLaguerre;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("laguerre");
    for deg in [3, 10, 40, 200, 1_000] {
        let rule = GaussLaguerre::new(deg, -0.5).unwrap();
        group.bench_function(&format!("constructor, degree {deg}"), |b| {
            b.iter(|| black_box(GaussLaguerre::new(deg, -0.5).unwrap()))
        });
        group.bench_function(&format!("cheap integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| x * x - x - 1.0)))
        });
        group.bench_function(&format!("expensive integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| x.sin().cos().asin().acos().sin().cos())))
        });
        #[cfg(feature = "rayon")]
        group.bench_function(
            &format!("cheap integrand, degree {deg}, parallelized"),
            |b| b.iter(|| black_box(rule.par_integrate(|x| x * x - x - 1.0))),
        );
        #[cfg(feature = "rayon")]
        group.bench_function(
            &format!("expensive integrand, degree {deg}, parallelized"),
            |b| {
                b.iter(|| {
                    black_box(rule.par_integrate(|x| x.sin().cos().asin().acos().sin().cos()))
                })
            },
        );
    }
}

criterion_group!(bench, benches);
criterion_main!(bench);
