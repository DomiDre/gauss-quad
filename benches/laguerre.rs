use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLaguerre;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("laguerre");
    for deg in [3, 10, 40, 200, 1_000] {
        let rule = GaussLaguerre::new(deg, -0.5);
        group.bench_function(&format!("cheap integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| x * x - x - 1.0)))
        });
        group.bench_function(&format!("expensive integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| x.sin().cos().asin().acos().sin().cos())))
        });
    }
}

criterion_group!(bench, benches);
criterion_main!(bench);
