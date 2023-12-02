use core::{f64::consts::PI, hint::black_box};
use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLegendre;

fn benches(c: &mut Criterion) {
    for deg in [3, 10, 40, 200, 1_000, 10_000, 100_000, 1_000_000] {
        let rule = GaussLegendre::new(deg);
        c.bench_function(&format!("degree {deg}, cheap integrand"), |b| {
            b.iter(|| black_box(rule.integrate(-1.0, 1.0, |x| x * x - x - 1.0)))
        });
        c.bench_function(&format!("degree {deg}, expensive integrand"), |b| {
            b.iter(|| {
                black_box(
                    rule.integrate(0.0, 2.0 * PI, |x| x.sin().cos().asin().acos().sin().cos()),
                )
            })
        });
    }
}

criterion_group!(bench, benches);
criterion_main!(bench);
