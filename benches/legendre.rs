use core::{f64::consts::PI, hint::black_box};
use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLegendre;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("legendre");
    for deg in [3, 10, 40, 200, 1_000, 10_000, 100_000, 1_000_000] {
        let rule = GaussLegendre::new(deg).unwrap();
        group.bench_function(&format!("constructor, degree {deg}"), |b| {
            b.iter(|| black_box(GaussLegendre::new(deg).unwrap()))
        });
        group.bench_function(&format!("cheap integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(-1.0, 1.0, |x| x * x - x - 1.0)))
        });
        group.bench_function(&format!("expensive integrand, degree {deg}"), |b| {
            b.iter(|| {
                black_box(
                    rule.integrate(0.0, 2.0 * PI, |x| x.sin().cos().asin().acos().sin().cos()),
                )
            })
        });
        if deg <= 1000 {
            #[cfg(feature = "rayon")]
            group.bench_function(&format!("double integral, degree {deg}, serial"), |b| {
                b.iter(|| {
                    black_box(rule.integrate(-1.0, 1.0, |_y| {
                        rule.integrate(-1.0, 1.0, |x| (x.sin().powi(2) + 2.0).ln().cos().acos())
                    }))
                })
            });
            #[cfg(feature = "rayon")]
            group.bench_function(
                &format!("double integral, degree {deg}, parallelized"),
                |b| {
                    b.iter(|| {
                        black_box(rule.par_integrate(-1.0, 1.0, |_y| {
                            rule.integrate(-1.0, 1.0, |x| (x.sin().powi(2) + 2.0).ln().cos().acos())
                        }))
                    })
                },
            );
        }
    }
}

criterion_group!(bench, benches);
criterion_main!(bench);
