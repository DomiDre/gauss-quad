use core::{f64::consts::PI, hint::black_box};
use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussJacobi;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobi");
    for deg in [3, 10, 40, 200, 1_000] {
        let deg = deg.try_into().unwrap();
        let rule = GaussJacobi::new(
            deg,
            (-0.25).try_into().unwrap(),
            (-1.0 / 3.0).try_into().unwrap(),
        );
        group.bench_function(&format!("constructor, degree {deg}"), |b| {
            b.iter(|| {
                black_box(GaussJacobi::new(
                    deg,
                    (-0.25).try_into().unwrap(),
                    (-0.25).try_into().unwrap(),
                ))
            })
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

criterion_group!(bench, benches);
criterion_main!(bench);
