// Copyright 2019-2024 Dominique Dresen
// Copyright 2023-2026 Johanna Sörngård
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::hint::black_box;
use criterion::{Criterion, criterion_group, criterion_main};
use gauss_quad::GaussHermite;

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("hermite");
    for deg in [3, 10, 40, 200, 1_000] {
        let deg = deg.try_into().unwrap();
        let rule = GaussHermite::new(deg);
        group.bench_function(&format!("constructor, degree {deg}"), |b| {
            b.iter(|| black_box(GaussHermite::new(deg)))
        });
        group.bench_function(&format!("cheap integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| x * x - x - 1.0)))
        });
        group.bench_function(&format!("expensive integrand, degree {deg}"), |b| {
            b.iter(|| black_box(rule.integrate(|x| (x.sin().powi(2) + 2.0).ln().cos().acos())))
        });
        #[cfg(feature = "rayon")]
        group.bench_function(&format!("double integral, degree {deg}, serial"), |b| {
            b.iter(|| {
                black_box(
                    rule.integrate(|_y| {
                        rule.integrate(|x| (x.sin().powi(2) + 2.0).ln().cos().acos())
                    }),
                )
            })
        });
        #[cfg(feature = "rayon")]
        group.bench_function(
            &format!("double integral, degree {deg}, parallelized"),
            |b| {
                b.iter(|| {
                    black_box(rule.par_integrate(|_y| {
                        rule.integrate(|x| (x.sin().powi(2) + 2.0).ln().cos().acos())
                    }))
                })
            },
        );
    }
}

criterion_group!(bench, benches);
criterion_main!(bench);
