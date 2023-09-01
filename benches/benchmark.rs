use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLegendre;

fn small_degrees(c: &mut Criterion) {
    for deg in [3, 10, 30, 100] {
        let mut group = c.benchmark_group(format!("degree: {deg}"));
        group.bench_function("Bogaert", |b| {
            b.iter(|| GaussLegendre::nodes_and_weights(deg))
        });
        group.bench_function("Golub_Welsch", |b| {
            b.iter(|| GaussLegendre::nodes_and_weights_old(deg))
        });
    }
}

fn medium_degrees(c: &mut Criterion) {
    for deg in [300, 1000] {
        let mut group = c.benchmark_group(format!("degree: {deg}"));
        group.bench_function("Bogaert", |b| {
            b.iter(|| GaussLegendre::nodes_and_weights(deg))
        });
        group.bench_function("Golub_Welsch", |b| {
            b.iter(|| GaussLegendre::nodes_and_weights_old(deg))
        });
    }
}

criterion_group!(benches, small_degrees, medium_degrees);
criterion_main!(benches);
