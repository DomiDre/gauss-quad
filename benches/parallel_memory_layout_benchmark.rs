use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::GaussLegendre;

fn serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("serial x^2");
    for deg in [3, 10, 40, 100] {
        let rule = GaussLegendre::new(deg);
        group.bench_function(format!("degree {deg}"), |b| {
            b.iter(|| core::hint::black_box(rule.integrate(0.0, 2.0, |x| x * x)))
        });
    }
}

criterion_group!(parallel_memory_layout_benchmark, serial);
criterion_main!(parallel_memory_layout_benchmark);
