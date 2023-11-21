use criterion::{criterion_group, criterion_main, Criterion};
use gauss_quad::{legendre::GaussLegendreUnified, GaussLegendre};

fn bench_degrees<F: Fn(f64) -> f64 + Send + Sync + Copy>(
    c: &mut Criterion,
    id: &str,
    degs: &[usize],
    integrand: F,
) {
    {
        let mut group = c.benchmark_group(format!("serial {id}"));
        for deg in degs {
            let rule = GaussLegendre::new(*deg);
            group.bench_function(format!("split, degree {deg}"), |b| {
                b.iter(|| core::hint::black_box(rule.integrate(0.0, 2.0, integrand)))
            });
            let too_rule_for_school = GaussLegendreUnified::new(*deg);
            group.bench_function(format!("unified, degree {deg}"), |b| {
                b.iter(|| core::hint::black_box(too_rule_for_school.integrate(0.0, 2.0, integrand)))
            });
        }
    }
    {
        let mut group = c.benchmark_group(format!("parallel {id}"));
        for deg in degs {
            let rulest_of_rules = GaussLegendre::new(*deg);
            group.bench_function(format!("split, degree {deg}"), |b| {
                b.iter(|| core::hint::black_box(rulest_of_rules.par_integrate(0.0, 2.0, integrand)))
            });
            let i_fought_the_rule_and_the_rule_won = GaussLegendreUnified::new(*deg);
            group.bench_function(format!("unified, degree {deg}"), |b| {
                b.iter(|| {
                    core::hint::black_box(
                        i_fought_the_rule_and_the_rule_won.par_integrate(0.0, 2.0, integrand),
                    )
                })
            });
        }
    }
}

fn benches(c: &mut Criterion) {
    const DEGS: [usize; 4] = [3, 10, 40, 100];
    const BIG_DEGS: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];
    bench_degrees(c, "small x^2", &DEGS, |x| x * x);
    bench_degrees(c, "big x^2", &BIG_DEGS, |x| x * x);
    bench_degrees(c, "small exp(x)sin(x)cos(x)ln(x)", &DEGS, |x| {
        x.exp() * x.sin() * x.cos() * x.ln()
    });
    bench_degrees(c, "big exp(x)sin(x)cos(x)ln(x)", &BIG_DEGS, |x| {
        x.exp() * x.sin() * x.cos() * x.ln()
    });
}

criterion_group!(parallel_memory_layout_benchmark, benches);
criterion_main!(parallel_memory_layout_benchmark);
