use configural::bitset::and_popcount;
use configural::stats::{mi_2x2, wilson_lower};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_and_popcount(c: &mut Criterion) {
    for &w in &[1, 4, 16, 64] {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64; w];
        let b = vec![0xAAAA_AAAA_AAAA_AAAAu64; w];
        c.bench_function(&format!("and_popcount_w{}", w), |bencher| {
            bencher.iter(|| black_box(and_popcount(black_box(&a), black_box(&b))))
        });
    }
}

fn bench_mi_2x2(c: &mut Criterion) {
    c.bench_function("mi_2x2", |b| {
        b.iter(|| {
            black_box(mi_2x2(
                black_box(5.0),
                black_box(10.0),
                black_box(10.0),
                black_box(20.0),
            ))
        })
    });
}

fn bench_wilson_lower(c: &mut Criterion) {
    c.bench_function("wilson_lower", |b| {
        b.iter(|| {
            black_box(wilson_lower(
                black_box(0.5),
                black_box(100.0),
                black_box(1.96),
            ))
        })
    });
}

criterion_group!(
    benches,
    bench_and_popcount,
    bench_mi_2x2,
    bench_wilson_lower
);
criterion_main!(benches);
