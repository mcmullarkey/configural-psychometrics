use configural::exhaustive::ascend;
use configural::pairmi::{EvaluationMode, PairMiEngine};
use configural::BinaryMatrix;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn make_matrix(n: usize, v: usize, seed: u64) -> BinaryMatrix {
    // Simple deterministic data generation
    let mut data = Vec::with_capacity(n * v);
    for i in 0..n {
        for j in 0..v {
            data.push(if (i as u64 + j as u64 * seed).is_multiple_of(3) {
                1.0
            } else {
                0.0
            });
        }
    }
    BinaryMatrix::new(&data, n, v).unwrap()
}

fn bench_emsc_ascent(c: &mut Criterion) {
    let m = make_matrix(200, 13, 42);
    let target: Vec<u8> = (0..200).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    c.bench_function("emsc_v13_n200_d8", |b| {
        b.iter(|| {
            black_box(ascend(
                black_box(&m),
                black_box(std::slice::from_ref(&target)),
                black_box(&[0.5]),
                black_box(2),
                black_box(0.05),
                black_box(8),
                black_box(f64::MAX),
                black_box(&[None]),
            ))
        })
    });

    let m2 = make_matrix(200, 25, 42);
    c.bench_function("emsc_v25_n200_d5", |b| {
        b.iter(|| {
            black_box(ascend(
                black_box(&m2),
                black_box(std::slice::from_ref(&target)),
                black_box(&[0.5]),
                black_box(2),
                black_box(0.05),
                black_box(5),
                black_box(f64::MAX),
                black_box(&[None]),
            ))
        })
    });
}

fn bench_pairmi_depth(c: &mut Criterion) {
    let m = make_matrix(200, 20, 42);
    c.bench_function("pairmi_depth_loop", |b| {
        b.iter(|| {
            let mut engine = PairMiEngine::create(black_box(&m)).unwrap();
            engine.depth(2, EvaluationMode::Alpha(0.05));
            engine.depth(3, EvaluationMode::Alpha(0.05));
            engine.depth(4, EvaluationMode::Alpha(0.05));
            black_box(engine.results())
        })
    });
}

criterion_group!(benches, bench_emsc_ascent, bench_pairmi_depth);
criterion_main!(benches);
