use compute::{compress_lossless, decompress_lossless, CompressParams};
use ndarray::{Array2, Axis};
use rand::Rng;
use std::time::Instant;
use rand_distr::{StandardNormal, Distribution};


fn main() {

    // Generate large array of shape (1000, 512) with values ~ residuals
    let mut rng = rand::rng();
    // let uniform = Uniform::new_inclusive(-32_000, 32_000);
    // let data: Array2<i32> = Array2::from_shape_fn((1000, 5000), |_| rng.sample(&uniform));

    let normal = StandardNormal;
    let data: Array2<i32> = Array2::from_shape_fn((10000, 4000), |_| {
        let sample: f64 = normal.sample(&mut rand::rng());
        (sample * 1000.0).round() as i32 // scale for larger dynamic range
    });

    let guard = pprof::ProfilerGuard::new(1000).unwrap(); // 100Hz sampling
    // Compress
    let start = Instant::now();

    use rayon::ThreadPoolBuilder;
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    pool.install(|| {
        let params = CompressParams::new(2000, 2000, 0, 0, 2);
        let compressed = compress_lossless(&data, params.clone()).unwrap();
        //let compressed = compress_residuals(&data).unwrap();
        println!("Compression took {:?}", start.elapsed());

        // Decompress
        let start = Instant::now();
        let recovered = decompress_lossless(&compressed, (10000, 4000), params.clone()).unwrap();
        //let recovered = decompress_residuals(&compressed).unwrap();
        println!("Decompression took {:?}", start.elapsed());

        // Element-wise equality check
        let eq = data.iter().zip(recovered.iter()).all(|(a, b)| a == b);
        if eq {
            println!("Roundtrip successful: all elements match.");
        } else {
            println!("Roundtrip failed: data mismatch.");
        }
    });
    

    if let Ok(report) = guard.report().build() {
        use std::fs::File;
        let file = File::create("flamegraph_other.svg").unwrap();

        use pprof::flamegraph::Options;

        let mut options = Options::default();
        options.image_width = Some(2000); // wider than default 1200

        report.flamegraph_with_options(file, &mut options).unwrap();

        // report.flamegraph(file).unwrap();
    }
}
