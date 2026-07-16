//! ML keystone verifier + microbenchmark.
//!
//! Loads SRanipal weights (TVM params) in pure Rust, runs the EyePrediction CNN
//! forward, and compares to the Python ONNX reference on an identical fixed
//! input. Proves pure-Rust ML straight from the user's SRanipal directory.
//!
//! Run: cargo run --release --example ml_keystone -- [params path] [rust_ref dir]

use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

const DEFAULT_REF: &str = "rust_ref";

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let params = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: ml_keystone <SRanipal params path> [rust_ref dir]");
        std::process::exit(2);
    });
    let refdir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| DEFAULT_REF.to_string());

    println!("[ml] loading SRanipal weights: {params}");
    let map = tvm_params::parse_map(&params).expect("parse params");
    println!("[ml] {} tensors loaded", map.len());
    let mut net = match EyeNet::new(map) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[ml] model validation failed: {e}");
            std::process::exit(1);
        }
    };

    let input = read_f32(&format!("{refdir}/input.bin"));
    let refout = read_f32(&format!("{refdir}/ref_out.bin"));
    assert_eq!(input.len(), 2 * 2 * 100 * 100, "input size");
    assert_eq!(refout.len(), 2 * 5, "ref size");

    let per_eye = 2 * 100 * 100;
    let mut got = Vec::new();
    for n in 0..2 {
        let out = net.forward_one(&input[n * per_eye..(n + 1) * per_eye]);
        got.extend_from_slice(&out);
    }

    let mut max_abs = 0f32;
    for (g, r) in got.iter().zip(refout.iter()) {
        max_abs = max_abs.max((g - r).abs());
    }
    println!("[ml] rust   = {:?}", round(&got));
    println!("[ml] onnx   = {:?}", round(&refout));
    println!("[ml] max_abs_diff = {max_abs:.6}");
    if max_abs < 1e-3 {
        println!("[ml] KEYSTONE OK: pure-Rust SRanipal inference matches ONNX");
    } else {
        println!("[ml] MISMATCH (>1e-3) — investigate layer/weight mapping");
        std::process::exit(2);
    }

    // Honest perf check (the point is low load): time the naive forward.
    let iters = 300usize;
    let t0 = std::time::Instant::now();
    let mut sink = 0f32;
    for _ in 0..iters {
        sink += net.forward_one(&input[0..per_eye])[1];
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "[bench] {:.3} ms/inference  (~{:.0} Hz/eye, ~{:.0} Hz both eyes)  [sink {:.2}]",
        per * 1e3,
        1.0 / per,
        1.0 / (2.0 * per),
        sink
    );
}

fn round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 1e5).round() / 1e5).collect()
}
