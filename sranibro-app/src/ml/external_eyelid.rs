//! Data-only XR5 model packages. Native inference runs only in a supervised child.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read},
    path::{Path, PathBuf},
};

const MAX_MODEL: u64 = 32 * 1024 * 1024;
/// Continuous fixed-axis readout, independent of the eyelid's fixed strip.
/// Projection bins are cached in canonical network orientation; no new inference.
fn weighted_wide_height(foreground: &[f32], bins: &[usize], mass: &mut [f64]) -> f32 {
    mass.fill(0.0);
    for (&p, &bin) in foreground.iter().zip(bins) { mass[bin] += p as f64; }
    let total: f64 = mass.iter().sum();
    if total <= 1e-9 { return 0.0; }
    (mass.iter().map(|m| m*m).sum::<f64>() / total) as f32
}
#[test]
fn weighted_wide_readout_is_continuous_and_translation_invariant() {
    let bins: Vec<usize> = (0..40000).map(|i| i % 200).collect();
    let mut mass = vec![0.0;200];
    let mut image=vec![0.0;40000];
    assert_eq!(weighted_wide_height(&image,&bins,&mut mass),0.0);
    for y in 40..60 { for x in 20..70 { image[y*200+x]=1.0; } }
    assert_eq!(weighted_wide_height(&image,&bins,&mut mass),20.0);
    image.fill(0.0);
    for y in 100..120 { for x in 100..150 { image[y*200+x]=1.0; } }
    assert_eq!(weighted_wide_height(&image,&bins,&mut mass),20.0);
    for x in 100..150 { image[120*200+x]=0.5; }
    assert_eq!(weighted_wide_height(&image,&bins,&mut mass),20.5);
    let mirrored: Vec<usize> = (0..40000).map(|i| 199-i%200).collect();
    assert_eq!(weighted_wide_height(&image,&mirrored,&mut mass),20.5);
}
fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub name: String,
    pub device: String,
    pub model_sha256: String,
    pub profile_sha256: String,
}
#[derive(Clone, Deserialize)]
pub struct Reference {
    pub center: [f64; 2],
    pub angle_rad: f64,
    pub width: f64,
    pub height_px: f64,
    pub central_interval: [f64; 2],
}
#[derive(Clone, Deserialize)]
pub struct Profile {
    pub references: [Reference; 2],
    pub normalizers: [f32; 2],
}
pub struct Package {
    pub manifest: Manifest,
    pub profile: Profile,
    pub model: Vec<u8>,
}
pub fn resolve(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|d| d.join(&p)))
            .unwrap_or(p)
    }
}
impl Package {
    pub fn read(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() > MAX_MODEL + 128 * 1024 {
            return Err(invalid("Model package is too large"));
        }
        let mut zip = zip::ZipArchive::new(file).map_err(invalid)?;
        if zip.len() != 3 {
            return Err(invalid("Expected a data-only SRanibro model package"));
        }
        let mut parts = [None, None, None];
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(invalid)?;
            let (slot, limit) = match entry.name() {
                "manifest.json" => (0, 16 * 1024),
                "profile.json" => (1, 64 * 1024),
                "model.onnx" => (2, MAX_MODEL),
                _ => return Err(invalid("Unexpected model package entry")),
            };
            if parts[slot].is_some() || entry.size() > limit || entry.is_dir() {
                return Err(invalid("Duplicate or oversized model package entry"));
            }
            let mut bytes = Vec::new();
            (&mut entry).take(limit + 1).read_to_end(&mut bytes)?;
            if bytes.len() as u64 > limit {
                return Err(invalid("Oversized model package entry"));
            }
            parts[slot] = Some(bytes);
        }
        let [Some(manifest), Some(profile), Some(model)] = parts else {
            return Err(invalid("Incomplete model package"));
        };
        let manifest: Manifest = serde_json::from_slice(&manifest).map_err(invalid)?;
        if manifest.format != "sranibro-xr5-aperture-v1"
            || manifest.device != "pimax_xr5"
            || manifest.name.is_empty()
            || manifest.name.len() > 96
            || manifest.name.chars().any(char::is_control)
        {
            return Err(invalid("Unsupported XR5 eyelid model format"));
        }
        if format!("{:x}", Sha256::digest(&model)) != manifest.model_sha256
            || format!("{:x}", Sha256::digest(&profile)) != manifest.profile_sha256
        {
            return Err(invalid("Model package checksum mismatch"));
        }
        let profile: Profile = serde_json::from_slice(&profile).map_err(invalid)?;
        profile.validate()?;
        Ok(Self {
            manifest,
            profile,
            model,
        })
    }
}
impl Profile {
    fn validate(&self) -> io::Result<()> {
        for (r, n) in self.references.iter().zip(self.normalizers) {
            if !r
                .center
                .iter()
                .all(|v| v.is_finite() && (0.0..=199.0).contains(v))
                || !r.angle_rad.is_finite()
                || r.angle_rad.abs() > std::f64::consts::PI
                || !(10.0..=200.0).contains(&r.width)
                || !(5.0..=100.0).contains(&r.height_px)
                || !r
                    .central_interval
                    .iter()
                    .all(|v| v.is_finite() && v.abs() <= 140.0)
                || r.central_interval[1] - r.central_interval[0] < 5.0
                || !n.is_finite()
                || !(3.0..=200.0).contains(&n)
            {
                return Err(invalid("Invalid eye reference in model package"));
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
mod native {
    use super::*;
    use crate::ml::xr5_research::EyePrediction;
    use ort::{session::Session, value::TensorRef};
    use std::{
        io::Write,
        time::{Duration, Instant},
    };

    pub struct Runner {
        session: Session,
        profile: Profile,
        roi: [Vec<f32>; 3],
        // (pixel index, lateral bin) in raw camera orientation.
        bands: [Vec<(usize, usize)>; 2],
        columns: [usize; 2],
        pub gpu: bool,
    }
    fn area_pool(input: &[f32], size: usize) -> Vec<f32> {
        let mut out = vec![0.0; 2 * size * size];
        for e in 0..2 {
            for y in 0..size {
                for x in 0..size {
                    let (a, b, c, d) = (
                        y * 200 / size,
                        ((y + 1) * 200).div_ceil(size),
                        x * 200 / size,
                        ((x + 1) * 200).div_ceil(size),
                    );
                    let mut sum = 0.0;
                    for yy in a..b {
                        for xx in c..d {
                            sum += input[e * 40000 + yy * 200 + xx];
                        }
                    }
                    out[e * size * size + y * size + x] = sum / ((b - a) * (d - c)) as f32;
                }
            }
        }
        out
    }
    impl Runner {
        pub fn new(package: Package, gpu: bool) -> io::Result<Self> {
            // Never load a library named by a model or found via CWD/PATH.
            let dll = std::env::current_exe()?
                .parent()
                .ok_or_else(|| invalid("No application folder"))?
                .join("model-runtime/onnxruntime.dll");
            if !dll.is_file() {
                return Err(invalid(
                    "Model runtime missing; restore the model-runtime folder from the app ZIP",
                ));
            }
            ort::init_from(&dll)
                .map_err(invalid)?
                .with_name("SRanibro XR5 worker")
                .commit();
            let mut builder = Session::builder()
                .map_err(invalid)?
                .with_intra_threads(2)
                .map_err(invalid)?
                .with_parallel_execution(false)
                .map_err(invalid)?
                .with_memory_pattern(false)
                .map_err(invalid)?;
            if gpu {
                builder = builder
                    .with_execution_providers([ort::ep::DirectML::default()
                        .with_device_id(0)
                        .build()
                        .error_on_failure()])
                    .map_err(invalid)?;
            }
            let session = builder
                .commit_from_memory(&package.model)
                .map_err(invalid)?;
            if session.inputs().len() != 4 || session.outputs().len() != 3 {
                return Err(invalid("Expected four XR5 inputs and three outputs"));
            }
            let mut full = vec![0.0; 80000];
            let mut bands: [Vec<(usize, usize)>; 2] = Default::default();
            let mut columns = [0; 2];
            for (e, r) in package.profile.references.iter().enumerate() {
                let (s, c) = r.angle_rad.sin_cos();
                let first = r.central_interval[0].ceil() as i32;
                columns[e] = (r.central_interval[1].floor() as i32 - first + 1) as usize;
                for y in 0..200 {
                    for x in 0..200 {
                        let dx = x as f64 - r.center[0];
                        let dy = y as f64 - r.center[1];
                        let u = dx * c + dy * s;
                        let v = -dx * s + dy * c;
                        let net_x = if e == 1 { 199 - x } else { x };
                        full[e * 40000 + y * 200 + net_x] = f32::from(
                            (u / (r.width * 0.65)).powi(2) + (v / (r.height_px * 1.3)).powi(2)
                                <= 1.0,
                        );
                        let bin = u.round_ties_even() as i32 - first;
                        if bin >= 0
                            && (bin as usize) < columns[e]
                            && v.abs() <= 12.0f64.max(r.height_px * 1.5)
                        {
                            bands[e].push((y * 200 + x, bin as usize));
                        }
                    }
                }
            }
            let roi = [25, 13, 7].map(|n| area_pool(&full, n));
            let mut runner = Self {
                session,
                profile: package.profile,
                roi,
                bands,
                columns,
                gpu,
            };
            // Validate actual graph inputs, output types/shapes and values before Ready.
            runner.predict(&vec![128; 80000])?;
            Ok(runner)
        }
        pub fn predict(&mut self, pixels: &[u8]) -> io::Result<[EyePrediction; 2]> {
            if pixels.len() != 80000 {
                return Err(invalid("Expected a 200x200 stereo frame"));
            }
            let mut input = vec![0.0; 80000];
            for e in 0..2 {
                for y in 0..200 {
                    for x in 0..200 {
                        let nx = if e == 1 { 199 - x } else { x };
                        input[e * 40000 + y * 200 + nx] =
                            pixels[e * 40000 + y * 200 + x] as f32 / 255.0;
                    }
                }
            }
            let outputs=self.session.run(ort::inputs![
                "image" => TensorRef::from_array_view(([2,1,200,200],input.as_slice())).map_err(invalid)?,
                "roi25" => TensorRef::from_array_view(([2,1,25,25],self.roi[0].as_slice())).map_err(invalid)?,
                "roi13" => TensorRef::from_array_view(([2,1,13,13],self.roi[1].as_slice())).map_err(invalid)?,
                "roi7" => TensorRef::from_array_view(([2,1,7,7],self.roi[2].as_slice())).map_err(invalid)?
            ]).map_err(invalid)?;
            let (fs, foreground) = outputs
                .get("foreground")
                .ok_or_else(|| invalid("Missing foreground"))?
                .try_extract_tensor::<f32>()
                .map_err(invalid)?;
            let (ms, mask) = outputs
                .get("mask")
                .ok_or_else(|| invalid("Missing mask"))?
                .try_extract_tensor::<u8>()
                .map_err(invalid)?;
            let (cs, closed) = outputs
                .get("closed_score")
                .ok_or_else(|| invalid("Missing closed score"))?
                .try_extract_tensor::<f32>()
                .map_err(invalid)?;
            if fs.as_ref() != [2, 200, 200]
                || ms.as_ref() != [2, 200, 200]
                || cs.as_ref() != [2]
                || !foreground
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                || !closed
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                || mask.iter().any(|v| *v > 3)
            {
                return Err(invalid("Model output is incompatible or non-finite"));
            }
            let mut result = [EyePrediction::default(); 2];
            for e in 0..2 {
                let raw = &pixels[e * 40000..(e + 1) * 40000];
                let mean = raw.iter().map(|v| *v as f64).sum::<f64>() / 40000.0;
                let variance =
                    raw.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / 40000.0;
                let sensor_ok = variance > 4.0
                    && raw.iter().filter(|v| **v < 2).count() < 39200
                    && raw.iter().filter(|v| **v > 253).count() < 39200;
                let fg = mask[e * 40000..(e + 1) * 40000]
                    .iter()
                    .filter(|v| **v > 0)
                    .count();
                let mut sum = 0.0f64;
                let mut count = 0;
                let mut bins = vec![false; self.columns[e]];
                for &(index, bin) in &self.bands[e] {
                    let idx = e * 40000
                        + if e == 1 {
                            index / 200 * 200 + 199 - index % 200
                        } else {
                            index
                        };
                    sum += foreground[idx] as f64;
                    if mask[idx] > 0 {
                        count += 1;
                        bins[bin] = true;
                    }
                }
                let height = (sum / self.columns[e] as f64) as f32;
                let support = bins.iter().filter(|v| **v).count() as f32 / self.columns[e] as f32;
                let opening = height / self.profile.normalizers[e];
                let status = if !sensor_ok || height > 200.0 || opening > 8.0 {
                    5
                } else if fg == 0 {
                    if closed[e] >= 0.9 {
                        1
                    } else {
                        2
                    }
                } else if closed[e] >= 0.9 {
                    3
                } else if support < 0.35 || count < 12 {
                    4
                } else {
                    0
                };
                result[e] = EyePrediction {
                    wide_height: None,
                    status,
                    opening: match status {
                        0 => Some(opening),
                        1 => Some(0.0),
                        _ => None,
                    },
                    closed_score: closed[e],
                    height: height.min(200.0),
                };
            }
            Ok(result)
        }
    }
    pub fn run(path: &Path, gpu: bool) -> io::Result<()> {
        let mut runner = Runner::new(Package::read(path)?, gpu)?;
        let mut input = io::stdin().lock();
        let mut output = io::stdout().lock();
        output.write_all(b"XR5LIV17")?;
        output.flush()?;
        let mut pixels = vec![0; 80000];
        let mut previous = None;
        let mut reference: Option<(Instant, [Vec<f32>; 2])> = None;
        let mut reference_status = 0u8;
        loop {
            let mut header = [0; 9];
            // EOF between packets is a normal parent shutdown, partial packets are not.
            if input.read(&mut header[..1])? == 0 {
                return Ok(());
            }
            input.read_exact(&mut header[1..])?;
            input.read_exact(&mut pixels)?;
            let seq = u64::from_le_bytes(header[..8].try_into().unwrap());
            if header[8] > 1 || previous.is_some_and(|p| seq <= p) {
                return Err(invalid("Invalid model request"));
            }
            previous = Some(seq);
            if header[8] == 1 {
                reference = Some((Instant::now(), Default::default()));
                reference_status = 1;
            }
            let mut eyes = runner.predict(&pixels)?;
            if let Some((start, samples)) = reference.as_mut() {
                for e in 0..2 {
                    if eyes[e].status == 0
                        && eyes[e].closed_score < 0.5
                        && eyes[e].height > 3.0
                        && samples[e].len() < 240
                    {
                        samples[e].push(eyes[e].height);
                    }
                }
                if start.elapsed() >= Duration::from_millis(1500)
                    && samples.iter().all(|s| s.len() >= 30)
                {
                    for e in 0..2 {
                        samples[e].sort_by(f32::total_cmp);
                        let n = samples[e].len();
                        runner.profile.normalizers[e] =
                            (samples[e][(n - 1) / 2] + samples[e][n / 2]) * 0.5;
                        if eyes[e].status == 0 {
                            eyes[e].opening = Some(eyes[e].height / runner.profile.normalizers[e]);
                        }
                    }
                    reference = None;
                    reference_status = 2;
                } else if start.elapsed() >= Duration::from_secs(4) {
                    reference = None;
                    reference_status = 3;
                }
            }
            output.write_all(b"X5R1")?;
            output.write_all(&seq.to_le_bytes())?;
            output.write_all(&[u8::from(runner.gpu), reference_status])?;
            for eye in eyes {
                output.write_all(&[eye.status])?;
                for v in [eye.opening.unwrap_or(0.0), eye.closed_score, eye.height] {
                    output.write_all(&v.to_le_bytes())?;
                }
            }
            output.flush()?;
        }
    }
}

#[cfg(windows)]
pub fn worker_entrypoint() -> bool {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new("--xr5-model-worker")) {
        return false;
    }
    let result = (|| {
        let path = args.next().ok_or_else(|| invalid("Missing model file"))?;
        let backend = args
            .next()
            .ok_or_else(|| invalid("Missing inference mode"))?;
        if backend != "cpu" && backend != "gpu" {
            return Err(invalid("Invalid inference mode"));
        }
        native::run(Path::new(&path), backend == "gpu")
    })();
    if let Err(e) = result {
        eprintln!("[xr5-model] {e}");
        std::process::exit(2);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn package(entries: &[(&str, &[u8])]) -> io::Result<Package> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "sranimodel-test-{}-{}.zip",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        {
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&path)?);
            for (name, bytes) in entries {
                zip.start_file(*name, zip::write::SimpleFileOptions::default())
                    .map_err(invalid)?;
                zip.write_all(bytes)?;
            }
            zip.finish().map_err(invalid)?;
        }
        let result = Package::read(&path);
        let _ = std::fs::remove_file(&path);
        result
    }
    #[test]
    fn package_checks_hash_device_and_rejects_extra_code_or_paths() {
        let profile=br#"{"references":[{"center":[75,100],"angle_rad":0.4,"width":110,"height_px":45,"central_interval":[-27.5,27.5]},{"center":[125,100],"angle_rad":-0.4,"width":110,"height_px":45,"central_interval":[-27.5,27.5]}],"normalizers":[40,40]}"#;
        let model = b"test graph; validated by worker separately";
        let mut manifest = serde_json::json!({"format":"sranibro-xr5-aperture-v1","device":"pimax_xr5","name":"Test","model_sha256":format!("{:x}",Sha256::digest(model)),"profile_sha256":format!("{:x}",Sha256::digest(profile))});
        let valid = serde_json::to_vec(&manifest).unwrap();
        assert!(package(&[
            ("manifest.json", &valid),
            ("profile.json", profile),
            ("model.onnx", model)
        ])
        .is_ok());
        assert!(package(&[
            ("manifest.json", &valid),
            ("profile.json", profile),
            ("model.onnx", b"changed")
        ])
        .is_err());
        assert!(package(&[
            ("manifest.json", &valid),
            ("profile.json", profile),
            ("../model.onnx", model)
        ])
        .is_err());
        assert!(package(&[
            ("manifest.json", &valid),
            ("profile.json", profile),
            ("model.onnx", model),
            ("worker.py", b"code")
        ])
        .is_err());
        manifest["device"] = "psvr2".into();
        let wrong = serde_json::to_vec(&manifest).unwrap();
        assert!(package(&[
            ("manifest.json", &wrong),
            ("profile.json", profile),
            ("model.onnx", model)
        ])
        .is_err());
    }
    #[test]
    fn invalid_profile_is_rejected() {
        let r = Reference {
            center: [100.0; 2],
            angle_rad: 0.0,
            width: 110.0,
            height_px: 45.0,
            central_interval: [-27.5, 27.5],
        };
        let mut p = Profile {
            references: [r.clone(), r],
            normalizers: [40.0; 2],
        };
        assert!(p.validate().is_ok());
        p.normalizers[1] = f32::NAN;
        assert!(p.validate().is_err());
        p.normalizers[1] = 0.0;
        assert!(p.validate().is_err());
        p.normalizers[1] = 40.0;
        p.references[0].central_interval = [10.0, 0.0];
        assert!(p.validate().is_err());
    }
}
