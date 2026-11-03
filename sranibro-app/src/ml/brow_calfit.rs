
use std::path::Path;

use crate::ml::brow_fit::fit_head;
use crate::ml::brow_net::BrowNet;
use crate::ml::preprocess::{brow_input, BROW_SIDE};

pub struct Dataset {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
    pub rows: usize,
    pub skipped: usize,
}

pub fn load_dataset(brow_data_dir: &Path, backbone: &mut BrowNet) -> Result<Dataset, String> {
    let out_dim = backbone.out_dim();
    let labels_path = brow_data_dir.join("labels.csv");
    let text = std::fs::read_to_string(&labels_path)
        .map_err(|e| format!("labels.csv '{}': {e}", labels_path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    if !header.starts_with("filename,") {
        return Err(format!(
            "unexpected header '{header}' (want 'filename,brow,inner,outer')"
        ));
    }

    let images_dir = brow_data_dir.join("images");
    let mut features: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();
    let mut rows = 0usize;
    let mut skipped = 0usize;
    let mut buf = [0f32; BROW_SIDE * BROW_SIDE];

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let mut it = line.split(',');
        let fname = it.next().unwrap_or("").trim();
        let brow = match it
            .next()
            .map(|s| s.trim())
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(v) => v,
            None => return Err(format!("row '{line}': unparseable brow value")),
        };
        let inner = it
            .next()
            .map(|s| s.trim())
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let outer = it
            .next()
            .map(|s| s.trim())
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let label: Vec<f32> = if out_dim >= 3 {
            vec![brow, inner, outer]
        } else {
            vec![brow]
        };

        let img_path = images_dir.join(fname);
        match decode_gray8(&img_path) {
            Some((w, h, px)) => {
                brow_input(&px, w, h, false, &mut buf);
                features.push(backbone.brow_features(&buf));
                labels.push(label);
            }
            None => skipped += 1,
        }
    }

    if features.is_empty() {
        return Err(format!(
            "no usable frames ({rows} rows, {skipped} skipped) under {}",
            images_dir.display()
        ));
    }
    Ok(Dataset {
        features,
        labels,
        rows,
        skipped,
    })
}

pub fn fit_to_bytes(
    brow_data_dir: &Path,
    backbone_bin: &Path,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let mut backbone = BrowNet::load(backbone_bin)?;
    let out_dim = backbone.out_dim();
    let ds = load_dataset(brow_data_dir, &mut backbone)?;
    let head = fit_head(&ds.features, &ds.labels, out_dim, seed);
    backbone.to_bytes_with_head(&head)
}

fn decode_gray8(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = png::Decoder::new(file).read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    let (w, h) = (info.width as usize, info.height as usize);
    buf.truncate(info.buffer_size());
    if buf.len() < w * h {
        return None;
    }
    buf.truncate(w * h);
    Some((w, h, buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("sranibro_calfit_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn det_f32s(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed | 1;
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (st >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn synth_backbone_bytes(out_dim: usize) -> Vec<u8> {
        fn put(out: &mut Vec<u8>, v: &[f32]) {
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BROWNET1");
        bytes.extend_from_slice(&(out_dim as u32).to_le_bytes());
        let conv: [(usize, usize); 4] = [(1, 16), (16, 32), (32, 64), (64, 64)];
        let mut seed = 1u64;
        for (ic, oc) in conv {
            let w: Vec<f32> = det_f32s(oc * ic * 9, seed)
                .iter()
                .map(|x| x * 0.05)
                .collect();
            put(&mut bytes, &w);
            seed += 1;
            put(&mut bytes, &det_f32s(oc, seed));
            seed += 1;
        }
        put(&mut bytes, &det_f32s(128 * 1024, seed));
        seed += 1;
        put(&mut bytes, &det_f32s(128, seed));
        seed += 1;
        put(&mut bytes, &det_f32s(out_dim * 128, seed));
        seed += 1;
        put(&mut bytes, &det_f32s(out_dim, seed));
        bytes
    }

    fn write_png(path: &Path, w: u32, h: u32, px: &[u8]) {
        let file = std::fs::File::create(path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().unwrap();
        wr.write_image_data(px).unwrap();
    }

    fn conv_region_end() -> usize {
        let conv_f32: usize = [(1usize, 16usize), (16, 32), (32, 64), (64, 64)]
            .iter()
            .map(|(ic, oc)| oc * ic * 9 + oc)
            .sum();
        12 + conv_f32 * 4
    }

    #[test]
    fn fit_to_bytes_produces_loadable_model() {
        let dir = tmp_dir("fit_bytes");
        let images = dir.join("images");
        let mut csv = String::from("filename,brow,inner,outer\n");
        for (folder, brow) in [("neutral", 0.0f32), ("brow_up", 1.0f32)] {
            std::fs::create_dir_all(images.join(folder)).unwrap();
            for side in ['l', 'r'] {
                for seq in 0..2u32 {
                    let (w, h) = (40u32, 30u32);
                    let mut px = vec![0u8; (w * h) as usize];
                    for (i, p) in px.iter_mut().enumerate() {
                        let bias = if folder == "brow_up" { 90 } else { 0 }
                            + if side == 'r' { 17 } else { 0 };
                        *p = ((i as u32 * 7 + seq * 40 + bias) % 256) as u8;
                    }
                    let name = format!("{folder}/{folder}_{side}_{seq:08}.png");
                    write_png(&images.join(&name), w, h, &px);
                    csv.push_str(&format!("{name},{brow},0,0\n"));
                }
            }
        }
        std::fs::write(dir.join("labels.csv"), csv).unwrap();
        let backbone_bin = dir.join("brow.bin");
        std::fs::write(&backbone_bin, synth_backbone_bytes(1)).unwrap();

        let bytes = fit_to_bytes(&dir, &backbone_bin, 7).unwrap();
        let mut net = BrowNet::from_bytes(&bytes).expect("fit output loads");
        assert_eq!(net.out_dim(), 1);
        let y = net.forward_one(&vec![0.1f32; 64 * 64]);
        assert_eq!(y.len(), 1);
        assert!(y[0].is_finite(), "forward is finite");

        let backbone_bytes = std::fs::read(&backbone_bin).unwrap();
        let end = conv_region_end();
        assert_eq!(
            &bytes[12..end],
            &backbone_bytes[12..end],
            "frozen conv backbone preserved"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_dataset_reads_labels_and_shapes() {
        let dir = tmp_dir("load_shapes");
        let images = dir.join("images");
        std::fs::create_dir_all(images.join("neutral")).unwrap();
        let n = 5u32;
        let mut csv = String::from("filename,brow,inner,outer\n");
        for seq in 0..n {
            let (w, h) = (24u32, 20u32);
            let px: Vec<u8> = (0..w * h).map(|i| ((i + seq) % 200) as u8).collect();
            let name = format!("neutral/neutral_l_{seq:08}.png");
            write_png(&images.join(&name), w, h, &px);
            csv.push_str(&format!("{name},0.0,0,0\n"));
        }
        std::fs::write(dir.join("labels.csv"), csv).unwrap();

        let mut backbone = BrowNet::from_bytes(&synth_backbone_bytes(1)).unwrap();
        let ds = load_dataset(&dir, &mut backbone).unwrap();
        assert_eq!(ds.features.len(), n as usize);
        assert_eq!(ds.labels.len(), n as usize);
        assert!(
            ds.features.iter().all(|f| f.len() == 1024),
            "features are 1024-d"
        );
        assert!(
            ds.labels.iter().all(|l| l.len() == 1),
            "labels match out_dim"
        );
        assert_eq!(ds.rows, n as usize);
        assert_eq!(ds.skipped, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_dataset_errs_on_missing_labels() {
        let dir = tmp_dir("no_labels");
        let mut backbone = BrowNet::from_bytes(&synth_backbone_bytes(1)).unwrap();
        assert!(
            load_dataset(&dir, &mut backbone).is_err(),
            "missing labels.csv -> Err"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fit_to_bytes_errs_without_backbone() {
        let dir = tmp_dir("no_backbone");
        let missing = dir.join("nope.bin");
        assert!(
            fit_to_bytes(&dir, &missing, 0).is_err(),
            "nonexistent backbone -> Err"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
