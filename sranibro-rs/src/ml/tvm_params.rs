//! Parser for SRanipal's TVM params file (`*.params_opencl.params`).
//!
//! Distribution note: we ship this parser (our reverse-engineering), NOT the
//! weights. At runtime the user points us at their SRanipal install directory
//! and we read the proprietary weights they already own.
//!
//! Binary format (little-endian), reverse-engineered in `extract_weights.py`:
//! ```text
//! magic:u64  reserved:u64  num_params:u64
//! names[n]:  { name_len:u64, name:[u8; name_len] (utf-8) }
//! arrays[n]: { reserved:u64, dev_type:u32, dev_id:u32, ndim:u32,
//!              dtype_code:u8 (2=float), dtype_bits:u8, dtype_lanes:u16,
//!              shape:[i64; ndim], data_size:u64, data:[u8; data_size] }  // row-major
//! ```

use std::collections::HashMap;
use std::io;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor {
    #[allow(dead_code)] // shape helper; used by validation/tooling
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn need(&self, n: usize) -> io::Result<()> {
        // `self.p <= self.b.len()` always holds (p only advances past a checked need),
        // so `len - p` can't underflow. Comparing remaining-vs-n avoids the `p + n`
        // overflow a garbage length field would otherwise cause (which wrapped the
        // bounds check and panicked on the slice).
        if n > self.b.len() - self.p {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated TVM params",
            ))
        } else {
            Ok(())
        }
    }
    fn u64(&mut self) -> io::Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        Ok(v)
    }
    fn u32(&mut self) -> io::Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        Ok(v)
    }
    fn u16(&mut self) -> io::Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.b[self.p..self.p + 2].try_into().unwrap());
        self.p += 2;
        Ok(v)
    }
    fn u8(&mut self) -> io::Result<u8> {
        self.need(1)?;
        let v = self.b[self.p];
        self.p += 1;
        Ok(v)
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
}

/// Parse a TVM params file into ordered tensors.
pub fn parse(path: &str) -> io::Result<Vec<Tensor>> {
    let data = std::fs::read(path)?;
    let mut r = Reader { b: &data, p: 0 };

    let dbg = std::env::var("TVM_DEBUG").is_ok();
    let magic = r.u64()?;
    let _reserved = r.u64()?;
    let n = r.u64()? as usize;
    if dbg {
        eprintln!(
            "[dbg] file={} bytes  magic={:#018x}  n={}",
            data.len(),
            magic,
            n
        );
    }

    // Cap the pre-allocation: a garbage `n` (huge u64 from a wrong/corrupt file) would
    // otherwise abort on a giant allocation. Legit models have ~10 params; the bounded
    // per-item reads below return Err gracefully if `n` is bogus.
    let mut names = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        let len = r.u64()? as usize;
        let name = String::from_utf8_lossy(r.bytes(len)?).into_owned();
        names.push(name);
    }
    if dbg {
        eprintln!("[dbg] names={:?}  pos_after_names={}", names, r.p);
    }

    // After the names array comes the NDArray list: a u64 count, then each
    // entry is a full TVM NDArray (magic + reserved + DLTensor fields + data).
    let _arrays_count = r.u64()?;

    let mut out = Vec::with_capacity(n.min(1 << 16));
    for name in names {
        let pos0 = r.p;
        let magic2 = r.u64()?; // kTVMNDArrayMagic 0xDD5E40F096B4A13F
        let _reserved = r.u64()?;
        let _dev_type = r.u32()?;
        let _dev_id = r.u32()?;
        let ndim = r.u32()? as usize;
        let dtype_code = r.u8()?;
        let dtype_bits = r.u8()?;
        let _lanes = r.u16()?;
        if dbg {
            eprintln!(
                "[dbg] array '{}' pos0={} magic2={:#018x} ndim={} dtype={}/{}",
                name, pos0, magic2, ndim, dtype_code, dtype_bits
            );
        }
        let mut shape = Vec::with_capacity(ndim.min(64));
        for _ in 0..ndim {
            shape.push(r.i64()?);
        }
        let data_size = r.u64()? as usize;
        if dbg {
            eprintln!(
                "[dbg]   shape={:?} data_size={} (pos before data={})",
                shape, data_size, r.p
            );
        }
        let raw = r.bytes(data_size)?;
        let floats: Vec<f32> = if dtype_code == 2 && dtype_bits == 32 {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            // Non-f32 tensor (unexpected for this model) — leave empty; caller checks.
            Vec::new()
        };
        out.push(Tensor {
            name,
            shape,
            data: floats,
        });
    }
    Ok(out)
}

/// Parse into a name -> tensor map.
pub fn parse_map(path: &str) -> io::Result<HashMap<String, Tensor>> {
    Ok(parse(path)?
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect())
}
