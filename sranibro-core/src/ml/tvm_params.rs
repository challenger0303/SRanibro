
use std::collections::HashMap;
use std::io;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor {
    #[allow(dead_code)]
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
        if n > self.b.len() - self.p {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated TVM params"))
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

pub fn parse(path: &str) -> io::Result<Vec<Tensor>> {
    let data = std::fs::read(path)?;
    let mut r = Reader { b: &data, p: 0 };

    let dbg = std::env::var("TVM_DEBUG").is_ok();
    let magic = r.u64()?;
    let _reserved = r.u64()?;
    let n = r.u64()? as usize;
    if dbg {
        eprintln!("[dbg] file={} bytes  magic={:#018x}  n={}", data.len(), magic, n);
    }

    let mut names = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        let len = r.u64()? as usize;
        let name = String::from_utf8_lossy(r.bytes(len)?).into_owned();
        names.push(name);
    }
    if dbg {
        eprintln!("[dbg] names={:?}  pos_after_names={}", names, r.p);
    }

    let _arrays_count = r.u64()?;

    let mut out = Vec::with_capacity(n.min(1 << 16));
    for name in names {
        let pos0 = r.p;
        let magic2 = r.u64()?;
        let _reserved = r.u64()?;
        let _dev_type = r.u32()?;
        let _dev_id = r.u32()?;
        let ndim = r.u32()? as usize;
        let dtype_code = r.u8()?;
        let dtype_bits = r.u8()?;
        let _lanes = r.u16()?;
        if dbg {
            eprintln!("[dbg] array '{}' pos0={} magic2={:#018x} ndim={} dtype={}/{}",
                      name, pos0, magic2, ndim, dtype_code, dtype_bits);
        }
        let mut shape = Vec::with_capacity(ndim.min(64));
        for _ in 0..ndim {
            shape.push(r.i64()?);
        }
        let data_size = r.u64()? as usize;
        if dbg {
            eprintln!("[dbg]   shape={:?} data_size={} (pos before data={})", shape, data_size, r.p);
        }
        let raw = r.bytes(data_size)?;
        let floats: Vec<f32> = if dtype_code == 2 && dtype_bits == 32 {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        } else {
            Vec::new()
        };
        out.push(Tensor { name, shape, data: floats });
    }
    Ok(out)
}

pub fn parse_map(path: &str) -> io::Result<HashMap<String, Tensor>> {
    Ok(parse(path)?.into_iter().map(|t| (t.name.clone(), t)).collect())
}
