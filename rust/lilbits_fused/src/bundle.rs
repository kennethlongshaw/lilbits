//! Load a `.lbm` bundle (`lilbits.inference.bundle.write`) and run it without Python: a plain LeanLM
//! (`fused.export` arrays) or a SpaceHNet (`fused_hnet.export` arrays, recognised by `hmeta`).

use super::hnet::HModel;
use super::*;
use std::collections::HashMap;

struct Entry {
    dtype: String,
    off: usize,
    len: usize,
}

pub struct Bundle {
    buf: Vec<u64>, // 8-byte aligned copy of the file; arrays start at 64-byte aligned offsets
    map: HashMap<String, Entry>,
}

impl Bundle {
    pub fn open(path: &str) -> Result<Bundle, String> {
        use std::io::Read;
        // read straight into an 8-byte aligned buffer (one copy of the weights in memory, not two)
        let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let n = f.metadata().map_err(|e| format!("{path}: {e}"))?.len() as usize;
        let mut buf = vec![0u64; n.div_ceil(8)];
        // Safety: buf owns at least n bytes; u8 has no invalid bit patterns.
        let raw = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n) };
        f.read_exact(raw).map_err(|e| format!("{path}: {e}"))?;
        let end = raw
            .windows(2)
            .position(|w| w == b"\n\n")
            .ok_or("not an .lbm bundle (no header end)")?;
        let head = std::str::from_utf8(&raw[..end]).map_err(|_| "bad header")?;
        let mut lines = head.lines();
        if lines.next() != Some("LBM1") {
            return Err("not an .lbm bundle".into());
        }
        let n: usize = lines.next().and_then(|l| l.parse().ok()).ok_or("bad count")?;
        let mut map = HashMap::new();
        for line in lines.take(n) {
            let f: Vec<&str> = line.split(' ').collect();
            if f.len() != 4 {
                return Err(format!("bad header line {line:?}"));
            }
            let (off, len) = (
                f[2].parse().map_err(|_| "bad offset")?,
                f[3].parse().map_err(|_| "bad size")?,
            );
            if off % 8 != 0 || off + len > raw.len() {
                return Err(format!("{}: out of range", f[0]));
            }
            map.insert(
                f[0].to_string(),
                Entry {
                    dtype: f[1].to_string(),
                    off,
                    len,
                },
            );
        }
        Ok(Bundle { buf, map })
    }

    pub fn has(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }

    fn ptr(&self, k: &str) -> Result<*const c_void, String> {
        let e = self.map.get(k).ok_or(format!("missing array {k}"))?;
        Ok(unsafe { (self.buf.as_ptr() as *const u8).add(e.off) } as *const c_void)
    }

    fn meta(&self, k: &str) -> Result<Vec<i64>, String> {
        let e = self.map.get(k).ok_or(format!("missing array {k}"))?;
        if e.dtype != "int64" {
            return Err(format!("{k}: expected int64"));
        }
        let p = self.ptr(k)? as *const i64;
        Ok(unsafe { core::slice::from_raw_parts(p, e.len / 8) }.to_vec())
    }

    pub fn bytes(&self) -> usize {
        self.map.values().map(|e| e.len).sum()
    }

    fn mat(&self, name: &str, rows: i64, cols: i64, cpad: i64, fmt: i64) -> Result<Mat, String> {
        let s = if self.has(&format!("{name}.s")) {
            self.ptr(&format!("{name}.s"))? as *const f32
        } else {
            core::ptr::null()
        };
        let rpad = if self.has(&format!("{name}.rpad")) {
            self.meta(&format!("{name}.rpad"))?[0]
        } else {
            0
        };
        Ok(Mat {
            rows: rows as i32,
            cols: cols as i32,
            cpad: cpad as i32,
            fmt: fmt as i32,
            w: self.ptr(&format!("{name}.w"))?,
            s,
            rpad: rpad as i32,
        })
    }
}

fn empty_mat() -> Mat {
    Mat {
        rows: 0,
        cols: 0,
        cpad: 0,
        fmt: 0,
        w: core::ptr::null(),
        s: core::ptr::null(),
        rpad: 0,
    }
}

/// A model built over a bundle's arrays (the bundle must outlive it).
pub enum Engine {
    Plain { m: Box<Model>, _layers: Vec<Layer> },
    HNet { m: Box<HModel>, _layers: [Vec<Layer>; 3] },
}

pub struct Session {
    pub engine: Engine,
    pub state: Vec<f32>,
    pub zd: Vec<f32>,
    pub bst: [i32; 3],
    pub logits: Vec<f32>,
    pub work: Vec<f32>,
    pub main_steps: i64,
    block: i32,
}

/// (model, layers) for one stack named `p` (`""` = plain LeanLM keys "l{i}.*", else "{p}.l{i}.*" + "{p}.meta").
fn stack(b: &Bundle, p: &str, with_io: bool) -> Result<(Model, Vec<Layer>), String> {
    let pre = if p.is_empty() { String::new() } else { format!("{p}.") };
    let meta = b.meta(&format!("{pre}meta"))?;
    let (d, dp, h, dh, hid, hp, l, v, fmt) = (
        meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], meta[6], meta[7], meta[8],
    );
    let mut layers = Vec::with_capacity(l as usize);
    for i in 0..l {
        let q = format!("{pre}l{i}");
        layers.push(Layer {
            in_: b.mat(&format!("{q}.in"), 4 * d + 2 * h, d, dp, fmt)?,
            o: b.mat(&format!("{q}.o"), d, d, dp, fmt)?,
            up: b.mat(&format!("{q}.up"), 2 * hid, d, dp, fmt)?,
            down: b.mat(&format!("{q}.down"), d, hid, hp, fmt)?,
            neg_a: b.ptr(&format!("{q}.negA"))? as *const f32,
            dt_bias: b.ptr(&format!("{q}.dt_bias"))? as *const f32,
        });
    }
    let emb_fmt = if fmt == 4 { 2 } else { fmt }; // embedding lookups read int8 rows
    let (emb, head) = if with_io {
        (b.mat("emb", v, d, dp, emb_fmt)?, b.mat("head", v, d, dp, fmt)?)
    } else {
        (empty_mat(), empty_mat())
    };
    let m = Model {
        d: d as i32,
        dpad: dp as i32,
        h: h as i32,
        dh: dh as i32,
        hid: hid as i32,
        hpad: hp as i32,
        l: l as i32,
        vocab: v as i32,
        eps: f32::EPSILON,
        emb,
        head,
        layers: layers.as_ptr(),
    };
    Ok((m, layers))
}

impl Session {
    pub fn new(b: &Bundle, block: i32) -> Result<Session, String> {
        if b.has("hmeta") {
            let hm = b.meta("hmeta")?;
            let (enc, le) = stack(b, "enc", true)?;
            let (main, lm) = stack(b, "main", false)?;
            let (dec, ld) = stack(b, "dec", false)?;
            let fmt = hm[2];
            let (do_, po, dm, pm) = (enc.d as i64, enc.dpad as i64, main.d as i64, main.dpad as i64);
            let m = Box::new(HModel::new(
                enc,
                main,
                dec,
                b.mat("up", dm, do_, po, fmt)?,
                b.mat("down", do_, dm, pm, fmt)?,
                b.mat("res", do_, do_, po, fmt)?,
                hm[0] as i32,
                hm[1] as i32,
            ));
            let nw = unsafe { hnet::hnet_work_floats(&*m, block) } as usize;
            let ns = unsafe { hnet::hnet_state_floats(&*m) } as usize;
            Ok(Session {
                state: vec![0.0; ns],
                zd: vec![0.0; po as usize],
                bst: [-1, -1, 0],
                logits: vec![0.0; 256],
                work: vec![0.0; nw],
                main_steps: 0,
                block,
                engine: Engine::HNet {
                    m,
                    _layers: [le, lm, ld],
                },
            })
        } else {
            let (m, layers) = stack(b, "", true)?;
            let m = Box::new(m);
            let nw = unsafe { work_floats(&*m, block) } as usize;
            let ns = (m.l * m.h * m.dh * m.dh) as usize;
            Ok(Session {
                state: vec![0.0; ns],
                zd: Vec::new(),
                bst: [-1, -1, 0],
                logits: vec![0.0; m.vocab as usize],
                work: vec![0.0; nw],
                main_steps: 0,
                block,
                engine: Engine::Plain { m, _layers: layers },
            })
        }
    }

    pub fn weight_floats_state(&self) -> usize {
        (self.state.len() + self.zd.len()) * 4
    }

    /// Feed bytes; self.logits = next-byte logits after the last one.
    pub fn feed(&mut self, toks: &[i32]) {
        let t = toks.len() as i32;
        match &self.engine {
            Engine::Plain { m, .. } => unsafe {
                forward(
                    &**m,
                    toks.as_ptr(),
                    t,
                    self.state.as_mut_ptr(),
                    self.logits.as_mut_ptr(),
                    0,
                    self.work.as_mut_ptr(),
                    self.block,
                    1,
                )
            },
            Engine::HNet { m, .. } => {
                self.main_steps += unsafe {
                    hnet::hnet_forward(
                        &**m,
                        toks.as_ptr(),
                        t,
                        self.state.as_mut_ptr(),
                        self.zd.as_mut_ptr(),
                        self.bst.as_mut_ptr(),
                        self.logits.as_mut_ptr(),
                        0,
                        self.work.as_mut_ptr(),
                        self.block,
                    )
                }
            }
        }
    }

    /// Greedy: feed `first`, then each argmax, n times (native loop). Returns the n bytes.
    pub fn greedy(&mut self, first: i32, n: usize) -> Vec<i32> {
        let mut out = vec![0i32; n];
        match &self.engine {
            Engine::Plain { m, .. } => unsafe {
                generate(
                    &**m,
                    first,
                    n as i32,
                    self.state.as_mut_ptr(),
                    self.logits.as_mut_ptr(),
                    self.work.as_mut_ptr(),
                    out.as_mut_ptr(),
                    1,
                )
            },
            Engine::HNet { m, .. } => {
                self.main_steps += unsafe {
                    hnet::hnet_generate(
                        &**m,
                        first,
                        n as i32,
                        self.state.as_mut_ptr(),
                        self.zd.as_mut_ptr(),
                        self.bst.as_mut_ptr(),
                        self.logits.as_mut_ptr(),
                        self.work.as_mut_ptr(),
                        out.as_mut_ptr(),
                    )
                }
            }
        }
        out
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.zd.fill(0.0);
        self.bst = [-1, -1, 0];
        self.main_steps = 0;
    }
}
