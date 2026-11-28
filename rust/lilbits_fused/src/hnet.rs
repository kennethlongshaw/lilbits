//! Space-patched H-Net (`lilbits.models.space_hnet.SpaceHNet`): three GDN stacks chained.
//!
//! ```text
//! byte -> emb -> ENCODER stack (d_outer) = h
//!      -> on a patch boundary: up(rms(h)) -> MAIN stack (d_main) -> zd = down(rms(z))   (once per patch)
//!      -> y = zd + res(h) -> DECODER stack (d_outer) -> rms -> head -> logits
//! ```
//!
//! Norm gains are folded into up / down / head by `lilbits.inference.fused_hnet.export`. Boundaries follow
//! `space_hnet.boundaries`: a byte is space-like unless it is an ASCII letter / digit or a UTF-8 continuation;
//! align 0 ("end") steps the main stack on the first space-like byte after a word, align 1 ("start") on the byte
//! after it; a step is forced every max_patch bytes after the last one. Prefill runs each stack on whole blocks of
//! bytes (the main stack on the block's boundary rows only). Batch 1, one thread.

use super::*;

#[repr(C)]
pub struct HModel {
    enc: Model,
    main: Model,
    dec: Model,
    up: Mat,
    down: Mat,
    res: Mat,
    align: i32,
    max_patch: i32,
}

impl HModel {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        enc: Model,
        main: Model,
        dec: Model,
        up: Mat,
        down: Mat,
        res: Mat,
        align: i32,
        max_patch: i32,
    ) -> Self {
        HModel {
            enc,
            main,
            dec,
            up,
            down,
            res,
            align,
            max_patch,
        }
    }
}

pub struct HModelV<'a> {
    pub enc: ModelV<'a>,
    pub main: ModelV<'a>,
    pub dec: ModelV<'a>,
    pub up: MatV<'a>,
    pub down: MatV<'a>,
    pub res: MatV<'a>,
    pub align: i32,
    pub max_patch: usize,
}

/// Layers-only view: emb / head of `m` are not read (the encoder's are used).
unsafe fn view_stack<'a>(m: &'a Model, emb: MatV<'a>, head: MatV<'a>) -> ModelV<'a> {
    let (h, l) = (m.h as usize, m.l as usize);
    let layers = unsafe { core::slice::from_raw_parts(m.layers, l) };
    let layers = layers
        .iter()
        .map(|ly| unsafe {
            LayerV {
                inp: view_mat(&ly.in_),
                o: view_mat(&ly.o),
                up: view_mat(&ly.up),
                down: view_mat(&ly.down),
                neg_a: core::slice::from_raw_parts(ly.neg_a, h),
                dt_bias: core::slice::from_raw_parts(ly.dt_bias, h),
            }
        })
        .collect();
    ModelV {
        d: m.d as usize,
        dp: m.dpad as usize,
        h,
        dh: m.dh as usize,
        hid: m.hid as usize,
        hp: m.hpad as usize,
        vocab: m.vocab as usize,
        eps: m.eps,
        emb,
        head,
        layers,
    }
}

unsafe fn view<'a>(m: &'a HModel) -> HModelV<'a> {
    let enc = unsafe { view_model(&m.enc) };
    let (emb, head) = (enc.emb, enc.head);
    HModelV {
        main: unsafe { view_stack(&m.main, emb, head) },
        dec: unsafe { view_stack(&m.dec, emb, head) },
        enc,
        up: unsafe { view_mat(&m.up) },
        down: unsafe { view_mat(&m.down) },
        res: unsafe { view_mat(&m.res) },
        align: m.align,
        max_patch: m.max_patch.max(0) as usize,
    }
}

fn state_len(m: &ModelV) -> usize {
    m.layers.len() * m.h * m.dh * m.dh
}

pub fn hnet_state_len(m: &HModelV) -> usize {
    state_len(&m.enc) + state_len(&m.main) + state_len(&m.dec)
}

fn rest_len(m: &ModelV, bt: usize) -> usize {
    work_len(m, bt) - bt * m.dp
}

pub fn hnet_work_len(m: &HModelV, bt: usize) -> usize {
    let (po, pm) = (m.enc.dp, m.main.dp);
    let rest = rest_len(&m.enc, bt).max(rest_len(&m.dec, bt));
    // h | y | zd rows | enc/dec rest | norm rows (outer) | main residual | main rest | norm rows (main)
    3 * bt * po + rest + bt * po + bt * pm + rest_len(&m.main, bt) + bt * pm + 64
}

fn spacelike(b: i32) -> bool {
    let alnum = (48..=57).contains(&b) || (65..=90).contains(&b) || (97..=122).contains(&b);
    !(alnum || (0x80..0xC0).contains(&b))
}

/// Boundary state carried between calls: [previous byte space-like (-1 none / 0 / 1), the one before, bytes since
/// the last boundary].
fn is_boundary(m: &HModelV, byte: i32, bst: &mut [i32; 3]) -> bool {
    let sp = spacelike(byte) as i32;
    let mut b = if bst[0] < 0 {
        true
    } else if m.align == 0 {
        sp == 1 && bst[0] == 0
    } else {
        bst[0] == 1 && bst[1] != 1
    };
    if !b && m.max_patch > 0 && bst[2] + 1 >= m.max_patch as i32 {
        b = true;
    }
    bst[1] = bst[0];
    bst[0] = sp;
    bst[2] = if b { 0 } else { bst[2] + 1 };
    b
}

pub struct HRun<'a> {
    pub toks: &'a [i32],
    pub state: &'a mut [f32],
    /// the current patch's down(rms(z)) row (d_outer padded), carried between calls
    pub zd: &'a mut [f32],
    pub bst: &'a mut [i32; 3],
    pub logits: &'a mut [f32],
    pub all_logits: bool,
    pub work: &'a mut [f32],
    pub bt: usize,
    /// filled with the number of main-stack steps taken
    pub main_steps: usize,
}

#[inline(always)]
fn hnet_impl<S: Simd>(simd: S, m: &HModelV, r: &mut HRun) {
    let (e, mn, dc) = (&m.enc, &m.main, &m.dec);
    let (d, po, dm, pm, v, eps) = (e.d, e.dp, mn.d, mn.dp, e.vocab, e.eps);
    let bt_max = r.bt;
    let (se, rest) = r.state.split_at_mut(state_len(e));
    let (sm, sd) = rest.split_at_mut(state_len(mn));
    let w = &mut *r.work;
    let (hb, w) = w.split_at_mut(bt_max * po);
    let (yb, w) = w.split_at_mut(bt_max * po);
    let (zb, w) = w.split_at_mut(bt_max * po);
    let (rest_o, w) = w.split_at_mut(rest_len(e, bt_max).max(rest_len(dc, bt_max)));
    let (no, w) = w.split_at_mut(bt_max * po);
    let (xm, w) = w.split_at_mut(bt_max * pm);
    let (rest_m, w) = w.split_at_mut(rest_len(mn, bt_max));
    let (nm, _) = w.split_at_mut(bt_max * pm);
    let tt = r.toks.len();
    let mut rows = Vec::with_capacity(bt_max);
    let mut t0 = 0;
    while t0 < tt {
        let bt = (tt - t0).min(bt_max);
        // encoder over the block
        for t in 0..bt {
            get_row(&e.emb, r.toks[t0 + t] as usize, &mut hb[t * po..(t + 1) * po]);
        }
        stack_impl(simd, e, hb, rest_o, bt_max, bt, se, 1);
        // main stack on the block's boundary rows
        rows.clear();
        for t in 0..bt {
            if is_boundary(m, r.toks[t0 + t], r.bst) {
                rows.push(t);
            }
        }
        let nb = rows.len();
        if nb > 0 {
            for (i, &t) in rows.iter().enumerate() {
                rms(&hb[t * po..(t + 1) * po], &mut no[i * po..(i + 1) * po], d, eps);
            }
            let inp = MatIn {
                x: no,
                ldx: po,
                t: nb,
                cols: d,
            };
            matmul(simd, &m.up, inp, xm, pm, false, 1);
            stack_impl(simd, mn, xm, rest_m, bt_max, nb, sm, 1);
            for i in 0..nb {
                rms(&xm[i * pm..(i + 1) * pm], &mut nm[i * pm..(i + 1) * pm], dm, eps);
            }
            let inp = MatIn {
                x: nm,
                ldx: pm,
                t: nb,
                cols: dm,
            };
            matmul(simd, &m.down, inp, zb, po, false, 1);
            r.main_steps += nb;
        }
        // decoder input: y = zd(patch of the byte) + res(h)
        let inp = MatIn {
            x: hb,
            ldx: po,
            t: bt,
            cols: d,
        };
        matmul(simd, &m.res, inp, yb, po, false, 1);
        let mut next = 0;
        for t in 0..bt {
            if next < nb && rows[next] == t {
                r.zd[..d].copy_from_slice(&zb[next * po..next * po + d]);
                next += 1;
            }
            for (y, z) in yb[t * po..t * po + d].iter_mut().zip(&r.zd[..d]) {
                *y += z;
            }
        }
        stack_impl(simd, dc, yb, rest_o, bt_max, bt, sd, 1);
        if r.all_logits {
            for t in 0..bt {
                rms(&yb[t * po..(t + 1) * po], &mut no[t * po..(t + 1) * po], d, eps);
            }
            let inp = MatIn {
                x: no,
                ldx: po,
                t: bt,
                cols: d,
            };
            matmul(simd, &e.head, inp, &mut r.logits[t0 * v..], v, false, 1);
        } else if t0 + bt == tt {
            rms(&yb[(bt - 1) * po..bt * po], &mut no[..po], d, eps);
            let inp = MatIn {
                x: no,
                ldx: po,
                t: 1,
                cols: d,
            };
            matmul(simd, &e.head, inp, &mut r.logits[..v], v, false, 1);
        }
        t0 += bt;
    }
}

pub fn hnet_forward_safe(level: Level, m: &HModelV, r: &mut HRun) {
    dispatch!(level, simd => hnet_impl(simd, m, r));
}

// ---------------------------------------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------------------------------------

/// # Safety
/// `m` must come from `lilbits.inference.fused_hnet` (FusedHNet builds it).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnet_work_floats(m: *const HModel, bt: i32) -> i64 {
    let mv = unsafe { view(&*m) };
    hnet_work_len(&mv, bt.max(1) as usize) as i64
}

/// # Safety
/// As `hnet_work_floats`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hnet_state_floats(m: *const HModel) -> i64 {
    let mv = unsafe { view(&*m) };
    hnet_state_len(&mv) as i64
}

/// Run t bytes from the carried state. Returns the number of main-stack steps taken.
///
/// # Safety
/// toks[t], state[hnet_state_floats], zd[enc dpad], bst[3], logits[(all ? t : 1) * vocab], work[hnet_work_floats(bt)].
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hnet_forward(
    m: *const HModel,
    toks: *const i32,
    t: i32,
    state: *mut f32,
    zd: *mut f32,
    bst: *mut i32,
    logits: *mut f32,
    all_logits: i32,
    work: *mut f32,
    bt: i32,
) -> i64 {
    let mv = unsafe { view(&*m) };
    let (t, bt) = (t.max(0) as usize, bt.max(1) as usize);
    let nl = if all_logits != 0 {
        t * mv.enc.vocab
    } else {
        mv.enc.vocab
    };
    let mut r = unsafe {
        HRun {
            toks: core::slice::from_raw_parts(toks, t),
            state: core::slice::from_raw_parts_mut(state, hnet_state_len(&mv)),
            zd: core::slice::from_raw_parts_mut(zd, mv.enc.dp),
            bst: &mut *(bst as *mut [i32; 3]),
            logits: core::slice::from_raw_parts_mut(logits, nl),
            all_logits: all_logits != 0,
            work: core::slice::from_raw_parts_mut(work, hnet_work_len(&mv, bt)),
            bt,
            main_steps: 0,
        }
    };
    hnet_forward_safe(isa::level(isa::current()), &mv, &mut r);
    r.main_steps as i64
}

/// Greedy decode: feed `first`, write the argmax to out[0], feed it back, ... (n one-byte steps, like `generate`).
/// Returns main-stack steps.
///
/// # Safety
/// As `hnet_forward` with bt = 1; out[n].
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hnet_generate(
    m: *const HModel,
    first: i32,
    n: i32,
    state: *mut f32,
    zd: *mut f32,
    bst: *mut i32,
    logits: *mut f32,
    work: *mut f32,
    out: *mut i32,
) -> i64 {
    let mv = unsafe { view(&*m) };
    let level = isa::level(isa::current());
    let (state, zd, bst, logits, work, out) = unsafe {
        (
            core::slice::from_raw_parts_mut(state, hnet_state_len(&mv)),
            core::slice::from_raw_parts_mut(zd, mv.enc.dp),
            &mut *(bst as *mut [i32; 3]),
            core::slice::from_raw_parts_mut(logits, mv.enc.vocab),
            core::slice::from_raw_parts_mut(work, hnet_work_len(&mv, 1)),
            core::slice::from_raw_parts_mut(out, n.max(0) as usize),
        )
    };
    let mut steps = 0;
    let mut tok = [first];
    for o in out.iter_mut() {
        let mut r = HRun {
            toks: &tok,
            state: &mut *state,
            zd: &mut *zd,
            bst: &mut *bst,
            logits: &mut *logits,
            all_logits: false,
            work: &mut *work,
            bt: 1,
            main_steps: 0,
        };
        hnet_forward_safe(level, &mv, &mut r);
        steps += r.main_steps;
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |b, (j, &x)| if x > logits[b] { j } else { b });
        *o = best as i32;
        tok[0] = best as i32;
    }
    steps as i64
}
