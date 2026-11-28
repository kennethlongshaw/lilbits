//! Standalone CLI: run a `.lbm` bundle (plain GDN LM or SpaceHNet) with no Python.
//!
//!     lilbits MODEL.lbm [--prompt TEXT] [-n BYTES] [--temperature T] [--top-k K] [--seed S]
//!     lilbits MODEL.lbm --bench [--n_gen N] [--rounds R]      # one JSON line: prefill / decode speed, RSS
//!
//! Greedy decoding (--temperature 0) runs the engine's native loop; sampling feeds one byte per call.

use lilbits_fused::bundle::{Bundle, Session};
use std::io::Write;
use std::time::Instant;

const BENCH_PROMPT: &str = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. ";

fn status_kib(field: &str) -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::NAN)
}

fn cpu_s() -> f64 {
    // utime + stime of this process, in clock ticks (100 Hz on Linux)
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let t = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (t(11) + t(12)) / 100.0
}

fn argmax(v: &[f32]) -> i32 {
    v.iter().enumerate().fold(0, |b, (j, &x)| if x > v[b] { j } else { b }) as i32
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample(logits: &[f32], temp: f64, top_k: usize, rng: &mut Rng) -> i32 {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let k = if top_k == 0 { idx.len() } else { top_k.min(idx.len()) };
    let mx = logits[idx[0]] as f64;
    let p: Vec<f64> = idx[..k]
        .iter()
        .map(|&i| ((logits[i] as f64 - mx) / temp).exp())
        .collect();
    let mut u = rng.f() * p.iter().sum::<f64>();
    for (j, &pj) in p.iter().enumerate() {
        u -= pj;
        if u <= 0.0 {
            return idx[j] as i32;
        }
    }
    idx[k - 1] as i32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "-h" || args[1] == "--help" {
        eprintln!("usage: lilbits MODEL.lbm [--prompt TEXT] [-n N] [--temperature T] [--top-k K] [--seed S] [--bench]");
        std::process::exit(2);
    }
    let (mut prompt, mut n, mut temp, mut top_k, mut seed) = (
        "It was a dark and stormy night; ".to_string(),
        400usize,
        0.8f64,
        40usize,
        1u64,
    );
    let (mut bench, mut n_gen, mut rounds) = (false, 256usize, 5usize);
    let mut i = 2;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--prompt" => prompt = v,
            "-n" => n = v.parse().expect("-n"),
            "--temperature" => temp = v.parse().expect("--temperature"),
            "--top-k" => top_k = v.parse().expect("--top-k"),
            "--seed" => seed = v.parse().expect("--seed"),
            "--gen" => n_gen = v.parse().expect("--n_gen"),
            "--rounds" => rounds = v.parse().expect("--rounds"),
            "--bench" => {
                bench = true;
                i += 1;
                continue;
            }
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    let rss0 = status_kib("VmRSS:");
    let b = Bundle::open(&args[1]).unwrap_or_else(|e| panic!("{e}"));
    let mut s = Session::new(&b, 64).unwrap_or_else(|e| panic!("{e}"));
    if bench {
        let p: Vec<i32> = BENCH_PROMPT.bytes().map(|c| c as i32).collect();
        s.feed(&p[..32]);
        s.greedy(argmax(&s.logits), 8);
        let mut best = (f64::INFINITY, 0.0, 0.0, 0.0, 0i64);
        for _ in 0..rounds {
            s.reset();
            let (w, c) = (Instant::now(), cpu_s());
            s.feed(&p);
            let (pw, pc) = (w.elapsed().as_secs_f64(), cpu_s() - c);
            let steps0 = s.main_steps;
            let (w, c) = (Instant::now(), cpu_s());
            s.greedy(argmax(&s.logits), n_gen);
            let (dw, dc) = (w.elapsed().as_secs_f64(), cpu_s() - c);
            if dw < best.0 {
                best = (dw, dc, pw, pc, s.main_steps - steps0);
            }
        }
        let (dw, dc, pw, _pc, ms) = best;
        println!(
            "{{\"decode_ms_per_byte\": {:.4}, \"decode_bytes_per_s\": {:.0}, \"decode_cpu_ms_per_byte\": {:.4}, \
             \"prefill_bytes_per_s\": {:.0}, \"main_steps_per_byte\": {:.4}, \"bundle_bytes\": {}, \
             \"state_bytes\": {}, \"rss_mib\": {:.2}, \"peak_rss_mib\": {:.2}, \"rss_before_model_mib\": {:.2}}}",
            dw / n_gen as f64 * 1e3,
            n_gen as f64 / dw,
            dc / n_gen as f64 * 1e3,
            p.len() as f64 / pw,
            ms as f64 / n_gen as f64,
            b.bytes(),
            s.weight_floats_state(),
            status_kib("VmRSS:") / 1024.0,
            status_kib("VmHWM:") / 1024.0,
            rss0 / 1024.0
        );
        return;
    }
    let p: Vec<i32> = prompt.bytes().map(|c| c as i32).collect();
    s.feed(&p);
    let t = Instant::now();
    let out: Vec<i32> = if temp <= 0.0 {
        let first = argmax(&s.logits);
        let mut o = vec![first];
        o.extend(s.greedy(first, n.saturating_sub(1)));
        o.truncate(n);
        o
    } else {
        let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        let mut o = Vec::with_capacity(n);
        for _ in 0..n {
            let tok = sample(&s.logits, temp, top_k, &mut rng);
            o.push(tok);
            s.feed(&[tok]);
        }
        o
    };
    let dt = t.elapsed().as_secs_f64();
    let bytes: Vec<u8> = out.iter().map(|&x| x as u8).collect();
    let mut so = std::io::stdout();
    so.write_all(prompt.as_bytes()).unwrap();
    so.write_all(&bytes).unwrap();
    so.write_all(b"\n").unwrap();
    eprintln!(
        "[{} bytes in {:.0} ms, {:.0} B/s, RSS {:.1} MiB]",
        n,
        dt * 1e3,
        n as f64 / dt,
        status_kib("VmRSS:") / 1024.0
    );
}
