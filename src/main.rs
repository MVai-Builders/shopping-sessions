use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Date32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use hashbrown::HashMap;
use memmap2::Mmap;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use serde::Deserialize;

// ─── Fast Date & Civil Math ──────────────────────────────────────────────────

#[inline(always)]
fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

#[inline(always)]
fn days_from_civil(mut y: i64, m: i64, d: i64) -> i32 {
    if m <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let y_oe = y.rem_euclid(400);
    let d_oy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    (era * 146097 + y_oe * 365 + y_oe / 4 - y_oe / 100 + d_oy - 719468) as i32
}

#[inline(always)]
fn parse_2_digits(s: &[u8]) -> Option<i64> {
    if s.len() != 2 || !s[0].is_ascii_digit() || !s[1].is_ascii_digit() {
        return None;
    }
    Some((s[0] - b'0') as i64 * 10 + (s[1] - b'0') as i64)
}

#[inline(always)]
fn parse_4_digits(s: &[u8]) -> Option<i64> {
    if s.len() != 4
        || !s[0].is_ascii_digit()
        || !s[1].is_ascii_digit()
        || !s[2].is_ascii_digit()
        || !s[3].is_ascii_digit()
    {
        return None;
    }
    Some(
        (s[0] - b'0') as i64 * 1000
            + (s[1] - b'0') as i64 * 100
            + (s[2] - b'0') as i64 * 10
            + (s[3] - b'0') as i64,
    )
}

#[inline(always)]
fn parse_iso_f1(s: &[u8]) -> Option<i64> {
    if s.len() != 20
        || s[10] != b'T'
        || s[19] != b'Z'
        || s[4] != b'-'
        || s[7] != b'-'
        || s[13] != b':'
        || s[16] != b':'
    {
        return None;
    }
    for &idx in &[0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !s[idx].is_ascii_digit() {
            return None;
        }
    }
    let y = parse_4_digits(&s[0..4])?;
    let m = parse_2_digits(&s[5..7])?;
    let d = parse_2_digits(&s[8..10])?;
    let h = parse_2_digits(&s[11..13])?;
    let mi = parse_2_digits(&s[14..16])?;
    let se = parse_2_digits(&s[17..19])?;

    if !(1..=12).contains(&m) || !(0..=23).contains(&h) || !(0..=59).contains(&mi) || !(0..=59).contains(&se) {
        return None;
    }
    let dim = [0, 31, if is_leap(y) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=dim[m as usize]).contains(&d) {
        return None;
    }
    let ed = days_from_civil(y, m, d) as i64;
    Some((ed * 86400 + h * 3600 + mi * 60 + se) * 1000)
}

#[inline(always)]
fn parse_iso_f2(s: &[u8]) -> Option<i64> {
    if s.len() != 25
        || s[10] != b'T'
        || s[4] != b'-'
        || s[7] != b'-'
        || s[13] != b':'
        || s[16] != b':'
        || s[22] != b':'
    {
        return None;
    }
    let sign = s[19];
    if sign != b'+' && sign != b'-' {
        return None;
    }
    for &idx in &[0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24] {
        if !s[idx].is_ascii_digit() {
            return None;
        }
    }
    let y = parse_4_digits(&s[0..4])?;
    let m = parse_2_digits(&s[5..7])?;
    let d = parse_2_digits(&s[8..10])?;
    let h = parse_2_digits(&s[11..13])?;
    let mi = parse_2_digits(&s[14..16])?;
    let se = parse_2_digits(&s[17..19])?;
    let oh = parse_2_digits(&s[20..22])?;
    let omi = parse_2_digits(&s[23..25])?;

    if !(1..=12).contains(&m)
        || !(0..=23).contains(&h)
        || !(0..=59).contains(&mi)
        || !(0..=59).contains(&se)
        || !(0..=23).contains(&oh)
        || !(0..=59).contains(&omi)
    {
        return None;
    }
    let dim = [0, 31, if is_leap(y) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=dim[m as usize]).contains(&d) {
        return None;
    }
    let ed = days_from_civil(y, m, d) as i64;
    let local_secs = ed * 86400 + h * 3600 + mi * 60 + se;
    let offset_secs = oh * 3600 + omi * 60;
    let utc_secs = if sign == b'+' {
        local_secs - offset_secs
    } else {
        local_secs + offset_secs
    };
    Some(utc_secs * 1000)
}

#[inline(always)]
fn parse_timestamp(s: &[u8]) -> Option<i64> {
    if s.len() == 20 && s[19] == b'Z' {
        return parse_iso_f1(s);
    }
    if s.len() == 25 && (s[19] == b'+' || s[19] == b'-') {
        return parse_iso_f2(s);
    }
    if !s.is_empty() && s.iter().all(|&b| b.is_ascii_digit()) {
        let mut val = 0i64;
        for &b in s {
            val = val.checked_mul(10)?.checked_add((b - b'0') as i64)?;
        }
        return Some(val);
    }
    None
}

// ─── Deterministic FNV-1a Hash ───────────────────────────────────────────────

#[inline(always)]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── Fast Zero-Copy CSV Dialect ──────────────────────────────────────────────

#[inline(always)]
fn unquote_field(mut s: &[u8]) -> Option<&[u8]> {
    if s.starts_with(b"\"") {
        if s.len() < 2 || !s.ends_with(b"\"") {
            return None;
        }
        s = &s[1..s.len() - 1];
    }
    if s.contains(&b'"') {
        return None;
    }
    Some(s)
}

#[inline(always)]
fn parse_line_7<'a>(line: &'a [u8], cols: &mut [&'a [u8]; 7]) -> bool {
    let mut start = 0;
    let mut col_idx = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col_idx >= 6 {
                return false;
            }
            let Some(f) = unquote_field(&line[start..i]) else {
                return false;
            };
            cols[col_idx] = f;
            col_idx += 1;
            start = i + 1;
        }
    }
    if col_idx != 6 {
        return false;
    }
    let Some(f) = unquote_field(&line[start..]) else {
        return false;
    };
    cols[6] = f;
    true
}

#[inline(always)]
fn parse_line_4<'a>(line: &'a [u8], cols: &mut [&'a [u8]; 4]) -> bool {
    let mut start = 0;
    let mut col_idx = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col_idx >= 3 {
                return false;
            }
            let Some(f) = unquote_field(&line[start..i]) else {
                return false;
            };
            cols[col_idx] = f;
            col_idx += 1;
            start = i + 1;
        }
    }
    if col_idx != 3 {
        return false;
    }
    let Some(f) = unquote_field(&line[start..]) else {
        return false;
    };
    cols[3] = f;
    true
}

#[inline(always)]
fn parse_line_2<'a>(line: &'a [u8], cols: &mut [&'a [u8]; 2]) -> bool {
    let mut start = 0;
    let mut col_idx = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if col_idx >= 1 {
                return false;
            }
            let Some(f) = unquote_field(&line[start..i]) else {
                return false;
            };
            cols[col_idx] = f;
            col_idx += 1;
            start = i + 1;
        }
    }
    if col_idx != 1 {
        return false;
    }
    let Some(f) = unquote_field(&line[start..]) else {
        return false;
    };
    cols[1] = f;
    true
}

// ─── DSU for Linked IDs ──────────────────────────────────────────────────────

struct Dsu {
    parent: HashMap<Vec<u8>, Vec<u8>>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: HashMap::with_capacity(500_000),
        }
    }

    fn find(&mut self, x: &[u8]) -> Vec<u8> {
        if let Some(p) = self.parent.get(x) {
            if p.as_slice() == x {
                return p.clone();
            }
            let p_clone = p.clone();
            let root = self.find(&p_clone);
            self.parent.insert(x.to_vec(), root.clone());
            root
        } else {
            x.to_vec()
        }
    }

    fn union(&mut self, a: &[u8], b: &[u8]) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            if ra < rb {
                self.parent.insert(rb, ra);
            } else {
                self.parent.insert(ra, rb);
            }
        }
    }
}

// ─── Compact Zero-Allocation Event Struct ────────────────────────────────────

#[derive(Clone, Copy)]
struct FastEvent<'a> {
    person_hash: u64,
    person: &'a [u8],
    ts: i64,
    price_cents: u32,
    etype: u8,
    category: u8,
}

#[derive(Clone, Copy)]
struct FastClick<'a> {
    person_hash: u64,
    person: &'a [u8],
    cid: &'a [u8],
    ts: i64,
    camp_id: u32,
}

// ─── Main Pipeline ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let (data_dir, output_dir) = if args.len() >= 3 {
        (args[1].clone(), args[2].clone())
    } else {
        ("/data".to_string(), "/output".to_string())
    };

    let manifest_path = Path::new(&data_dir).join("manifest.json");
    let manifest_content = fs::read_to_string(manifest_path)?;
    let manifest: Manifest = serde_json::from_str(&manifest_content)?;

    let w_start = parse_iso_f1(manifest.window_start.as_bytes()).expect("valid window_start");
    let w_end = parse_iso_f1(manifest.window_end.as_bytes()).expect("valid window_end");
    let click_w_start = w_start - 7 * 86400 * 1000;

    // Window day range for fixed-array person_daily
    let w_start_day = w_start.div_euclid(86_400_000) as i32;
    let w_end_day = (w_end - 1).div_euclid(86_400_000) as i32;
    let n_days = (w_end_day - w_start_day + 1) as usize;

    // 1. Categories & Products
    let mut cat_names: Vec<String> = Vec::with_capacity(32);
    cat_names.push("unknown".to_string());
    let unknown_cat_id = 0u8;

    let mut cat_map: HashMap<Vec<u8>, u8> = HashMap::with_capacity(32);
    cat_map.insert(b"unknown".to_vec(), 0u8);

    let mut products: HashMap<Vec<u8>, u8> = HashMap::with_capacity(65536);
    let p_path = Path::new(&data_dir).join("products.csv");
    if p_path.exists() {
        let content = fs::read(&p_path)?;
        let mut cols = [b"".as_slice(); 2];
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line_2(line, &mut cols) && !cols[0].is_empty() {
                let cat_bytes = cols[1];
                let next_id = cat_names.len() as u8;
                let cat_id = *cat_map.entry(cat_bytes.to_vec()).or_insert_with(|| {
                    cat_names.push(String::from_utf8_lossy(cat_bytes).to_string());
                    next_id
                });
                products.insert(cols[0].to_vec(), cat_id);
            }
        }
    }

    // 2. Campaigns
    let mut campaign_names: Vec<(String, String)> = Vec::with_capacity(4096);
    // Index 0 is reserved for "organic"
    campaign_names.push(("organic".to_string(), "organic".to_string()));
    const ORGANIC_CAMP_ID: u32 = 0;

    let mut campaign_map: HashMap<Vec<u8>, u32> = HashMap::with_capacity(4096);
    let c_path = Path::new(&data_dir).join("campaigns.csv");
    if c_path.exists() {
        let content = fs::read(&c_path)?;
        let mut cols = [b"".as_slice(); 2];
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line_2(line, &mut cols) && !cols[0].is_empty() {
                let camp_id = campaign_names.len() as u32;
                campaign_names.push((
                    String::from_utf8_lossy(cols[0]).to_string(),
                    String::from_utf8_lossy(cols[1]).to_string(),
                ));
                campaign_map.insert(cols[0].to_vec(), camp_id);
            }
        }
    }

    let n_campaigns = campaign_names.len(); // includes organic at index 0

    // 3. User IDs & Links DSU
    let mut dsu = Dsu::new();
    let links_pattern = format!("{}/links/*.csv", data_dir);
    let mut link_files: Vec<PathBuf> = glob::glob(&links_pattern)?
        .filter_map(Result::ok)
        .collect();
    link_files.sort();

    for lf in link_files {
        let content = fs::read(lf)?;
        let mut cols = [b"".as_slice(); 2];
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line_2(line, &mut cols) && !cols[0].is_empty() && !cols[1].is_empty() {
                dsu.union(cols[0], cols[1]);
            }
        }
    }

    // Pre-resolve DSU into a shared read-only map: (hash, Arc<[u8]>)
    let mut resolved_links: HashMap<Vec<u8>, (u64, Arc<[u8]>)> = HashMap::with_capacity(dsu.parent.len());
    let keys: Vec<Vec<u8>> = dsu.parent.keys().cloned().collect();
    for k in keys {
        let root = dsu.find(&k);
        let h = fnv1a(&root);
        resolved_links.insert(k, (h, Arc::from(root.into_boxed_slice())));
    }
    let resolved_links = Arc::new(resolved_links);

    // 4. Events: PARALLEL PARSE & FAST SORT DEDUP
    let events_pattern = format!("{}/events/*.csv", data_dir);
    let mut event_files: Vec<PathBuf> = glob::glob(&events_pattern)?
        .filter_map(Result::ok)
        .collect();
    event_files.sort();

    let mmaps: Vec<Mmap> = event_files
        .iter()
        .map(|ef| {
            let file = File::open(ef).expect("open event file");
            unsafe { Mmap::map(&file).expect("mmap event file") }
        })
        .collect();

    #[derive(Clone, Copy)]
    struct EventLoc {
        eid_hash: u64,
        ingest_ms: i64,
        file_idx: u32,
        line_offset: u32,
        line_len: u32,
    }

    const NUM_SHARDS: usize = 64;

    // Phase 4a: Scan files in parallel once into shard buffers
    let thread_shards: Vec<Vec<Vec<EventLoc>>> = mmaps
        .par_iter()
        .enumerate()
        .map(|(f_idx, mmap)| {
            let mut shards: Vec<Vec<EventLoc>> = (0..NUM_SHARDS)
                .map(|_| Vec::with_capacity(mmap.len() / (80 * NUM_SHARDS)))
                .collect();
            let mut cols = [b"".as_slice(); 7];
            let mut start = 0;
            let bytes = mmap.as_ref();
            let mut line_num = 0;

            for (i, &b) in bytes.iter().enumerate() {
                if b == b'\n' {
                    let line = &bytes[start..i];
                    start = i + 1;
                    line_num += 1;
                    if line_num == 1 || line.is_empty() {
                        continue;
                    }

                    if parse_line_7(line, &mut cols) {
                        let eid = cols[0];
                        if !eid.is_empty() {
                            if let Some(ingest_ms) = parse_iso_f1(cols[3]) {
                                let eid_hash = fnv1a(eid);
                                let shard = (eid_hash as usize) % NUM_SHARDS;
                                shards[shard].push(EventLoc {
                                    eid_hash,
                                    ingest_ms,
                                    file_idx: f_idx as u32,
                                    line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                    line_len: line.len() as u32,
                                });
                            }
                        }
                    }
                }
            }
            if start < bytes.len() {
                let line = &bytes[start..];
                if line_num > 0 && !line.is_empty() && parse_line_7(line, &mut cols) {
                    let eid = cols[0];
                    if !eid.is_empty() {
                        if let Some(ingest_ms) = parse_iso_f1(cols[3]) {
                            let eid_hash = fnv1a(eid);
                            let shard = (eid_hash as usize) % NUM_SHARDS;
                            shards[shard].push(EventLoc {
                                eid_hash,
                                ingest_ms,
                                file_idx: f_idx as u32,
                                line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                line_len: line.len() as u32,
                            });
                        }
                    }
                }
            }
            shards
        })
        .collect();

    // Phase 4b: Deduplicate & Validate each shard in parallel
    let mmaps_ref = &mmaps;
    let products_ref = &products;
    let resolved_links_ref = &resolved_links;

    let shard_valid_events: Vec<Vec<FastEvent>> = (0..NUM_SHARDS)
        .into_par_iter()
        .map(|shard_idx| {
            let total_locs: usize = thread_shards.iter().map(|ts| ts[shard_idx].len()).sum();
            let mut locs = Vec::with_capacity(total_locs);
            for ts in &thread_shards {
                locs.extend_from_slice(&ts[shard_idx]);
            }

            // In-place sort by (eid_hash, ingest_ms) — hash collisions are negligible at 2^-64
            locs.sort_unstable_by(|a, b| {
                a.eid_hash
                    .cmp(&b.eid_hash)
                    .then_with(|| a.ingest_ms.cmp(&b.ingest_ms))
            });

            let mut valid = Vec::with_capacity(locs.len());
            let mut i = 0;
            while i < locs.len() {
                let best_loc = locs[i];
                let best_line = &mmaps_ref[best_loc.file_idx as usize][best_loc.line_offset as usize..(best_loc.line_offset + best_loc.line_len) as usize];
                let mut cols_best = [b"".as_slice(); 7];
                parse_line_7(best_line, &mut cols_best);
                let best_eid = cols_best[0];

                let mut j = i + 1;
                while j < locs.len() && locs[j].eid_hash == best_loc.eid_hash {
                    let other_line = &mmaps_ref[locs[j].file_idx as usize][locs[j].line_offset as usize..(locs[j].line_offset + locs[j].line_len) as usize];
                    let mut cols_other = [b"".as_slice(); 7];
                    if parse_line_7(other_line, &mut cols_other) && cols_other[0] == best_eid {
                        j += 1;
                    } else {
                        break;
                    }
                }
                i = j;

                // Validate best_loc
                let uid_bytes = cols_best[1];
                let pid_bytes = cols_best[5];
                if uid_bytes.is_empty() || pid_bytes.is_empty() {
                    continue;
                }

                let etype = match cols_best[4] {
                    b"view" => 1u8,
                    b"cart" => 2u8,
                    b"purchase" => 3u8,
                    _ => continue,
                };

                let mut price_cents = 0u32;
                if etype == 3 {
                    let pr = cols_best[6];
                    if pr.is_empty() || !pr.iter().all(|&b| b.is_ascii_digit()) {
                        continue;
                    }
                    let mut p = 0u64;
                    for &b in pr {
                        p = p * 10 + (b - b'0') as u64;
                        if p > 10_000_000 {
                            break;
                        }
                    }
                    if !(1..=10_000_000).contains(&p) {
                        continue;
                    }
                    price_cents = p as u32;
                }

                let Some(ts) = parse_timestamp(cols_best[2]) else {
                    continue;
                };
                if ts < w_start || ts >= w_end {
                    continue;
                }
                if (best_loc.ingest_ms - ts) > 86400 * 1000 {
                    continue;
                }

                let (person_hash, person_bytes): (u64, &[u8]) = match resolved_links_ref.get(uid_bytes) {
                    Some(root) => (root.0, &root.1),
                    None => (fnv1a(uid_bytes), uid_bytes),
                };

                let category = *products_ref.get(pid_bytes).unwrap_or(&unknown_cat_id);

                valid.push(FastEvent {
                    person_hash,
                    person: person_bytes,
                    ts,
                    price_cents,
                    etype,
                    category,
                });
            }
            valid
        })
        .collect();

    // Flatten all valid events
    let total_valid: usize = shard_valid_events.iter().map(|s| s.len()).sum();
    let mut valid_events: Vec<FastEvent> = Vec::with_capacity(total_valid);
    for s in shard_valid_events {
        valid_events.extend(s);
    }

    // 5. PARALLEL SORT EVENTS BY (PERSON_HASH, PERSON, TS)
    valid_events.par_sort_unstable_by(|a, b| {
        a.person_hash
            .cmp(&b.person_hash)
            .then_with(|| a.person.cmp(b.person))
            .then_with(|| a.ts.cmp(&b.ts))
    });

    // 6. Clicks: PARALLEL PARSE & SORT DEDUP
    let clicks_pattern = format!("{}/clicks/*.csv", data_dir);
    let mut click_files: Vec<PathBuf> = glob::glob(&clicks_pattern)?
        .filter_map(Result::ok)
        .collect();
    click_files.sort();

    let click_mmaps: Vec<Mmap> = click_files
        .iter()
        .map(|cf| {
            let file = File::open(cf).expect("open click file");
            unsafe { Mmap::map(&file).expect("mmap click file") }
        })
        .collect();

    #[derive(Clone, Copy)]
    struct RawClickLoc {
        cid_hash: u64,
        ts: i64,
        f_idx: u32,
        line_offset: u32,
        line_len: u32,
    }

    let per_file_raw_clicks: Vec<Vec<RawClickLoc>> = click_mmaps
        .par_iter()
        .enumerate()
        .map(|(f_idx, mmap)| {
            let mut list = Vec::with_capacity(mmap.len() / 60);
            let mut cols = [b"".as_slice(); 4];
            let mut start = 0;
            let bytes = mmap.as_ref();
            let mut line_num = 0;

            for (i, &b) in bytes.iter().enumerate() {
                if b == b'\n' {
                    let line = &bytes[start..i];
                    start = i + 1;
                    line_num += 1;
                    if line_num == 1 || line.is_empty() {
                        continue;
                    }
                    if parse_line_4(line, &mut cols) {
                        let cid = cols[0];
                        if !cid.is_empty() {
                            if let Some(ts) = parse_timestamp(cols[3]) {
                                if ts >= click_w_start && ts < w_end {
                                    list.push(RawClickLoc {
                                        cid_hash: fnv1a(cid),
                                        ts,
                                        f_idx: f_idx as u32,
                                        line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                        line_len: line.len() as u32,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if start < bytes.len() {
                let line = &bytes[start..];
                if line_num > 0 && !line.is_empty() && parse_line_4(line, &mut cols) {
                    let cid = cols[0];
                    if !cid.is_empty() {
                        if let Some(ts) = parse_timestamp(cols[3]) {
                            if ts >= click_w_start && ts < w_end {
                                list.push(RawClickLoc {
                                    cid_hash: fnv1a(cid),
                                    ts,
                                    f_idx: f_idx as u32,
                                    line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                    line_len: line.len() as u32,
                                });
                            }
                        }
                    }
                }
            }
            list
        })
        .collect();

    let total_raw_clicks: usize = per_file_raw_clicks.iter().map(|c| c.len()).sum();
    let mut all_raw_clicks = Vec::with_capacity(total_raw_clicks);
    for fc in per_file_raw_clicks {
        all_raw_clicks.extend(fc);
    }

    // Sort by cid_hash only for dedup
    all_raw_clicks.par_sort_unstable_by(|a, b| a.cid_hash.cmp(&b.cid_hash));

    let mut valid_clicks: Vec<FastClick> = Vec::with_capacity(all_raw_clicks.len());
    let mut c_idx = 0;
    while c_idx < all_raw_clicks.len() {
        let best_c = all_raw_clicks[c_idx];
        let best_line = &click_mmaps[best_c.f_idx as usize][best_c.line_offset as usize..(best_c.line_offset + best_c.line_len) as usize];
        let mut cols_best = [b"".as_slice(); 4];
        parse_line_4(best_line, &mut cols_best);
        let best_cid = cols_best[0];

        let mut next_c = c_idx + 1;
        while next_c < all_raw_clicks.len() && all_raw_clicks[next_c].cid_hash == best_c.cid_hash {
            let other_line = &click_mmaps[all_raw_clicks[next_c].f_idx as usize][all_raw_clicks[next_c].line_offset as usize..(all_raw_clicks[next_c].line_offset + all_raw_clicks[next_c].line_len) as usize];
            let mut cols_other = [b"".as_slice(); 4];
            if parse_line_4(other_line, &mut cols_other) && cols_other[0] == best_cid {
                next_c += 1;
            } else {
                break;
            }
        }
        c_idx = next_c;

        let uid_bytes = cols_best[1];
        let camp_bytes = cols_best[2];
        if uid_bytes.is_empty() {
            continue;
        }

        let Some(&camp_id) = campaign_map.get(camp_bytes) else {
            continue;
        };

        let (person_hash, person_bytes): (u64, &[u8]) = match resolved_links.get(uid_bytes) {
            Some(root) => (root.0, &root.1),
            None => (fnv1a(uid_bytes), uid_bytes),
        };

        valid_clicks.push(FastClick {
            person_hash,
            person: person_bytes,
            cid: best_cid,
            ts: best_c.ts,
            camp_id,
        });
    }

    // Sort clicks by (person_hash, person, ts, cid) for sessionization
    valid_clicks.par_sort_unstable_by(|a, b| {
        a.person_hash
            .cmp(&b.person_hash)
            .then_with(|| a.person.cmp(b.person))
            .then_with(|| a.ts.cmp(&b.ts))
            .then_with(|| a.cid.cmp(b.cid))
    });

    // 7. PARALLEL SESSIONIZE & ATTRIBUTE
    const THIRTY_MIN_MS: i64 = 1_800_000;
    const SEVEN_DAYS_MS: i64 = 7 * 86400 * 1000;

    struct UserDailyRecord<'a> {
        person: &'a [u8],
        day: i32,
        sessions: i64,
        events: i64,
        revenue: i64,
        top_category: Option<u8>,
    }

    #[derive(Clone, Copy)]
    struct CampaignDailyEntry {
        sessions: i64,
        purchasing_sessions: i64,
        revenue: i64,
        assisted_sessions: i64,
    }

    let n_threads = rayon::current_num_threads();
    let chunk_size = (valid_events.len() + n_threads - 1) / n_threads;
    let mut split_indices = Vec::with_capacity(n_threads + 1);
    split_indices.push(0);

    for i in 1..n_threads {
        let mut split = (i * chunk_size).min(valid_events.len());
        while split < valid_events.len()
            && (valid_events[split].person_hash == valid_events[split - 1].person_hash
                && valid_events[split].person == valid_events[split - 1].person)
        {
            split += 1;
        }
        split_indices.push(split);
    }
    split_indices.push(valid_events.len());

    let clicks_slice = &valid_clicks;
    let cat_names_ref = &cat_names;

    let thread_results: Vec<(Vec<UserDailyRecord>, HashMap<(u32, i32), CampaignDailyEntry>)> = (0..n_threads)
        .into_par_iter()
        .map(|t_idx| {
            let start_idx = split_indices[t_idx];
            let end_idx = split_indices[t_idx + 1];
            if start_idx >= end_idx {
                return (Vec::new(), HashMap::new());
            }

            let t_events = &valid_events[start_idx..end_idx];
            let mut user_records = Vec::with_capacity(t_events.len() / 15);
            let mut camp_map: HashMap<(u32, i32), CampaignDailyEntry> = HashMap::with_capacity(16384);

            // Per-thread reusable bitmask for O(1) assist-seen lookup
            // Size = n_campaigns (includes organic at 0)
            let mut assist_seen: Vec<bool> = vec![false; n_campaigns];
            // Track which indices were dirtied so we can reset cheaply
            let mut dirty_camps: Vec<u32> = Vec::with_capacity(64);

            // Per-thread reusable fixed-array for person_daily indexed by (day - w_start_day)
            // Use Option<PersonDayAgg> per slot; we clear only used slots
            struct PersonDayAgg {
                sessions: i64,
                events: i64,
                revenue: i64,
                cat_rev: [i64; 32],
            }
            let mut person_daily_arr: Vec<Option<PersonDayAgg>> = (0..n_days.max(1)).map(|_| None).collect();
            // Track which day slots are used so we can clear only them
            let mut used_days: Vec<usize> = Vec::with_capacity(n_days.min(64));

            let mut p_start = 0;
            while p_start < t_events.len() {
                let p_hash = t_events[p_start].person_hash;
                let p_name = t_events[p_start].person;

                let mut p_end = p_start + 1;
                while p_end < t_events.len()
                    && t_events[p_end].person_hash == p_hash
                    && t_events[p_end].person == p_name
                {
                    p_end += 1;
                }

                let person_events = &t_events[p_start..p_end];
                p_start = p_end;

                // Binary-search to find this person's click range
                let c_start = clicks_slice.partition_point(|c| (c.person_hash, c.person) < (p_hash, p_name));
                let c_end = clicks_slice.partition_point(|c| (c.person_hash, c.person) <= (p_hash, p_name));
                let person_clicks = &clicks_slice[c_start..c_end];

                // Reset used_days from previous person
                for &d_idx in &used_days {
                    person_daily_arr[d_idx] = None;
                }
                used_days.clear();

                // Sessionize
                let mut sess_start_ts = person_events[0].ts;
                let mut sess_prev_ts = person_events[0].ts;
                let mut sess_events = 0i64;
                let mut sess_revenue = 0i64;
                let mut sess_purchases = 0i64;
                let mut sess_cat_rev = [0i64; 32];

                // Macro-like inline session processor to avoid closure capture overhead
                macro_rules! flush_session {
                    () => {{
                        let s_start_ts = sess_start_ts;
                        let day_epoch = s_start_ts.div_euclid(86_400_000) as i32;
                        let d_idx = (day_epoch - w_start_day) as usize;

                        // Update person daily (fixed array, no alloc)
                        if d_idx < person_daily_arr.len() {
                            if person_daily_arr[d_idx].is_none() {
                                person_daily_arr[d_idx] = Some(PersonDayAgg {
                                    sessions: 0,
                                    events: 0,
                                    revenue: 0,
                                    cat_rev: [0; 32],
                                });
                                used_days.push(d_idx);
                            }
                            let pd = person_daily_arr[d_idx].as_mut().unwrap();
                            pd.sessions += 1;
                            pd.events += sess_events;
                            pd.revenue += sess_revenue;
                            for i in 0..32 {
                                pd.cat_rev[i] += sess_cat_rev[i];
                            }
                        }

                        // Find best click and assisting campaigns
                        // Binary-search for window start: s_start_ts - 7 days
                        let win_start = s_start_ts - SEVEN_DAYS_MS;
                        let lo = person_clicks.partition_point(|c| c.ts < win_start);

                        dirty_camps.clear();
                        let mut best_click: Option<(i64, &[u8], u32)> = None;

                        for c in &person_clicks[lo..] {
                            if c.ts > s_start_ts {
                                break;
                            }
                            // c.ts is in [win_start, s_start_ts]
                            let cid_usize = c.camp_id as usize;
                            if cid_usize < assist_seen.len() && !assist_seen[cid_usize] {
                                assist_seen[cid_usize] = true;
                                dirty_camps.push(c.camp_id);
                            }
                            if let Some(best) = &mut best_click {
                                if c.ts > best.0 {
                                    *best = (c.ts, c.cid, c.camp_id);
                                } else if c.ts == best.0 && c.cid < best.1 {
                                    *best = (c.ts, c.cid, c.camp_id);
                                }
                            } else {
                                best_click = Some((c.ts, c.cid, c.camp_id));
                            }
                        }

                        let credited_camp_id = best_click.map(|b| b.2).unwrap_or(ORGANIC_CAMP_ID);
                        let cd = camp_map.entry((credited_camp_id, day_epoch)).or_insert_with(|| CampaignDailyEntry {
                            sessions: 0,
                            purchasing_sessions: 0,
                            revenue: 0,
                            assisted_sessions: 0,
                        });
                        cd.sessions += 1;
                        if sess_purchases > 0 {
                            cd.purchasing_sessions += 1;
                        }
                        cd.revenue += sess_revenue;

                        for &camp_id in &dirty_camps {
                            let ac = camp_map.entry((camp_id, day_epoch)).or_insert_with(|| CampaignDailyEntry {
                                sessions: 0,
                                purchasing_sessions: 0,
                                revenue: 0,
                                assisted_sessions: 0,
                            });
                            ac.assisted_sessions += 1;
                        }

                        // Reset assist_seen bitmask (only dirty slots)
                        for &id in &dirty_camps {
                            assist_seen[id as usize] = false;
                        }
                    }};
                }

                for ev in person_events {
                    if (ev.ts - sess_prev_ts) > THIRTY_MIN_MS {
                        flush_session!();
                        sess_start_ts = ev.ts;
                        sess_events = 0;
                        sess_revenue = 0;
                        sess_purchases = 0;
                        sess_cat_rev = [0; 32];
                    }
                    sess_prev_ts = ev.ts;
                    sess_events += 1;
                    if ev.etype == 3 {
                        sess_purchases += 1;
                        sess_revenue += ev.price_cents as i64;
                        sess_cat_rev[ev.category as usize] += ev.price_cents as i64;
                    }
                }
                flush_session!();

                // Collect person daily records
                for &d_idx in &used_days {
                    let pd = person_daily_arr[d_idx].as_ref().unwrap();
                    let day_epoch = w_start_day + d_idx as i32;
                    let mut best_cat_idx: Option<usize> = None;
                    for i in 0..32 {
                        let amt = pd.cat_rev[i];
                        if amt > 0 {
                            if let Some(b) = best_cat_idx {
                                if amt > pd.cat_rev[b] {
                                    best_cat_idx = Some(i);
                                } else if amt == pd.cat_rev[b] && cat_names_ref[i] < cat_names_ref[b] {
                                    best_cat_idx = Some(i);
                                }
                            } else {
                                best_cat_idx = Some(i);
                            }
                        }
                    }
                    user_records.push(UserDailyRecord {
                        person: p_name,
                        day: day_epoch,
                        sessions: pd.sessions,
                        events: pd.events,
                        revenue: pd.revenue,
                        top_category: best_cat_idx.map(|idx| idx as u8),
                    });
                }
            }
            (user_records, camp_map)
        })
        .collect();

    // 8. PARQUET OUTPUT GENERATION
    let u_dir = Path::new(&output_dir).join("user_daily");
    fs::create_dir_all(&u_dir)?;
    let c_dir = Path::new(&output_dir).join("campaign_daily");
    fs::create_dir_all(&c_dir)?;

    let total_user_rows: usize = thread_results.iter().map(|(ur, _)| ur.len()).sum();
    let mut u_ids: Vec<String> = Vec::with_capacity(total_user_rows);
    let mut u_days: Vec<i32> = Vec::with_capacity(total_user_rows);
    let mut u_sessions: Vec<i64> = Vec::with_capacity(total_user_rows);
    let mut u_events: Vec<i64> = Vec::with_capacity(total_user_rows);
    let mut u_rev: Vec<i64> = Vec::with_capacity(total_user_rows);
    let mut u_top_cat: Vec<Option<String>> = Vec::with_capacity(total_user_rows);

    let mut merged_campaign_daily: HashMap<(u32, i32), CampaignDailyEntry> = HashMap::with_capacity(32768);

    for (ur, cm) in thread_results {
        for r in ur {
            u_ids.push(String::from_utf8_lossy(r.person).to_string());
            u_days.push(r.day);
            u_sessions.push(r.sessions);
            u_events.push(r.events);
            u_rev.push(r.revenue);
            u_top_cat.push(r.top_category.map(|idx| cat_names[idx as usize].clone()));
        }
        for (k, v) in cm {
            let entry = merged_campaign_daily.entry(k).or_insert_with(|| CampaignDailyEntry {
                sessions: 0,
                purchasing_sessions: 0,
                revenue: 0,
                assisted_sessions: 0,
            });
            entry.sessions += v.sessions;
            entry.purchasing_sessions += v.purchasing_sessions;
            entry.revenue += v.revenue;
            entry.assisted_sessions += v.assisted_sessions;
        }
    }

    // Write user_daily.parquet
    let u_schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let u_batch = RecordBatch::try_new(
        u_schema.clone(),
        vec![
            Arc::new(StringArray::from(u_ids)) as ArrayRef,
            Arc::new(Date32Array::from(u_days)) as ArrayRef,
            Arc::new(Int64Array::from(u_sessions)) as ArrayRef,
            Arc::new(Int64Array::from(u_events)) as ArrayRef,
            Arc::new(Int64Array::from(u_rev)) as ArrayRef,
            Arc::new(StringArray::from(u_top_cat)) as ArrayRef,
        ],
    )?;

    let u_file = File::create(u_dir.join("part-0.parquet"))?;
    let props = WriterProperties::builder().build();
    let mut u_writer = ArrowWriter::try_new(u_file, u_schema, Some(props))?;
    u_writer.write(&u_batch)?;
    u_writer.close()?;

    // Write campaign_daily.parquet
    let mut c_ids: Vec<String> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_channels: Vec<String> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_days: Vec<i32> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_sessions: Vec<i64> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_purchasing: Vec<i64> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_rev: Vec<i64> = Vec::with_capacity(merged_campaign_daily.len());
    let mut c_assisted: Vec<i64> = Vec::with_capacity(merged_campaign_daily.len());

    for ((camp_id, day_epoch), data) in merged_campaign_daily {
        let (cid_str, ch_str) = &campaign_names[camp_id as usize];
        c_ids.push(cid_str.clone());
        c_channels.push(ch_str.clone());
        c_days.push(day_epoch);
        c_sessions.push(data.sessions);
        c_purchasing.push(data.purchasing_sessions);
        c_rev.push(data.revenue);
        c_assisted.push(data.assisted_sessions);
    }

    let c_schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]));

    let c_batch = RecordBatch::try_new(
        c_schema.clone(),
        vec![
            Arc::new(StringArray::from(c_ids)) as ArrayRef,
            Arc::new(StringArray::from(c_channels)) as ArrayRef,
            Arc::new(Date32Array::from(c_days)) as ArrayRef,
            Arc::new(Int64Array::from(c_sessions)) as ArrayRef,
            Arc::new(Int64Array::from(c_purchasing)) as ArrayRef,
            Arc::new(Int64Array::from(c_rev)) as ArrayRef,
            Arc::new(Int64Array::from(c_assisted)) as ArrayRef,
        ],
    )?;

    let c_file = File::create(c_dir.join("part-0.parquet"))?;
    let props = WriterProperties::builder().build();
    let mut c_writer = ArrowWriter::try_new(c_file, c_schema, Some(props))?;
    c_writer.write(&c_batch)?;
    c_writer.close()?;

    Ok(())
}
