use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hashbrown::{HashMap, HashSet};
use memmap2::Mmap;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
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
    let sec = parse_2_digits(&s[17..19])?;
    if !(1..=12).contains(&m)
        || !(0..=23).contains(&h)
        || !(0..=59).contains(&mi)
        || !(0..=59).contains(&sec)
    {
        return None;
    }
    let dim = [0, 31, if is_leap(y) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=dim[m as usize]).contains(&d) {
        return None;
    }
    let ed = days_from_civil(y, m, d) as i64;
    Some((ed * 86400 + h * 3600 + mi * 60 + sec) * 1000)
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
    let sec = parse_2_digits(&s[17..19])?;
    let oh = parse_2_digits(&s[20..22])?;
    let omi = parse_2_digits(&s[23..25])?;
    if !(1..=12).contains(&m)
        || !(0..=23).contains(&h)
        || !(0..=59).contains(&mi)
        || !(0..=59).contains(&sec)
    {
        return None;
    }
    if !(0..=23).contains(&oh) || !(0..=59).contains(&omi) {
        return None;
    }
    let dim = [0, 31, if is_leap(y) { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=dim[m as usize]).contains(&d) {
        return None;
    }
    let ed = days_from_civil(y, m, d) as i64;
    let local_sec = ed * 86400 + h * 3600 + mi * 60 + sec;
    let offset_sec = oh * 3600 + omi * 60;
    let utc_sec = if sign == b'+' {
        local_sec - offset_sec
    } else {
        local_sec + offset_sec
    };
    Some(utc_sec * 1000)
}

#[inline(always)]
fn parse_timestamp(s: &[u8]) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    if s.ends_with(b"Z") {
        return parse_iso_f1(s);
    }
    if s.len() == 25 && (s[19] == b'+' || s[19] == b'-') {
        return parse_iso_f2(s);
    }
    if s.iter().all(|&b| b.is_ascii_digit()) {
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
    if s.contains(&b'\"') {
        return None;
    }
    Some(s)
}

#[inline(always)]
fn parse_line<'a>(line: &'a [u8], expected_cols: usize, out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();
    let mut start = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            let field = unquote_field(&line[start..i]);
            match field {
                Some(f) => out.push(f),
                None => return false,
            }
            start = i + 1;
        }
    }
    let field = unquote_field(&line[start..]);
    match field {
        Some(f) => out.push(f),
        None => return false,
    }
    out.len() == expected_cols
}

// ─── String Interner & DSU ───────────────────────────────────────────────────

struct Interner {
    map: HashMap<Vec<u8>, u32>,
    strings: Vec<Vec<u8>>,
}

impl Interner {
    fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            strings: Vec::with_capacity(cap),
        }
    }

    fn intern(&mut self, s: &[u8]) -> u32 {
        if let Some(&id) = self.map.get(s) {
            id
        } else {
            let id = self.strings.len() as u32;
            let v = s.to_vec();
            self.map.insert(v.clone(), id);
            self.strings.push(v);
            id
        }
    }

    fn get_id(&self, s: &[u8]) -> Option<u32> {
        self.map.get(s).copied()
    }

    fn resolve(&self, id: u32) -> &[u8] {
        &self.strings[id as usize]
    }
}

struct FastDsu {
    parent: Vec<u32>,
}

impl FastDsu {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size as u32).collect(),
        }
    }

    fn ensure_capacity(&mut self, size: usize) {
        while self.parent.len() < size {
            self.parent.push(self.parent.len() as u32);
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while x != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32, interner: &Interner) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            let sa = interner.resolve(ra);
            let sb = interner.resolve(rb);
            if sa < sb {
                self.parent[rb as usize] = ra;
            } else {
                self.parent[ra as usize] = rb;
            }
        }
    }
}

// ─── Event Struct ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CompactEvent {
    person: u32,
    ts: i64,
    price_cents: u32,
    etype: u8,
    category: u8,
}

// ─── Main Pipeline ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let out_dir = env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".to_string());

    let manifest_bytes = fs::read(Path::new(&data_dir).join("manifest.json"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let w_start = parse_iso_f1(manifest.window_start.as_bytes()).expect("bad window_start");
    let w_end = parse_iso_f1(manifest.window_end.as_bytes()).expect("bad window_end");
    let click_w_start = w_start - 7 * 86400 * 1000;

    // 1. Categories & Products
    let mut cat_interner = Interner::with_capacity(32);
    let unknown_cat_id = cat_interner.intern(b"unknown") as u8;

    let mut products: HashMap<Vec<u8>, u8> = HashMap::with_capacity(65536);
    let p_path = Path::new(&data_dir).join("products.csv");
    if p_path.exists() {
        let content = fs::read(&p_path)?;
        let mut cols = Vec::with_capacity(2);
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line(line, 2, &mut cols) && !cols[0].is_empty() {
                let cat_id = cat_interner.intern(cols[1]) as u8;
                products.insert(cols[0].to_vec(), cat_id);
            }
        }
    }

    // 2. Campaigns
    let mut campaign_interner = Interner::with_capacity(4096);
    let mut campaigns: HashMap<u16, (Vec<u8>, Vec<u8>)> = HashMap::with_capacity(4096);
    let c_path = Path::new(&data_dir).join("campaigns.csv");
    if c_path.exists() {
        let content = fs::read(&c_path)?;
        let mut cols = Vec::with_capacity(2);
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line(line, 2, &mut cols) && !cols[0].is_empty() {
                let camp_id = campaign_interner.intern(cols[0]) as u16;
                campaigns.insert(camp_id, (cols[0].to_vec(), cols[1].to_vec()));
            }
        }
    }

    // 3. User IDs & Links DSU
    let mut user_interner = Interner::with_capacity(2_000_000);
    let mut dsu = FastDsu::new(2_000_000);

    let links_pattern = format!("{}/links/*.csv", data_dir);
    let mut link_files: Vec<PathBuf> = glob::glob(&links_pattern)?
        .filter_map(Result::ok)
        .collect();
    link_files.sort();

    for lf in link_files {
        let content = fs::read(lf)?;
        let mut cols = Vec::with_capacity(2);
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line(line, 2, &mut cols) && !cols[0].is_empty() && !cols[1].is_empty() {
                let ua = user_interner.intern(cols[0]);
                let ub = user_interner.intern(cols[1]);
                let max_id = ua.max(ub) as usize + 1;
                dsu.ensure_capacity(max_id);
                dsu.union(ua, ub, &user_interner);
            }
        }
    }

    // 4. Events: PARALLEL PARSE & DETERMINISTIC SHARD DEDUP
    let events_pattern = format!("{}/events/*.csv", data_dir);
    let mut event_files: Vec<PathBuf> = glob::glob(&events_pattern)?
        .filter_map(Result::ok)
        .collect();
    event_files.sort();

    // Memory map all event files once
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
        file_idx: u32,
        line_offset: u32,
        line_len: u32,
        ingest_ms: i64,
    }

    const NUM_SHARDS: usize = 64;

    // Phase 4a: Scan files in parallel once, bucketing into shard buffers
    let per_file_locs: Vec<Vec<EventLoc>> = mmaps
        .par_iter()
        .enumerate()
        .map(|(f_idx, mmap)| {
            let mut locs = Vec::with_capacity(mmap.len() / 80);
            let mut cols = Vec::with_capacity(7);
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

                    if parse_line(line, 7, &mut cols) {
                        let eid = cols[0];
                        if !eid.is_empty() {
                            if let Some(ingest_ms) = parse_iso_f1(cols[3]) {
                                let eid_hash = fnv1a(eid);
                                locs.push(EventLoc {
                                    eid_hash,
                                    file_idx: f_idx as u32,
                                    line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                    line_len: line.len() as u32,
                                    ingest_ms,
                                });
                            }
                        }
                    }
                }
            }
            // trailing line if no \n
            if start < bytes.len() {
                let line = &bytes[start..];
                if line_num > 0 && !line.is_empty() && parse_line(line, 7, &mut cols) {
                    let eid = cols[0];
                    if !eid.is_empty() {
                        if let Some(ingest_ms) = parse_iso_f1(cols[3]) {
                            let eid_hash = fnv1a(eid);
                            locs.push(EventLoc {
                                eid_hash,
                                file_idx: f_idx as u32,
                                line_offset: (line.as_ptr() as usize - bytes.as_ptr() as usize) as u32,
                                line_len: line.len() as u32,
                                ingest_ms,
                            });
                        }
                    }
                }
            }
            locs
        })
        .collect();

    // Group into 64 shards
    let mut shard_locs: Vec<Vec<EventLoc>> = (0..NUM_SHARDS).map(|_| Vec::new()).collect();
    for file_loc in per_file_locs {
        for loc in file_loc {
            let shard = (loc.eid_hash as usize) % NUM_SHARDS;
            shard_locs[shard].push(loc);
        }
    }

    // Phase 4b: Deduplicate each shard in parallel
    let mmaps_ref = &mmaps;
    let surviving_locs: Vec<Vec<EventLoc>> = shard_locs
        .into_par_iter()
        .map(|locs| {
            // Deduplicate on eid: map from eid bytes to best EventLoc
            let mut map: HashMap<&[u8], EventLoc> = HashMap::with_capacity(locs.len());
            for loc in locs {
                let bytes = &mmaps_ref[loc.file_idx as usize][loc.line_offset as usize..(loc.line_offset + loc.line_len) as usize];
                // extract eid
                let mut cols = Vec::with_capacity(7);
                if parse_line(bytes, 7, &mut cols) {
                    let eid = cols[0];
                    if let Some(existing) = map.get_mut(eid) {
                        if loc.ingest_ms < existing.ingest_ms {
                            *existing = loc;
                        }
                    } else {
                        map.insert(eid, loc);
                    }
                }
            }
            map.into_values().collect()
        })
        .collect();

    // Phase 4c: Parse & validate surviving events
    let mut all_surviving: Vec<EventLoc> = Vec::with_capacity(surviving_locs.iter().map(|s| s.len()).sum());
    for s in surviving_locs {
        all_surviving.extend(s);
    }

    let mut valid_events: Vec<CompactEvent> = Vec::with_capacity(all_surviving.len());
    let mut cols = Vec::with_capacity(7);

    for loc in all_surviving {
        let line = &mmaps[loc.file_idx as usize][loc.line_offset as usize..(loc.line_offset + loc.line_len) as usize];
        if !parse_line(line, 7, &mut cols) {
            continue;
        }
        let uid_bytes = cols[1];
        let pid_bytes = cols[5];
        if uid_bytes.is_empty() || pid_bytes.is_empty() {
            continue;
        }

        let etype = match cols[4] {
            b"view" => 1u8,
            b"cart" => 2u8,
            b"purchase" => 3u8,
            _ => continue,
        };

        let mut price_cents = 0u32;
        if etype == 3 {
            if cols[6].is_empty() || !cols[6].iter().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let Ok(p) = std::str::from_utf8(cols[6]).unwrap().parse::<u32>() else {
                continue;
            };
            if !(1..=10_000_000).contains(&p) {
                continue;
            }
            price_cents = p;
        }

        let Some(ts) = parse_timestamp(cols[2]) else {
            continue;
        };
        if ts < w_start || ts >= w_end {
            continue;
        }
        if (loc.ingest_ms - ts) > 86400 * 1000 {
            continue;
        }

        let uid = user_interner.intern(uid_bytes);
        dsu.ensure_capacity(uid as usize + 1);
        let person = dsu.find(uid);

        let cat = products.get(pid_bytes).copied().unwrap_or(unknown_cat_id);

        valid_events.push(CompactEvent {
            person,
            ts,
            price_cents,
            etype,
            category: cat,
        });
    }

    // 5. PARALLEL SORT EVENTS BY (PERSON, TS)
    valid_events.par_sort_unstable_by_key(|e| (e.person, e.ts));

    // 6. Clicks: PARALLEL PARSE & DEDUP
    let clicks_pattern = format!("{}/clicks/*.csv", data_dir);
    let mut click_files: Vec<PathBuf> = glob::glob(&clicks_pattern)?
        .filter_map(Result::ok)
        .collect();
    click_files.sort();

    struct RawClick {
        cid: Vec<u8>,
        ts: i64,
        uid: Vec<u8>,
        camp_id: u16,
    }

    let per_file_clicks: Vec<Vec<RawClick>> = click_files
        .par_iter()
        .map(|cf| {
            let mut clicks = Vec::new();
            if let Ok(file) = File::open(cf) {
                if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                    let mut cols = Vec::with_capacity(4);
                    for line in mmap.split(|&b| b == b'\n').skip(1) {
                        if line.is_empty() {
                            continue;
                        }
                        if !parse_line(line, 4, &mut cols) {
                            continue;
                        }
                        let cid = cols[0];
                        let uid = cols[1];
                        if cid.is_empty() || uid.is_empty() {
                            continue;
                        }
                        if let Some(camp_id) = campaign_interner.get_id(cols[2]) {
                            if let Some(ts) = parse_timestamp(cols[3]) {
                                if ts >= click_w_start && ts < w_end {
                                    clicks.push(RawClick {
                                        cid: cid.to_vec(),
                                        ts,
                                        uid: uid.to_vec(),
                                        camp_id: camp_id as u16,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            clicks
        })
        .collect();

    // Deduplicate clicks on click_id
    let mut seen_clicks: HashSet<Vec<u8>> = HashSet::with_capacity(500_000);
    let mut valid_clicks_by_person: HashMap<u32, Vec<(i64, Vec<u8>, u16)>> = HashMap::with_capacity(500_000);

    for file_clicks in per_file_clicks {
        for c in file_clicks {
            if !seen_clicks.insert(c.cid.clone()) {
                continue;
            }
            let uid = user_interner.intern(&c.uid);
            dsu.ensure_capacity(uid as usize + 1);
            let person = dsu.find(uid);

            valid_clicks_by_person
                .entry(person)
                .or_default()
                .push((c.ts, c.cid, c.camp_id));
        }
    }

    for clicks in valid_clicks_by_person.values_mut() {
        clicks.sort_by_key(|c| c.0);
    }

    // 7. SESSIONIZE & ATTRIBUTE
    const THIRTY_MIN_MS: i64 = 1_800_000;
    const SEVEN_DAYS_MS: i64 = 7 * 86400 * 1000;

    struct UserDailyEntry {
        sessions: i64,
        events: i64,
        revenue: i64,
        cat_rev: [u64; 32],
    }

    struct CampaignDailyEntry {
        sessions: i64,
        purchasing_sessions: i64,
        revenue: i64,
        assisted_sessions: i64,
    }

    let mut user_daily_map: HashMap<(u32, i32), UserDailyEntry> = HashMap::with_capacity(200_000);
    let mut campaign_daily_map: HashMap<(Option<u16>, i32), CampaignDailyEntry> = HashMap::with_capacity(100_000);

    let mut event_idx = 0;
    while event_idx < valid_events.len() {
        let person = valid_events[event_idx].person;
        let mut person_end = event_idx + 1;
        while person_end < valid_events.len() && valid_events[person_end].person == person {
            person_end += 1;
        }

        let p_events = &valid_events[event_idx..person_end];
        let empty_clicks = Vec::new();
        let p_clicks = valid_clicks_by_person.get(&person).unwrap_or(&empty_clicks);

        let mut sess_start_ts = p_events[0].ts;
        let mut sess_prev_ts = p_events[0].ts;
        let mut sess_events = 0i64;
        let mut sess_revenue = 0i64;
        let mut sess_purchases = 0i64;
        let mut sess_cat_rev = [0u64; 32];

        let mut process_session = |start_ts: i64, events: i64, revenue: i64, purchases: i64, cat_rev: [u64; 32]| {
            let day_epoch = (start_ts.div_euclid(86_400_000)) as i32;

            // User daily
            let ud = user_daily_map.entry((person, day_epoch)).or_insert_with(|| UserDailyEntry {
                sessions: 0,
                events: 0,
                revenue: 0,
                cat_rev: [0; 32],
            });
            ud.sessions += 1;
            ud.events += events;
            ud.revenue += revenue;
            for i in 0..32 {
                ud.cat_rev[i] += cat_rev[i];
            }

            // Click attribution
            let mut best_click: Option<(i64, &[u8], u16)> = None;
            let mut assisting_camps: HashSet<u16> = HashSet::new();

            for (c_ts, c_id, c_camp) in p_clicks {
                if *c_ts > start_ts {
                    break;
                }
                if (start_ts - *c_ts) <= SEVEN_DAYS_MS {
                    assisting_camps.insert(*c_camp);
                    if let Some(best) = &mut best_click {
                        if *c_ts > best.0 {
                            *best = (*c_ts, c_id.as_slice(), *c_camp);
                        } else if *c_ts == best.0 && c_id.as_slice() < best.1 {
                            *best = (*c_ts, c_id.as_slice(), *c_camp);
                        }
                    } else {
                        best_click = Some((*c_ts, c_id.as_slice(), *c_camp));
                    }
                }
            }

            let credited_camp = best_click.map(|b| b.2);
            let cd = campaign_daily_map.entry((credited_camp, day_epoch)).or_insert_with(|| CampaignDailyEntry {
                sessions: 0,
                purchasing_sessions: 0,
                revenue: 0,
                assisted_sessions: 0,
            });
            cd.sessions += 1;
            if purchases > 0 {
                cd.purchasing_sessions += 1;
            }
            cd.revenue += revenue;

            for a_camp in assisting_camps {
                let ac = campaign_daily_map.entry((Some(a_camp), day_epoch)).or_insert_with(|| CampaignDailyEntry {
                    sessions: 0,
                    purchasing_sessions: 0,
                    revenue: 0,
                    assisted_sessions: 0,
                });
                ac.assisted_sessions += 1;
            }
        };

        for ev in p_events {
            if ev.ts - sess_prev_ts > THIRTY_MIN_MS {
                process_session(sess_start_ts, sess_events, sess_revenue, sess_purchases, sess_cat_rev);
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
                sess_cat_rev[ev.category as usize] += ev.price_cents as u64;
            }
        }
        process_session(sess_start_ts, sess_events, sess_revenue, sess_purchases, sess_cat_rev);

        event_idx = person_end;
    }

    // 8. WRITE OUTPUT PARQUET TABLES
    let ud_dir = Path::new(&out_dir).join("user_daily");
    let cd_dir = Path::new(&out_dir).join("campaign_daily");
    fs::create_dir_all(&ud_dir)?;
    fs::create_dir_all(&cd_dir)?;

    // Table 1: user_daily
    let mut u_ids: Vec<String> = Vec::with_capacity(user_daily_map.len());
    let mut u_days: Vec<i32> = Vec::with_capacity(user_daily_map.len());
    let mut u_sessions: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_events: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_rev: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_top_cat: Vec<Option<String>> = Vec::with_capacity(user_daily_map.len());

    for ((person, day_epoch), data) in user_daily_map {
        let person_str = String::from_utf8_lossy(user_interner.resolve(person)).to_string();
        u_ids.push(person_str);
        u_days.push(day_epoch);
        u_sessions.push(data.sessions);
        u_events.push(data.events);
        u_rev.push(data.revenue);

        // Find top category
        let mut best_cat: Option<(u8, u64)> = None;
        for (cat_idx, &amt) in data.cat_rev.iter().enumerate() {
            if amt > 0 {
                let cat_u8 = cat_idx as u8;
                if let Some(b) = best_cat {
                    if amt > b.1 {
                        best_cat = Some((cat_u8, amt));
                    } else if amt == b.1 {
                        let sa = cat_interner.resolve(cat_u8 as u32);
                        let sb = cat_interner.resolve(b.0 as u32);
                        if sa < sb {
                            best_cat = Some((cat_u8, amt));
                        }
                    }
                } else {
                    best_cat = Some((cat_u8, amt));
                }
            }
        }
        let top = best_cat.map(|b| String::from_utf8_lossy(cat_interner.resolve(b.0 as u32)).to_string());
        u_top_cat.push(top);
    }

    let ud_schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let ud_batch = RecordBatch::try_new(
        ud_schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(u_ids.iter().map(|s| s.as_str()))) as ArrayRef,
            Arc::new(Date32Array::from(u_days)) as ArrayRef,
            Arc::new(Int64Array::from(u_sessions)) as ArrayRef,
            Arc::new(Int64Array::from(u_events)) as ArrayRef,
            Arc::new(Int64Array::from(u_rev)) as ArrayRef,
            Arc::new(StringArray::from_iter(u_top_cat.iter().map(|opt| opt.as_deref()))) as ArrayRef,
        ],
    )?;

    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .build();

    let ud_file = File::create(ud_dir.join("part-0.parquet"))?;
    let mut ud_writer = ArrowWriter::try_new(ud_file, ud_schema, Some(props.clone()))?;
    ud_writer.write(&ud_batch)?;
    ud_writer.close()?;

    // Table 2: campaign_daily
    let mut c_ids: Vec<String> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_channels: Vec<String> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_days: Vec<i32> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_sessions: Vec<i64> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_purchasing: Vec<i64> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_rev: Vec<i64> = Vec::with_capacity(campaign_daily_map.len());
    let mut c_assisted: Vec<i64> = Vec::with_capacity(campaign_daily_map.len());

    for ((opt_camp, day_epoch), data) in campaign_daily_map {
        let (cid_str, channel_str) = match opt_camp {
            Some(cid) => {
                let (c, ch) = &campaigns[&cid];
                (
                    String::from_utf8_lossy(c).to_string(),
                    String::from_utf8_lossy(ch).to_string(),
                )
            }
            None => ("organic".to_string(), "organic".to_string()),
        };

        c_ids.push(cid_str);
        c_channels.push(channel_str);
        c_days.push(day_epoch);
        c_sessions.push(data.sessions);
        c_purchasing.push(data.purchasing_sessions);
        c_rev.push(data.revenue);
        c_assisted.push(data.assisted_sessions);
    }

    let cd_schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]));

    let cd_batch = RecordBatch::try_new(
        cd_schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(c_ids.iter().map(|s| s.as_str()))) as ArrayRef,
            Arc::new(StringArray::from_iter_values(c_channels.iter().map(|s| s.as_str()))) as ArrayRef,
            Arc::new(Date32Array::from(c_days)) as ArrayRef,
            Arc::new(Int64Array::from(c_sessions)) as ArrayRef,
            Arc::new(Int64Array::from(c_purchasing)) as ArrayRef,
            Arc::new(Int64Array::from(c_rev)) as ArrayRef,
            Arc::new(Int64Array::from(c_assisted)) as ArrayRef,
        ],
    )?;

    let cd_file = File::create(cd_dir.join("part-0.parquet"))?;
    let mut cd_writer = ArrowWriter::try_new(cd_file, cd_schema, Some(props))?;
    cd_writer.write(&cd_batch)?;
    cd_writer.close()?;

    Ok(())
}
