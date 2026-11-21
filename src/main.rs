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
use serde::Deserialize;

// ─── Fast Civil / Calendar Math ──────────────────────────────────────────────

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

fn days_from_civil(mut y: i64, m: i64, d: i64) -> i32 {
    if m <= 2 {
        y -= 1;
    }
    let era = y.div_euclid(400);
    let y_oe = y.rem_euclid(400);
    let d_oy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    (era * 146097 + y_oe * 365 + y_oe / 4 - y_oe / 100 + d_oy - 719468) as i32
}

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
    let is_leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dims = [
        0,
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=dims[m as usize]).contains(&d) {
        return None;
    }
    let ed = days_from_civil(y, m, d) as i64;
    Some((ed * 86400 + h * 3600 + mi * 60 + sec) * 1000)
}

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
    for &idx in &[
        0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24,
    ] {
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
    let is_leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dims = [
        0,
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=dims[m as usize]).contains(&d) {
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
    // F3: ASCII digits only
    if s.iter().all(|b| b.is_ascii_digit()) {
        std::str::from_utf8(s).ok()?.parse::<i64>().ok()
    } else {
        None
    }
}

// ─── CSV Field Unquoting ─────────────────────────────────────────────────────

#[inline(always)]
fn unquote_field(slice: &[u8]) -> Option<&[u8]> {
    if slice.is_empty() {
        return Some(slice);
    }
    if slice[0] == b'"' {
        if slice.len() < 2 || slice[slice.len() - 1] != b'"' {
            return None;
        }
        let inner = &slice[1..slice.len() - 1];
        if inner.iter().any(|&b| b == b'"') {
            return None;
        }
        Some(inner)
    } else {
        if slice.iter().any(|&b| b == b'"') {
            return None;
        }
        Some(slice)
    }
}

#[inline(always)]
fn parse_line<'a>(line: &'a [u8], expected_cols: usize, out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();
    let mut start = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if let Some(field) = unquote_field(&line[start..i]) {
                out.push(field);
            } else {
                return false;
            }
            start = i + 1;
        }
    }
    if let Some(field) = unquote_field(&line[start..]) {
        out.push(field);
    } else {
        return false;
    }
    out.len() == expected_cols
}

// ─── Disjoint Set Union (DSU) for Linked IDs ──────────────────────────────────

#[derive(Default)]
struct Dsu {
    parent: HashMap<Vec<u8>, Vec<u8>>,
}

impl Dsu {
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

// ─── Main Pipeline ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

struct RawEvent {
    ingest_ms: i64,
    uid: Vec<u8>,
    ts_str: Vec<u8>,
    etype: Vec<u8>,
    pid: Vec<u8>,
    price_str: Vec<u8>,
}

struct ValidEvent {
    ts: i64,
    etype: u8, // 1: view, 2: cart, 3: purchase
    category: Arc<str>,
    price_cents: i64,
}

struct Session {
    start_ts: i64,
    last_distinct_ts: i64,
    events: i64,
    revenue: i64,
    purchases: i64,
    cat_rev: HashMap<Arc<str>, i64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let out_dir = env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".to_string());

    let manifest_bytes = fs::read(Path::new(&data_dir).join("manifest.json"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let w_start = parse_iso_f1(manifest.window_start.as_bytes()).expect("bad window_start");
    let w_end = parse_iso_f1(manifest.window_end.as_bytes()).expect("bad window_end");
    let click_w_start = w_start - 7 * 86400 * 1000;

    // 1. Products: product_id -> category
    let mut products: HashMap<Vec<u8>, Arc<str>> = HashMap::new();
    let p_path = Path::new(&data_dir).join("products.csv");
    if p_path.exists() {
        let content = fs::read(&p_path)?;
        let mut cols = Vec::with_capacity(2);
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line(line, 2, &mut cols) && !cols[0].is_empty() {
                let cat_str: Arc<str> = String::from_utf8_lossy(cols[1]).into();
                products.insert(cols[0].to_vec(), cat_str);
            }
        }
    }

    // 2. Campaigns: campaign_id -> channel
    let mut campaigns: HashMap<Vec<u8>, Arc<str>> = HashMap::new();
    let c_path = Path::new(&data_dir).join("campaigns.csv");
    if c_path.exists() {
        let content = fs::read(&c_path)?;
        let mut cols = Vec::with_capacity(2);
        for line in content.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if parse_line(line, 2, &mut cols) && !cols[0].is_empty() {
                let ch_str: Arc<str> = String::from_utf8_lossy(cols[1]).into();
                campaigns.insert(cols[0].to_vec(), ch_str);
            }
        }
    }

    // 3. Links & DSU
    let mut dsu = Dsu::default();
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
                dsu.union(cols[0], cols[1]);
            }
        }
    }

    // 4. Events: Parse & Dedup by event_id (keep smallest ingest_ts)
    let events_pattern = format!("{}/events/*.csv", data_dir);
    let mut event_files: Vec<PathBuf> = glob::glob(&events_pattern)?
        .filter_map(Result::ok)
        .collect();
    event_files.sort();

    let mut kept_events: HashMap<Vec<u8>, RawEvent> = HashMap::new();

    for ef in event_files {
        let file = File::open(ef)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cols = Vec::with_capacity(7);
        for line in mmap.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if !parse_line(line, 7, &mut cols) {
                continue;
            }
            let eid = cols[0];
            if eid.is_empty() {
                continue;
            }
            let ingest_str = cols[3];
            let Some(ingest_ms) = parse_iso_f1(ingest_str) else {
                continue;
            };

            if let Some(existing) = kept_events.get_mut(eid) {
                if ingest_ms < existing.ingest_ms {
                    existing.ingest_ms = ingest_ms;
                    existing.uid = cols[1].to_vec();
                    existing.ts_str = cols[2].to_vec();
                    existing.etype = cols[4].to_vec();
                    existing.pid = cols[5].to_vec();
                    existing.price_str = cols[6].to_vec();
                }
            } else {
                kept_events.insert(
                    eid.to_vec(),
                    RawEvent {
                        ingest_ms,
                        uid: cols[1].to_vec(),
                        ts_str: cols[2].to_vec(),
                        etype: cols[4].to_vec(),
                        pid: cols[5].to_vec(),
                        price_str: cols[6].to_vec(),
                    },
                );
            }
        }
    }

    // Validate kept events
    let unknown_cat: Arc<str> = "unknown".into();
    let mut valid_events_by_person: HashMap<Vec<u8>, Vec<ValidEvent>> = HashMap::new();

    for (_eid, ev) in kept_events {
        if ev.uid.is_empty() || ev.pid.is_empty() {
            continue;
        }
        let etype = match ev.etype.as_slice() {
            b"view" => 1,
            b"cart" => 2,
            b"purchase" => 3,
            _ => continue,
        };
        let Some(ts) = parse_timestamp(&ev.ts_str) else {
            continue;
        };
        if ts < w_start || ts >= w_end {
            continue;
        }
        // Late event check: ingest_ts - ts > 24 hours
        if (ev.ingest_ms - ts) > 86400 * 1000 {
            continue;
        }

        let mut price = 0i64;
        if etype == 3 {
            if ev.price_str.is_empty() || !ev.price_str.iter().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let Ok(p) = std::str::from_utf8(&ev.price_str).unwrap().parse::<i64>() else {
                continue;
            };
            if !(1..=10_000_000).contains(&p) {
                continue;
            }
            price = p;
        }

        let person = dsu.find(&ev.uid);
        let category = products.get(&ev.pid).cloned().unwrap_or_else(|| unknown_cat.clone());

        valid_events_by_person
            .entry(person)
            .or_default()
            .push(ValidEvent {
                ts,
                etype,
                category,
                price_cents: price,
            });
    }

    // 5. Sessionize per person
    let mut person_sessions: HashMap<Vec<u8>, Vec<Session>> = HashMap::new();

    for (person, mut evs) in valid_events_by_person {
        evs.sort_by_key(|e| e.ts);
        let mut sessions: Vec<Session> = Vec::new();
        let mut cur_sess: Option<Session> = None;

        for ev in evs {
            if let Some(ref mut sess) = cur_sess {
                let gap = ev.ts - sess.last_distinct_ts;
                if gap > 1_800_000 {
                    sessions.push(cur_sess.take().unwrap());
                    let mut cat_rev = HashMap::new();
                    if ev.etype == 3 {
                        cat_rev.insert(ev.category.clone(), ev.price_cents);
                    }
                    cur_sess = Some(Session {
                        start_ts: ev.ts,
                        last_distinct_ts: ev.ts,
                        events: 1,
                        revenue: ev.price_cents,
                        purchases: if ev.etype == 3 { 1 } else { 0 },
                        cat_rev,
                    });
                } else {
                    sess.events += 1;
                    if ev.ts > sess.last_distinct_ts {
                        sess.last_distinct_ts = ev.ts;
                    }
                    if ev.etype == 3 {
                        sess.revenue += ev.price_cents;
                        sess.purchases += 1;
                        *sess.cat_rev.entry(ev.category.clone()).or_insert(0) += ev.price_cents;
                    }
                }
            } else {
                let mut cat_rev = HashMap::new();
                if ev.etype == 3 {
                    cat_rev.insert(ev.category.clone(), ev.price_cents);
                }
                cur_sess = Some(Session {
                    start_ts: ev.ts,
                    last_distinct_ts: ev.ts,
                    events: 1,
                    revenue: ev.price_cents,
                    purchases: if ev.etype == 3 { 1 } else { 0 },
                    cat_rev,
                });
            }
        }
        if let Some(sess) = cur_sess {
            sessions.push(sess);
        }
        person_sessions.insert(person, sessions);
    }

    // 6. Clicks: Parse, Dedup by click_id, Validate
    let clicks_pattern = format!("{}/clicks/*.csv", data_dir);
    let mut click_files: Vec<PathBuf> = glob::glob(&clicks_pattern)?
        .filter_map(Result::ok)
        .collect();
    click_files.sort();

    let mut seen_clicks: HashSet<Vec<u8>> = HashSet::new();
    let mut valid_clicks_by_person: HashMap<Vec<u8>, Vec<(i64, Vec<u8>, Vec<u8>)>> = HashMap::new();

    for cf in click_files {
        let file = File::open(cf)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cols = Vec::with_capacity(4);
        for line in mmap.split(|&b| b == b'\n').skip(1) {
            if line.is_empty() {
                continue;
            }
            if !parse_line(line, 4, &mut cols) {
                continue;
            }
            let cid = cols[0];
            if cid.is_empty() || !seen_clicks.insert(cid.to_vec()) {
                continue;
            }
            let uid = cols[1];
            let camp_id = cols[2];
            if uid.is_empty() || !campaigns.contains_key(camp_id) {
                continue;
            }
            let Some(ts) = parse_timestamp(cols[3]) else {
                continue;
            };
            if ts < click_w_start || ts >= w_end {
                continue;
            }
            let person = dsu.find(uid);
            valid_clicks_by_person
                .entry(person)
                .or_default()
                .push((ts, cid.to_vec(), camp_id.to_vec()));
        }
    }

    for clicks in valid_clicks_by_person.values_mut() {
        clicks.sort_by_key(|c| c.0);
    }

    // 7. Attribution & Output Aggregations
    struct UserDailyEntry {
        sessions: i64,
        events: i64,
        revenue: i64,
        cat_rev: HashMap<Arc<str>, i64>,
    }

    struct CampaignDailyEntry {
        sessions: i64,
        purchasing_sessions: i64,
        revenue: i64,
        assisted_sessions: i64,
    }

    let mut user_daily_map: HashMap<(Vec<u8>, i32), UserDailyEntry> = HashMap::new();
    let mut campaign_daily_map: HashMap<(Vec<u8>, i32), CampaignDailyEntry> = HashMap::new();
    let seven_days_ms = 7 * 86400 * 1000;

    for (person, sessions) in person_sessions {
        let empty_clicks = Vec::new();
        let person_clicks = valid_clicks_by_person.get(&person).unwrap_or(&empty_clicks);

        for sess in sessions {
            let day_epoch = (sess.start_ts.div_euclid(86_400_000)) as i32;

            // User daily
            let ud_key = (person.clone(), day_epoch);
            let ud_entry = user_daily_map.entry(ud_key).or_insert_with(|| UserDailyEntry {
                sessions: 0,
                events: 0,
                revenue: 0,
                cat_rev: HashMap::new(),
            });
            ud_entry.sessions += 1;
            ud_entry.events += sess.events;
            ud_entry.revenue += sess.revenue;
            for (cat, amt) in &sess.cat_rev {
                *ud_entry.cat_rev.entry(cat.clone()).or_insert(0) += amt;
            }

            // Click attribution
            let mut best_click: Option<(i64, &[u8], &[u8])> = None;
            let mut assisting_camps: HashSet<&[u8]> = HashSet::new();

            for (c_ts, c_id, c_camp) in person_clicks {
                if *c_ts > sess.start_ts {
                    break;
                }
                if (sess.start_ts - *c_ts) <= seven_days_ms {
                    assisting_camps.insert(c_camp.as_slice());
                    if let Some((best_ts, best_id, _)) = best_click {
                        if *c_ts > best_ts || (*c_ts == best_ts && c_id.as_slice() < best_id) {
                            best_click = Some((*c_ts, c_id.as_slice(), c_camp.as_slice()));
                        }
                    } else {
                        best_click = Some((*c_ts, c_id.as_slice(), c_camp.as_slice()));
                    }
                }
            }

            let credited_camp = best_click
                .map(|(_, _, camp)| camp.to_vec())
                .unwrap_or_else(|| b"organic".to_vec());

            // Campaign Daily credited row
            let cd_key = (credited_camp, day_epoch);
            let cd_entry = campaign_daily_map.entry(cd_key).or_insert_with(|| CampaignDailyEntry {
                sessions: 0,
                purchasing_sessions: 0,
                revenue: 0,
                assisted_sessions: 0,
            });
            cd_entry.sessions += 1;
            if sess.purchases > 0 {
                cd_entry.purchasing_sessions += 1;
            }
            cd_entry.revenue += sess.revenue;

            // Campaign Daily assists (organic never assists)
            for a_camp in assisting_camps {
                let ac_key = (a_camp.to_vec(), day_epoch);
                let ac_entry = campaign_daily_map.entry(ac_key).or_insert_with(|| CampaignDailyEntry {
                    sessions: 0,
                    purchasing_sessions: 0,
                    revenue: 0,
                    assisted_sessions: 0,
                });
                ac_entry.assisted_sessions += 1;
            }
        }
    }

    // 8. Write Parquet Outputs
    let out_path = Path::new(&out_dir);
    let ud_dir = out_path.join("user_daily");
    let cd_dir = out_path.join("campaign_daily");
    fs::create_dir_all(&ud_dir)?;
    fs::create_dir_all(&cd_dir)?;

    // Table 1: user_daily
    let mut u_ids: Vec<String> = Vec::with_capacity(user_daily_map.len());
    let mut u_days: Vec<i32> = Vec::with_capacity(user_daily_map.len());
    let mut u_sessions: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_events: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_rev: Vec<i64> = Vec::with_capacity(user_daily_map.len());
    let mut u_top_cat: Vec<Option<String>> = Vec::with_capacity(user_daily_map.len());

    for ((uid, day_epoch), data) in user_daily_map {
        u_ids.push(String::from_utf8(uid).unwrap_or_default());
        u_days.push(day_epoch);
        u_sessions.push(data.sessions);
        u_events.push(data.events);
        u_rev.push(data.revenue);

        let top = if data.cat_rev.is_empty() {
            None
        } else {
            // max revenue; tie breaker byte-wise smallest category
            let mut pairs: Vec<(&Arc<str>, &i64)> = data.cat_rev.iter().collect();
            pairs.sort_by(|a, b| {
                b.1.cmp(a.1).then_with(|| a.0.cmp(b.0))
            });
            Some(pairs[0].0.to_string())
        };
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

    let organic_str: Arc<str> = "organic".into();

    for ((cid, day_epoch), data) in campaign_daily_map {
        let channel = if cid.as_slice() == b"organic" {
            organic_str.clone()
        } else {
            campaigns.get(&cid).cloned().unwrap_or_else(|| "".into())
        };
        c_ids.push(String::from_utf8(cid).unwrap_or_default());
        c_channels.push(channel.to_string());
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
