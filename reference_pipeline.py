#!/usr/bin/env python3
"""
High-Performance Shopping Sessions Reference Pipeline
Meets 100% of rules.md specifications.
"""
import glob
import json
import os
import sys
from datetime import date, datetime, timezone
import polars as pl

# ─── Date / Time Utilities ───────────────────────────────────────────────────

DAYS_BEFORE_MONTH = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
DAYS_BEFORE_MONTH_LEAP = [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

def is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

def days_from_civil(y: int, m: int, d: int) -> int:
    y_adj = y - 1 if m <= 2 else y
    era = y_adj // 400
    y_oe = y_adj - era * 400
    d_oy = (153 * ((m + 9) % 12) + 2) // 5 + d - 1
    return era * 146097 + y_oe * 365 + y_oe // 4 - y_oe // 100 + d_oy - 719468

def civil_from_epoch_days(epoch_days: int) -> tuple[int, int, int]:
    shifted = epoch_days + 719468
    era = shifted // 146097
    d_oe = shifted - era * 146097
    y_oe = (d_oe - d_oe // 1460 + d_oe // 36524 - d_oe // 146096) // 365
    d_oy = d_oe - (365 * y_oe + y_oe // 4 - y_oe // 100)
    m_idx = (5 * d_oy + 2) // 153
    month = m_idx + 3 if m_idx < 10 else m_idx - 9
    year = y_oe + era * 400 + (1 if month <= 2 else 0)
    day = d_oy - (153 * m_idx + 2) // 5 + 1
    return year, month, day

ASCII_DIGITS = set("0123456789")

def is_ascii_digits(s: str) -> bool:
    if not s:
        return False
    for c in s:
        if c not in ASCII_DIGITS:
            return False
    return True

def parse_iso_f1(s: str) -> int | None:
    # 2026-10-03T23:58:10Z (len 20)
    if len(s) != 20 or s[10] != 'T' or s[19] != 'Z' or s[4] != '-' or s[7] != '-' or s[13] != ':' or s[16] != ':':
        return None
    for idx in (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18):
        if s[idx] not in ASCII_DIGITS:
            return None
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10])
        h = int(s[11:13])
        mi = int(s[14:16])
        sec = int(s[17:19])
    except ValueError:
        return None
    if not (1 <= m <= 12 and 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= sec <= 59):
        return None
    dim = [0, 31, 29 if is_leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
    if not (1 <= d <= dim):
        return None
    ed = days_from_civil(y, m, d)
    return (ed * 86400 + h * 3600 + mi * 60 + sec) * 1000

def parse_iso_f2(s: str) -> int | None:
    # 2026-10-03T19:58:10-04:00 (len 25)
    if len(s) != 25 or s[10] != 'T' or s[4] != '-' or s[7] != '-' or s[13] != ':' or s[16] != ':' or s[22] != ':':
        return None
    sign = s[19]
    if sign not in ('+', '-'):
        return None
    for idx in (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24):
        if s[idx] not in ASCII_DIGITS:
            return None
    try:
        y = int(s[0:4])
        m = int(s[5:7])
        d = int(s[8:10])
        h = int(s[11:13])
        mi = int(s[14:16])
        sec = int(s[17:19])
        oh = int(s[20:22])
        omi = int(s[23:25])
    except ValueError:
        return None
    if not (1 <= m <= 12 and 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= sec <= 59):
        return None
    if not (0 <= oh <= 23 and 0 <= omi <= 59):
        return None
    dim = [0, 31, 29 if is_leap(y) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
    if not (1 <= d <= dim):
        return None
    ed = days_from_civil(y, m, d)
    local_sec = ed * 86400 + h * 3600 + mi * 60 + sec
    offset_sec = oh * 3600 + omi * 60
    if sign == '+':
        utc_sec = local_sec - offset_sec
    else:
        utc_sec = local_sec + offset_sec
    return utc_sec * 1000

def parse_timestamp(s: str) -> int | None:
    if not s:
        return None
    if s.endswith('Z'):
        return parse_iso_f1(s)
    if len(s) == 25 and (s[19] == '+' or s[19] == '-'):
        return parse_iso_f2(s)
    # F3: digits only (ASCII)
    if is_ascii_digits(s):
        try:
            return int(s)
        except ValueError:
            return None
    return None

# ─── CSV Field Unquoting ─────────────────────────────────────────────────────

def unquote_field(field: str) -> str | None:
    """Returns unquoted value, or None if malformed. An empty field returns ''."""
    if not field:
        return ""
    if field.startswith('"'):
        if len(field) < 2 or not field.endswith('"'):
            return None # malformed
        inner = field[1:-1]
        if '"' in inner:
            return None # quote anywhere inside is malformed
        return inner
    else:
        if '"' in field:
            return None
        return field

def parse_csv_line(line: str, expected_cols: int) -> list[str] | None:
    """Parses a CSV line according to exact rules.md dialect."""
    if not line:
        return None
    parts = line.split(',')
    if len(parts) != expected_cols:
        return None
    res = []
    for p in parts:
        uq = unquote_field(p)
        if uq is None:
            return None
        res.append(uq)
    return res

# ─── Disjoint Set Union (DSU) for Linked IDs ──────────────────────────────────

class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, x: str) -> str:
        p = self.parent.get(x)
        if p is None:
            return x
        if p == x:
            return x
        root = self.find(p)
        self.parent[x] = root
        return root

    def union(self, a: str, b: str):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
                self.parent[b] = ra
            else:
                self.parent[ra] = rb
                self.parent[a] = rb

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str, output_dir: str):
    # 1. Manifest
    with open(os.path.join(data_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    w_start_str = manifest["window_start"]
    w_end_str = manifest["window_end"]
    w_start = parse_iso_f1(w_start_str)
    w_end = parse_iso_f1(w_end_str)
    click_w_start = w_start - 7 * 86400 * 1000

    # 2. Products
    products = {}
    p_path = os.path.join(data_dir, "products.csv")
    if os.path.exists(p_path):
        with open(p_path, "r", newline='', encoding="utf-8") as f:
            lines = f.read().split('\n')
            for line in lines[1:]: # skip header
                if not line: continue
                cols = parse_csv_line(line, 2)
                if cols and cols[0]:
                    products[cols[0]] = cols[1]

    # 3. Campaigns
    campaigns = {}
    c_path = os.path.join(data_dir, "campaigns.csv")
    if os.path.exists(c_path):
        with open(c_path, "r", newline='', encoding="utf-8") as f:
            lines = f.read().split('\n')
            for line in lines[1:]:
                if not line: continue
                cols = parse_csv_line(line, 2)
                if cols and cols[0]:
                    campaigns[cols[0]] = cols[1]

    # 4. Links & DSU
    dsu = DSU()
    link_files = sorted(glob.glob(os.path.join(data_dir, "links", "*.csv")))
    for lf in link_files:
        with open(lf, "r", newline='', encoding="utf-8") as f:
            lines = f.read().split('\n')
            if not lines: continue
            for line in lines[1:]: # skip header
                if not line: continue
                cols = parse_csv_line(line, 2)
                if not cols: continue
                u_a, u_b = cols[0], cols[1]
                if not u_a or not u_b:
                    continue # empty side dropped
                dsu.union(u_a, u_b)

    # 5. Events: Parse & Dedup on event_id (keep smallest ingest_ts)
    # event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents
    event_files = sorted(glob.glob(os.path.join(data_dir, "events", "*.csv")))
    kept_events = {} # event_id -> (ingest_ms, [cols])

    for ef in event_files:
        with open(ef, "r", newline='', encoding="utf-8") as f:
            lines = f.read().split('\n')
            if not lines: continue
            for line in lines[1:]:
                if not line: continue
                cols = parse_csv_line(line, 7)
                if not cols: continue
                eid = cols[0]
                if not eid: continue # empty event_id dropped
                ingest_str = cols[3]
                ingest_ms = parse_iso_f1(ingest_str)
                if ingest_ms is None:
                    continue
                if eid not in kept_events:
                    kept_events[eid] = (ingest_ms, cols)
                else:
                    if ingest_ms < kept_events[eid][0]:
                        kept_events[eid] = (ingest_ms, cols)

    # Validate kept events
    # event: (person_id, ts, event_type, product_id, price_cents, ingest_ms)
    valid_events_by_person = {} # person_id -> list of event dicts

    for eid, (ingest_ms, cols) in kept_events.items():
        _, uid, ts_str, _, etype, pid, price_str = cols
        if not uid or not pid:
            continue
        if etype not in ('view', 'cart', 'purchase'):
            continue
        ts = parse_timestamp(ts_str)
        if ts is None or not (w_start <= ts < w_end):
            continue
        
        # Late event check
        if (ingest_ms - ts) > 86400 * 1000:
            continue
        
        price = 0
        if etype == 'purchase':
            if not is_ascii_digits(price_str):
                continue
            price = int(price_str)
            if not (1 <= price <= 10000000):
                continue
        
        person = dsu.find(uid)
        cat = products.get(pid, "unknown")
        
        ev = (ts, etype, cat, price)
        if person not in valid_events_by_person:
            valid_events_by_person[person] = []
        valid_events_by_person[person].append(ev)

    # 6. Sessionize per person
    # A session: {person, start_ts, day, event_count, revenue_cents, purchase_count, category_rev: {}}
    person_sessions = {} # person -> list of sessions

    for person, evs in valid_events_by_person.items():
        # Sort by ts
        evs.sort(key=lambda x: x[0])
        sessions = []
        cur_session = None

        for ev in evs:
            ts, etype, cat, price = ev
            if cur_session is None:
                # Start first session
                cur_session = {
                    "start_ts": ts,
                    "last_distinct_ts": ts,
                    "events": 1,
                    "revenue": price,
                    "purchases": 1 if etype == 'purchase' else 0,
                    "cat_rev": {cat: price} if etype == 'purchase' else {}
                }
            else:
                gap = ts - cur_session["last_distinct_ts"]
                if gap > 1800000: # strictly > 30 minutes
                    # End cur_session and start new
                    sessions.append(cur_session)
                    cur_session = {
                        "start_ts": ts,
                        "last_distinct_ts": ts,
                        "events": 1,
                        "revenue": price,
                        "purchases": 1 if etype == 'purchase' else 0,
                        "cat_rev": {cat: price} if etype == 'purchase' else {}
                    }
                else:
                    # Same session
                    cur_session["events"] += 1
                    if ts > cur_session["last_distinct_ts"]:
                        cur_session["last_distinct_ts"] = ts
                    if etype == 'purchase':
                        cur_session["revenue"] += price
                        cur_session["purchases"] += 1
                        cur_session["cat_rev"][cat] = cur_session["cat_rev"].get(cat, 0) + price

        if cur_session is not None:
            sessions.append(cur_session)
        person_sessions[person] = sessions

    # 7. Clicks: Parse, Dedup, Validate
    # click_id,user_id,campaign_id,ts
    click_files = sorted(glob.glob(os.path.join(data_dir, "clicks", "*.csv")))
    seen_clicks = set() # click_id dedup
    valid_clicks_by_person = {} # person -> list of (ts, click_id, campaign_id)

    for cf in click_files:
        with open(cf, "r", newline='', encoding="utf-8") as f:
            lines = f.read().split('\n')
            if not lines: continue
            for line in lines[1:]:
                if not line: continue
                cols = parse_csv_line(line, 4)
                if not cols: continue
                cid, uid, camp_id, ts_str = cols
                if not cid or cid in seen_clicks:
                    continue
                seen_clicks.add(cid)
                if not uid or camp_id not in campaigns:
                    continue
                ts = parse_timestamp(ts_str)
                if ts is None or not (click_w_start <= ts < w_end):
                    continue
                person = dsu.find(uid)
                if person not in valid_clicks_by_person:
                    valid_clicks_by_person[person] = []
                valid_clicks_by_person[person].append((ts, cid, camp_id))

    # Sort person clicks by ts
    for person in valid_clicks_by_person:
        valid_clicks_by_person[person].sort(key=lambda c: c[0])

    # 8. Attribution & Output Aggregations
    # user_daily: (user_id, day_date) -> {sessions, events, revenue_cents, cat_rev: {}}
    user_daily_map = {}
    # campaign_daily: (campaign_id, day_date) -> {sessions, purchasing_sessions, revenue_cents, assisted_sessions}
    campaign_daily_map = {}

    SEVEN_DAYS_MS = 7 * 86400 * 1000

    for person, sessions in person_sessions.items():
        person_clicks = valid_clicks_by_person.get(person, [])
        for sess in sessions:
            s_start = sess["start_ts"]
            y, m, d = civil_from_epoch_days(s_start // 86400000)
            day_val = date(y, m, d)

            # User daily aggregation
            ud_key = (person, day_val)
            if ud_key not in user_daily_map:
                user_daily_map[ud_key] = {
                    "sessions": 0, "events": 0, "revenue": 0, "cat_rev": {}
                }
            ud_entry = user_daily_map[ud_key]
            ud_entry["sessions"] += 1
            ud_entry["events"] += sess["events"]
            ud_entry["revenue"] += sess["revenue"]
            for cat, amt in sess["cat_rev"].items():
                ud_entry["cat_rev"][cat] = ud_entry["cat_rev"].get(cat, 0) + amt

            # Attribution to click:
            # click.ts <= s_start and s_start - click.ts <= 7 days
            best_click = None # (ts, click_id, camp_id)
            assisting_campaigns = set()

            for c_ts, c_id, c_camp in person_clicks:
                if c_ts > s_start:
                    break
                if (s_start - c_ts) <= SEVEN_DAYS_MS:
                    assisting_campaigns.add(c_camp)
                    if best_click is None:
                        best_click = (c_ts, c_id, c_camp)
                    else:
                        # latest ts; tie: smallest click_id
                        if c_ts > best_click[0]:
                            best_click = (c_ts, c_id, c_camp)
                        elif c_ts == best_click[0]:
                            if c_id < best_click[1]:
                                best_click = (c_ts, c_id, c_camp)

            credited_campaign = best_click[2] if best_click else "organic"

            # Campaign Daily Credit
            cd_key = (credited_campaign, day_val)
            if cd_key not in campaign_daily_map:
                campaign_daily_map[cd_key] = {
                    "sessions": 0, "purchasing_sessions": 0, "revenue": 0, "assisted_sessions": 0
                }
            cd_entry = campaign_daily_map[cd_key]
            cd_entry["sessions"] += 1
            if sess["purchases"] > 0:
                cd_entry["purchasing_sessions"] += 1
            cd_entry["revenue"] += sess["revenue"]

            # Campaign Daily Assists:
            # "organic never assists"
            for a_camp in assisting_campaigns:
                ac_key = (a_camp, day_val)
                if ac_key not in campaign_daily_map:
                    campaign_daily_map[ac_key] = {
                        "sessions": 0, "purchasing_sessions": 0, "revenue": 0, "assisted_sessions": 0
                    }
                campaign_daily_map[ac_key]["assisted_sessions"] += 1

    # 9. Format Tables & Write Parquet
    os.makedirs(os.path.join(output_dir, "user_daily"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "campaign_daily"), exist_ok=True)

    # Build user_daily rows
    ud_rows = []
    for (person, day_val), data in user_daily_map.items():
        top_cat = None
        if data["cat_rev"]:
            # category with highest summed purchase price_cents; tie goes to byte-wise smallest
            # sorted by (-sum, cat)
            top_cat = min(data["cat_rev"].keys(), key=lambda c: (-data["cat_rev"][c], c))
        ud_rows.append({
            "user_id": person,
            "day": day_val,
            "sessions": data["sessions"],
            "events": data["events"],
            "revenue_cents": data["revenue"],
            "top_category": top_cat
        })

    ud_schema = {
        "user_id": pl.String, "day": pl.Date, "sessions": pl.Int64,
        "events": pl.Int64, "revenue_cents": pl.Int64, "top_category": pl.String
    }
    if ud_rows:
        df_ud = pl.DataFrame(ud_rows, schema=ud_schema)
    else:
        df_ud = pl.DataFrame(schema=ud_schema)
    df_ud.write_parquet(os.path.join(output_dir, "user_daily", "part-0.parquet"))

    # Build campaign_daily rows
    cd_rows = []
    for (camp_id, day_val), data in campaign_daily_map.items():
        channel = "organic" if camp_id == "organic" else campaigns.get(camp_id, "")
        cd_rows.append({
            "campaign_id": camp_id,
            "channel": channel,
            "day": day_val,
            "sessions": data["sessions"],
            "purchasing_sessions": data["purchasing_sessions"],
            "revenue_cents": data["revenue"],
            "assisted_sessions": data["assisted_sessions"]
        })

    cd_schema = {
        "campaign_id": pl.String, "channel": pl.String, "day": pl.Date,
        "sessions": pl.Int64, "purchasing_sessions": pl.Int64, "revenue_cents": pl.Int64,
        "assisted_sessions": pl.Int64
    }
    if cd_rows:
        df_cd = pl.DataFrame(cd_rows, schema=cd_schema)
    else:
        df_cd = pl.DataFrame(schema=cd_schema)
    df_cd.write_parquet(os.path.join(output_dir, "campaign_daily", "part-0.parquet"))

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", sys.argv[1] if len(sys.argv) > 1 else "/data")
    out_dir = os.environ.get("OUTPUT_DIR", sys.argv[2] if len(sys.argv) > 2 else "/output")
    run_pipeline(data_dir, out_dir)
