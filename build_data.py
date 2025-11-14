import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SURVEY_BASE = Path(os.environ.get("SURVEY_BASE", "output/seekho"))
BASE = SURVEY_BASE
DASH_DIR = BASE / "dashboard" / "seekho_junior_dashboard"
ASSETS_DIR = DASH_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_PATH = BASE / "analysis" / "canonical_dataset.csv"
DERIVED_PATH = BASE / "analysis" / "derived_metrics.csv"
MULTISELECT_PATH = (
    BASE / "analysis" / "stats" / "analysis" / "processed" / "multi_select_binary.csv"
)
INSIGHT_JSON_PATH = BASE / "analysis" / "insight_matrix.json"
INSIGHT_SUMMARY_JSON_PATH = BASE / "analysis" / "insight_summary.json"
INSIGHT_MATRIX_MD_PATH = BASE / "analysis" / "insight_matrix.md"
CHI_MD_PATH = BASE / "analysis" / "chi_square_highlights.md"
ANOVA_MD_PATH = BASE / "analysis" / "anova_highlights.md"
SEGMENT_DELTAS_PATH = BASE / "analysis" / "segment_deltas.md"
METHODOLOGY_PATH = BASE / "analysis_methodology" / "analysis_plan.md"
PLAYBOOKS_DIR = BASE / "analysis" / "playbooks"
STAT_CLINIC_PATH = BASE / "analysis" / "stat_clinic.md"
MACRO_GUARDRAILS_PATH = BASE / "analysis" / "macro_guardrails.md"
LEADERSHIP_SUMMARY_PATH = BASE / "analysis" / "leadership_summary.md"
VISUAL_INPUTS_DIR = BASE / "analysis" / "stats" / "analysis" / "visual_inputs"
PROMPT_OUTPUTS_DIR = BASE / "analysis" / "prompt_outputs"
SCRIPTS_DIR = Path("scripts")
CLUSTER_NOTES_PATH = BASE / "analysis" / "behavioural_clusters.md"
QUESTION_PATH_CANDIDATES = [
    BASE / "question" / "question.json",
    BASE / "questions" / "question_raw.json",
]
QUESTION_PATH = next((p for p in QUESTION_PATH_CANDIDATES if p.exists()), None)

SEGMENT_FIELDS = [
    "city_tier",
    "bucket_income_annual",
    "consumption_tier",
    "age_cohort",
    "gender_identity",
]

DEFAULT_METRIC_KEYS = {
    "session_duration_comfort": {
        "label": "Session comfort",
        "unit": "mins",
        "description": "Upper limit (minutes) caregivers are comfortable with per sitting.",
        "insight": "Higher = caregivers tolerate longer sessions before needing a break.",
    },
    "session_frequency_intent": {
        "label": "Weekly cadence intent",
        "unit": "days/week",
        "description": "Planned number of days per week the child can use Seekho.",
        "insight": "Higher = intent to build daily/near-daily habits.",
    },
    "use_case_breadth": {
        "label": "Use-case breadth",
        "unit": "contexts",
        "description": "Count of Q5 scenarios (travel, quiet-time, etc.) parents allow.",
        "insight": "Higher = caregivers integrate the app into more family routines.",
    },
    "format_affinity_score": {
        "label": "Format affinity",
        "unit": "score",
        "description": "Composite across Q8 preferred formats (shorts, guided activities, offline packs).",
        "insight": "Higher = appetite for richer, multi-format programming.",
    },
    "safety_trust_index": {
        "label": "Safety & trust index",
        "unit": "score",
        "description": "Weighted importance of Q10 safety features.",
        "insight": "Higher = non-negotiable need for safety guardrails before trial.",
    },
    "proof_confidence_index": {
        "label": "Proof confidence",
        "unit": "score",
        "description": "Strength of Q19 proof-points required before purchase.",
        "insight": "Higher = requires more social/academic proof to convert.",
    },
    "trial_benefit_pull": {
        "label": "Trial benefit pull",
        "unit": "score",
        "description": "Appeal of Q18 trial incentives (ad-free, dashboards, personalised paths).",
        "insight": "Higher = easier to activate with featured benefits.",
    },
    "monetisation_propensity": {
        "label": "Monetisation propensity",
        "unit": "score",
        "description": "Composite of willingness to pay, preferred plans, payment comfort.",
        "insight": "Higher = more open to paid tiers beyond the free plan.",
    },
    "learning_outcome_priority_count": {
        "label": "Learning outcome priorities",
        "unit": "count",
        "description": "Number of Q20 learning gains parents want simultaneously.",
        "insight": "Higher = need for broader curriculum coverage.",
    },
    "progress_feedback_options": {
        "label": "Progress feedback signals",
        "unit": "count",
        "description": "How many Q22 progress touchpoints (dashboards, printable reports) they want.",
        "insight": "Higher = expect more reporting/updates to stay engaged.",
    },
    "influencer_leverage_count": {
        "label": "Influencer leverage",
        "unit": "count",
        "description": "Number of trusted Q23 recommendation channels (teachers, pediatricians, etc.).",
        "insight": "Higher = more nodes that can nudge adoption.",
    },
    "blocker_count": {
        "label": "Blocker load",
        "unit": "count",
        "description": "Volume of objections from Q24 (entertainment-only, payment worries, etc.).",
        "insight": "Higher = heavier objection handling required pre-conversion.",
    },
    "blocker_types": {
        "label": "Blocker types",
        "unit": "tags",
        "description": "Taxonomy of blocker categories cited (string).",
        "insight": "",
    },
    "retention_hooks_index": {
        "label": "Retention hooks",
        "unit": "score",
        "description": "Strength of Q26 hooks (badges, playlists, loyalty).",
        "insight": "Higher = more features needed to sustain weekly use.",
    },
    "ad_messaging_resonance": {
        "label": "Ad messaging resonance",
        "unit": "score",
        "description": "Response to Q29 creative angles.",
        "insight": "Higher = segments resonate with safety/localisation/daily-routine positioning.",
    },
    "localisation_language_count": {
        "label": "Language breadth",
        "unit": "languages",
        "description": "Number of languages caregivers will use (Q12).",
        "insight": "Higher = need to localise into multiple tongues per household.",
    },
    "localisation_primary_language": {
        "label": "Primary localisation language",
        "unit": "text",
        "description": "Main requested language.",
        "insight": "",
    },
    "low_bandwidth_support": {
        "label": "Low-bandwidth support",
        "unit": "score",
        "description": "Need for offline/download/low-data options (Q14).",
        "insight": "Higher = emphasise offline packs + lighter media.",
    },
    "bandwidth_constraint_score": {
        "label": "Bandwidth constraint score",
        "unit": "score",
        "description": "Constraint index combining Q14 answers + persona infra signals.",
        "insight": "Higher = more low-data/offline expectation before paying.",
    },
    "q32_cluster_count": {
        "label": "Feature cluster hits",
        "unit": "count",
        "description": "Number of Q32 build/improve requests selected.",
        "insight": "Higher = expect richer roadmap shipment volume.",
    },
}

HAIR_METRIC_KEYS = {
    "pricing_monthly_spend_value": {
        "label": "Monthly spend",
        "unit": "₹",
        "description": "Average monthly hair-care spend.",
        "insight": "Higher = premium readiness for launch tiers.",
    },
    "premium_price_band_value": {
        "label": "Premium ceiling",
        "unit": "₹",
        "description": "Upper limit respondents will pay for premium SKUs.",
        "insight": "Keeps price ladder inside persona tolerance.",
    },
    "intro_offer_count": {
        "label": "Offer appetite",
        "unit": "count",
        "description": "Number of intro offers (Q33) each persona selected.",
        "insight": "Signals how many hooks launch creatives must cover.",
    },
    "message_hook_cluster_count": {
        "label": "Message hooks",
        "unit": "count",
        "description": "Distinct creative hooks (Q34 clusters) that resonate.",
        "insight": "Higher = broader creative palette per segment.",
    },
    "cadence_weekly_wash_score": {
        "label": "Wash cadence",
        "unit": "sessions/week",
        "description": "How often they wash hair each week.",
        "insight": "Guides refill sizing and ritual cadence.",
    },
    "cadence_brand_trust_count": {
        "label": "Trusted brands",
        "unit": "count",
        "description": "Number of brands they currently trust.",
        "insight": "Higher = more proof required to switch.",
    },
    "content_product_variety": {
        "label": "Product stack breadth",
        "unit": "count",
        "description": "How many products they use today.",
        "insight": "Perfect indicator for bundle depth.",
    },
    "pricing_refill_trigger_count": {
        "label": "Refill motivators",
        "unit": "count",
        "description": "Refill/eco triggers selected (Q15).",
        "insight": "Higher = stronger sustainability hook.",
    },
    "pricing_value_signal_count": {
        "label": "Value proof asks",
        "unit": "count",
        "description": "Value validation signals they need (results, ingredients).",
        "insight": "Higher = more messaging proof points.",
    },
    "salon_treatment_count": {
        "label": "Salon treatments",
        "unit": "count",
        "description": "Number of salon services tried past year.",
        "insight": "Helps scope pro-channel plays.",
    },
    "subscription_propensity_score": {
        "label": "Subscription propensity",
        "unit": "score",
        "description": "Likelihood to enroll in auto-replenishment.",
        "insight": "High score = easy conversion to refill programs.",
    },
    "localisation_satisfaction_score": {
        "label": "Routine satisfaction",
        "unit": "score",
        "description": "Self-reported satisfaction with current routine.",
        "insight": "Lower scores reveal where to focus product proof.",
    },
    "localisation_pain_count": {
        "label": "Pain point load",
        "unit": "count",
        "description": "Number of pain points cited (Q12).",
        "insight": "Higher = more urgency to switch.",
    },
    "online_channel_mix_count": {
        "label": "Channel mix breadth",
        "unit": "count",
        "description": "How many online purchase channels they use.",
        "insight": "Quantifies complexity of launch footprint.",
    },
    "influencer_reliance_score": {
        "label": "Influencer reliance",
        "unit": "score",
        "description": "Influencer/stylist influence on decisions.",
        "insight": "Higher = invest in creator/salon advocates.",
    },
    "personalisation_interest_score": {
        "label": "Personalisation interest",
        "unit": "score",
        "description": "Interest level in customised formulas.",
        "insight": "Identifies cohorts for quiz + booster flows.",
    },
    "brand_personality_count": {
        "label": "Brand personality hits",
        "unit": "count",
        "description": "Number of brand personalities that resonate.",
        "insight": "Helps creative team prioritise tone/mood boards.",
    },
}


def resolve_metric_keys(columns):
    if columns and "pricing_monthly_spend_value" in columns:
        return HAIR_METRIC_KEYS
    return DEFAULT_METRIC_KEYS


def parse_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def slugify_key(text: str, fallback: str = "field") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or fallback


def parse_markdown_table(lines, start_index):
    header_line = lines[start_index].strip()
    if start_index + 1 >= len(lines):
        return [], start_index + 1
    divider_line = lines[start_index + 1].strip()
    if "---" not in divider_line:
        return [], start_index + 1
    raw_headers = header_line.strip("|").split("|")
    headers = []
    for idx, header in enumerate(raw_headers):
        clean = header.strip()
        if not clean:
            clean = f"column_{idx + 1}"
        headers.append(clean)
    slug_headers = []
    seen = {}
    for idx, header in enumerate(headers):
        slug = slugify_key(header or f"col_{idx + 1}")
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}_{seen[slug]}"
        else:
            seen[slug] = 1
        slug_headers.append(slug)
    rows = []
    i = start_index + 2
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith("|") or "---" in line:
            break
        values = [v.strip() for v in line.strip().strip("|").split("|")]
        if not any(values):
            i += 1
            continue
        while len(values) < len(headers):
            values.append("")
        if len(values) > len(headers):
            values = values[: len(headers)]
        field_map = {}
        column_meta = []
        for slug, label, value in zip(slug_headers, headers, values):
            field_map[slug] = value
            column_meta.append({"key": slug, "label": label, "value": value})
        rows.append({"fields": field_map, "columns": column_meta})
        i += 1
    return rows, i


def read_text_if_exists(path: Path):
    if path and path.exists():
        return path.read_text()
    return None


def parse_playbook_markdown(md_text: str):
    lines = md_text.splitlines()
    title = ""
    stat_cues = []
    segments = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("# "):
            if not title:
                title = stripped[2:].strip()
        elif stripped.startswith("## "):
            section_title = stripped[3:].strip().lower()
            if section_title.startswith("step 5 statistical cues"):
                i += 1
                while i < len(lines):
                    cue_line = lines[i].strip()
                    if cue_line.startswith("- "):
                        stat_cues.append(cue_line[2:].strip())
                        i += 1
                    elif cue_line == "":
                        i += 1
                    else:
                        break
                continue
        if stripped.startswith("|"):
            if i + 1 < len(lines) and "---" in lines[i + 1]:
                table_rows, next_index = parse_markdown_table(lines, i)
                segments.extend(table_rows)
                i = next_index
                continue
        i += 1
    return {"title": title.strip(), "stat_cues": stat_cues, "segments": segments}


def parse_cluster_notes(path: Path):
    if not path.exists():
        return {}
    clusters = {}
    current_id = None
    current_section = None
    pattern = re.compile(r"## Cluster (\d+) – (.+?) \(n=(\d+)\)")
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            match = pattern.match(line)
            if match:
                cid, name, size = match.groups()
                current_id = cid
                clusters[cid] = {"name": name.strip(), "size": int(size), "notes": []}
                current_section = None
                continue
            if not current_id:
                continue
            if line.startswith("Top categorical signals"):
                current_section = "categorical"
                continue
            if line.startswith("Top blocker buckets"):
                clusters[current_id]["notes"].append(
                    "Top blockers: " + line.split(":", 1)[1].strip()
                )
                current_section = None
                continue
            if line.startswith("Top Q32 clusters"):
                clusters[current_id]["notes"].append("Build asks: " + line.split(":", 1)[1].strip())
                current_section = None
                continue
            if line.startswith("- "):
                if current_section == "categorical":
                    key, rest = line[2:].split(":", 1)
                    label_map = {
                        "city_tier": "City tier skew",
                        "consumption_tier": "Consumption mix",
                        "bucket_income_annual": "Income mix",
                    }
                    label = label_map.get(key.strip(), key.strip().title())
                    clusters[current_id]["notes"].append(f"{label}: {rest.strip()}")
            elif not line:
                current_section = None
    for info in clusters.values():
        info["summary"] = " | ".join(info["notes"])
    return clusters


cluster_context = parse_cluster_notes(CLUSTER_NOTES_PATH)

with DERIVED_PATH.open() as f:
    derived_reader = csv.DictReader(f)
    derived_columns = derived_reader.fieldnames or []
    derived_rows = list(derived_reader)

derived_id_field = (
    "respondent_id" if "respondent_id" in derived_columns else "persona_id"
)

METRIC_KEYS = resolve_metric_keys(derived_columns)

derived_map = {}
metric_stats = {
    key: {
        "sum": 0.0,
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
        "by_city_tier": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        "by_consumption_tier": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        "by_income": defaultdict(lambda: {"sum": 0.0, "count": 0}),
    }
    for key in METRIC_KEYS
}

segment_counts = {field: Counter() for field in SEGMENT_FIELDS}
gender_counts = Counter()

for row in derived_rows:
    rid = row.get(derived_id_field)
    if not rid:
        continue
    derived_map[rid] = row
    for key, meta in METRIC_KEYS.items():
        if key in ("blocker_types", "localisation_primary_language"):
            continue
        val = parse_float(row.get(key))
        if val is None:
            continue
        stats = metric_stats[key]
        stats["sum"] += val
        stats["count"] += 1
        if val < stats["min"]:
            stats["min"] = val
        if val > stats["max"]:
            stats["max"] = val
        stats["by_city_tier"][row.get("city_tier") or "unknown"]["sum"] += val
        stats["by_city_tier"][row.get("city_tier") or "unknown"]["count"] += 1
        stats["by_consumption_tier"][row.get("consumption_tier") or "unknown"]["sum"] += val
        stats["by_consumption_tier"][row.get("consumption_tier") or "unknown"]["count"] += 1
        stats["by_income"][row.get("bucket_income_annual") or "unknown"]["sum"] += val
        stats["by_income"][row.get("bucket_income_annual") or "unknown"]["count"] += 1

respondent_total = len(derived_map)

with CANONICAL_PATH.open() as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
respondent_field = None
if header:
    respondent_field = "respondent_id" if "respondent_id" in header else "persona_id"
else:
    respondent_field = "respondent_id"

question_meta = {}
if QUESTION_PATH and QUESTION_PATH.exists():
    with QUESTION_PATH.open() as fjson:
        qdata = json.load(fjson)
    if isinstance(qdata, dict) and "questions" in qdata:
        iterator = qdata["questions"]
    else:
        iterator = qdata or []
    for idx, q in enumerate(iterator, start=1):
        question_index = q.get("question_index", idx)
        try:
            question_index = int(question_index)
        except (TypeError, ValueError):
            question_index = idx
        raw_id = q.get("question_id") or q.get("id") or f"q{question_index:02d}"
        if isinstance(raw_id, str) and raw_id.lower().startswith("q"):
            try:
                numeric = int("".join(filter(str.isdigit, raw_id)))
                qid = f"q{numeric:02d}"
            except ValueError:
                qid = raw_id.lower()
        else:
            qid = f"q{question_index:02d}"
        goal_alignment = q.get("goal_alignment")
        if isinstance(goal_alignment, list):
            goal_tag = ", ".join(goal_alignment)
        else:
            goal_tag = goal_alignment
        question_meta[qid] = {
            "index": question_index,
            "title": q.get("question") or q.get("text") or qid,
            "provide_reason": q.get("provide_reason", q.get("requires_reason", False)),
            "topic": q.get("category"),
            "type": q.get("type"),
            "tag": goal_tag,
        }
else:
    question_meta = {}

DEFAULT_QUESTION_TOPICS = {
    "q01": {"topic": "Device access", "type": "multi", "tag": "Devices"},
    "q02": {"topic": "Connectivity", "type": "single", "tag": "Bandwidth"},
    "q03": {"topic": "Familiarity", "type": "single", "tag": "Awareness"},
    "q04": {"topic": "Interest", "type": "single", "tag": "Demand"},
    "q05": {"topic": "Use cases", "type": "multi", "tag": "Routines"},
    "q06": {"topic": "Session length", "type": "single", "tag": "Cadence"},
    "q07": {"topic": "Content roadmap", "type": "multi", "tag": "Content"},
    "q08": {"topic": "Format mix", "type": "multi", "tag": "Formats"},
    "q09": {"topic": "Avoid list", "type": "multi", "tag": "Content"},
    "q10": {"topic": "Safety & trust", "type": "multi", "tag": "Trust"},
    "q11": {"topic": "Home language", "type": "single", "tag": "Localisation"},
    "q12": {"topic": "Content languages", "type": "multi", "tag": "Localisation"},
    "q13": {"topic": "Cultural importance", "type": "single", "tag": "Localisation"},
    "q14": {"topic": "Low-bandwidth modes", "type": "multi", "tag": "Bandwidth"},
    "q15": {"topic": "Willingness to pay", "type": "single", "tag": "Pricing"},
    "q16": {"topic": "Plan preferences", "type": "multi", "tag": "Pricing"},
    "q17": {"topic": "Payment methods", "type": "multi", "tag": "Payments"},
    "q18": {"topic": "Trial hooks", "type": "multi", "tag": "Pricing"},
    "q19": {"topic": "Proof points", "type": "multi", "tag": "Trust"},
    "q20": {"topic": "Desired outcomes", "type": "multi", "tag": "Content"},
    "q21": {"topic": "Weekly cadence", "type": "single", "tag": "Cadence"},
    "q22": {"topic": "Progress tracking", "type": "single", "tag": "Retention"},
    "q23": {"topic": "Recommendation sources", "type": "multi", "tag": "Acquisition"},
    "q24": {"topic": "Blockers", "type": "multi", "tag": "Objections"},
    "q25": {"topic": "Post-trial intent", "type": "single", "tag": "Retention"},
    "q26": {"topic": "Retention hooks", "type": "multi", "tag": "Retention"},
    "q27": {"topic": "Ad tone", "type": "single", "tag": "Acquisition"},
    "q28": {"topic": "Ad channels", "type": "multi", "tag": "Acquisition"},
    "q29": {"topic": "Ad messages", "type": "multi", "tag": "Acquisition"},
    "q30": {"topic": "Printable kits", "type": "single", "tag": "Content"},
    "q31": {"topic": "Beta participation", "type": "single", "tag": "Research"},
    "q32": {"topic": "Build requests", "type": "multi", "tag": "Roadmap"},
}

question_topics = {}
for qid, meta in question_meta.items():
    topic = meta.get("topic")
    if not topic:
        continue
    question_topics[qid] = {
        "topic": topic,
        "type": meta.get("type"),
        "tag": meta.get("tag") or topic,
    }

if not question_topics:
    question_topics = DEFAULT_QUESTION_TOPICS.copy()
else:
    merged_topics = DEFAULT_QUESTION_TOPICS.copy()
    merged_topics.update({k: v for k, v in question_topics.items() if v})
    question_topics = merged_topics

question_counts = {}
question_denoms = {}
for qid in question_meta:
    question_counts[qid] = {
        "overall": Counter(),
        "segments": {field: defaultdict(Counter) for field in SEGMENT_FIELDS},
    }
    question_denoms[qid] = {
        "overall": 0,
        "segments": {field: defaultdict(int) for field in SEGMENT_FIELDS},
    }

persona_entries = []

with CANONICAL_PATH.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        rid = row.get(respondent_field) or row.get("persona_id")
        if not rid:
            continue
        for field in SEGMENT_FIELDS:
            value = row.get(field)
            if value:
                segment_counts[field][value] += 1
        gender = row.get("gender_identity")
        if gender:
            gender_counts[gender] += 1
        derived = derived_map.get(rid, {})
        persona_entry = {
            "respondent_id": rid,
            "persona_id": row.get("persona_id"),
            "summary": row.get("persona_summary", ""),
            "city_tier": row.get("city_tier"),
            "consumption_tier": row.get("consumption_tier"),
            "bucket_income_annual": row.get("bucket_income_annual"),
            "age_cohort": row.get("age_cohort"),
            "gender_identity": row.get("gender_identity"),
            "cluster_id": row.get("cluster_id"),
            "current_city": row.get("current_city"),
            "persona_highlights": {
                "price_elasticity": row.get("price_elasticity"),
                "cashback_sensitivity": row.get("cashback_sensitivity"),
                "risk_tolerance": row.get("risk_tolerance"),
                "loss_aversion_lambda": row.get("loss_aversion_lambda"),
            },
            "metrics": {},
        }
        for key in METRIC_KEYS:
            if key in ("blocker_types", "localisation_primary_language"):
                persona_entry["metrics"][key] = derived.get(key)
            else:
                persona_entry["metrics"][key] = parse_float(derived.get(key))
        persona_entries.append(persona_entry)

        for qid in question_meta:
            answer = row.get(f"{qid}_answer", "")
            if not answer:
                continue
            raw_options = [opt.strip() for opt in answer.split(";") if opt.strip()]
            if not raw_options:
                raw_options = [answer.strip()]
            question_denoms[qid]["overall"] += 1
            question_counts[qid]["overall"].update(raw_options)
            for field in SEGMENT_FIELDS:
                seg_value = row.get(field)
                if not seg_value:
                    continue
                question_denoms[qid]["segments"][field][seg_value] += 1
                question_counts[qid]["segments"][field][seg_value].update(raw_options)


def counter_to_share(counter, denom):
    if denom == 0:
        return []
    return [
        {"option": option, "count": count, "percent": round((count / denom) * 100, 1)}
        for option, count in counter.most_common()
    ]


question_data = {}
for qid in question_meta:
    meta = question_meta[qid].copy()
    meta.update(question_topics.get(qid, {}))
    overall = counter_to_share(question_counts[qid]["overall"], question_denoms[qid]["overall"])
    segments_payload = {}
    for field in SEGMENT_FIELDS:
        seg_payload = {}
        for seg_value, counts in question_counts[qid]["segments"][field].items():
            denom = question_denoms[qid]["segments"][field][seg_value]
            seg_payload[seg_value] = counter_to_share(counts, denom)
        segments_payload[field] = seg_payload
    question_data[qid] = {
        "meta": meta,
        "overall": overall,
        "respondent_count": question_denoms[qid]["overall"],
        "segments": segments_payload,
    }

metric_summary = []
for key, meta in METRIC_KEYS.items():
    stats = metric_stats.get(key)
    if not stats or stats["count"] == 0:
        continue
    avg = stats["sum"] / stats["count"]

    def rank_list(bucket):
        items = [
            {"segment": seg, "value": agg["sum"] / agg["count"]}
            for seg, agg in bucket.items()
            if agg["count"] > 0 and seg != "unknown"
        ]
        items.sort(key=lambda x: x["value"], reverse=True)
        return items[:3]

    metric_summary.append(
        {
            "key": key,
            "label": meta["label"],
            "unit": meta["unit"],
            "description": meta["description"],
            "insight": meta.get("insight", ""),
            "average": round(avg, 2),
            "min": round(stats["min"], 2) if stats["min"] != float("inf") else None,
            "max": round(stats["max"], 2) if stats["max"] != float("-inf") else None,
            "top_city_tiers": rank_list(stats["by_city_tier"]),
            "top_income_buckets": rank_list(stats["by_income"]),
            "top_consumption_tiers": rank_list(stats["by_consumption_tier"]),
        }
    )

source_files = []
tracked_files = [CANONICAL_PATH, DERIVED_PATH, MULTISELECT_PATH]
for path in tracked_files:
    if not path.exists():
        continue
    size_mb = os.path.getsize(path) / (1024 * 1024)
    source_files.append(
        {
            "label": path.name,
            "path": str(path),
            "size_mb": round(size_mb, 2),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        }
    )

timestamps = [os.path.getmtime(p) for p in [CANONICAL_PATH, DERIVED_PATH] if p.exists()]
last_updated = max(timestamps) if timestamps else datetime.now().timestamp()

insight_entries = []
if INSIGHT_JSON_PATH.exists():
    with INSIGHT_JSON_PATH.open() as f:
        insight_raw = json.load(f)
    for key, rows in insight_raw.items():
        if "|" in key:
            group, segment = key.split("|", 1)
        else:
            group, segment = "segment", key
        for row in rows:
            insight_entries.append(
                {
                    "group": group.strip(),
                    "segment_key": segment.strip(),
                    "segment_label": row.get("segment_label"),
                    "evidence": row.get("evidence"),
                    "recommendation": row.get("recommendation"),
                    "example_execution": row.get("example_execution"),
                }
            )
elif INSIGHT_SUMMARY_JSON_PATH.exists():
    with INSIGHT_SUMMARY_JSON_PATH.open() as f:
        summary_raw = json.load(f)
    for key, value in summary_raw.items():
        insight_entries.append(
            {
                "group": "summary",
                "segment_key": key,
                "segment_label": key.replace("_", " ").title(),
                "evidence": json.dumps(value, ensure_ascii=False),
                "recommendation": "",
                "example_execution": "",
            }
        )


def parse_stat_sections(path: Path):
    if not path.exists():
        return []
    sections = []
    current = None
    last_field = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": stripped[3:].strip(), "bullets": [], "actions": []}
            last_field = None
            continue
        if not current:
            continue
        if not stripped:
            continue
        if stripped.startswith("- "):
            current["bullets"].append(stripped[2:].strip())
            last_field = "bullet"
            continue
        if stripped.startswith("**Use it:**"):
            text = stripped.split("**Use it:**", 1)[1].strip()
            if text:
                current["actions"].append(text)
                last_field = "action"
            continue
        if stripped.startswith("→"):
            text = stripped.lstrip("→").strip(" -")
            if text:
                current["actions"].append(text)
                last_field = "action"
            continue
        if last_field == "bullet" and current["bullets"]:
            current["bullets"][-1] += f" {stripped}"
        elif last_field == "action" and current["actions"]:
            current["actions"][-1] += f" {stripped}"
    if current:
        sections.append(current)
    return sections


chi_highlights = parse_stat_sections(CHI_MD_PATH)
anova_highlights = parse_stat_sections(ANOVA_MD_PATH)

segment_deltas = []
if SEGMENT_DELTAS_PATH.exists():
    current_category = None
    with SEGMENT_DELTAS_PATH.open() as f:
        for line in f:
            if line.startswith("## "):
                current_category = line[3:].strip()
            elif line.startswith("|") and "---" not in line:
                parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                if len(parts) != 7 or parts[0] == "Segment":
                    continue
                segment_deltas.append(
                    {
                        "category": current_category,
                        "segment": parts[0],
                        "metric": parts[1],
                        "direction": parts[2],
                        "segment_mean": parts[3],
                        "overall_mean": parts[4],
                        "delta": parts[5],
                        "n": parts[6],
                    }
                )

visual_targets = [
    (
        "city_tier_q04_heatmap.csv",
        "Top hair concerns by city tier (Q04)",
        "Shows the share of respondents citing each hair/scalp concern across metros, Tier-1/2/3, and rural cohorts.",
    ),
    (
        "city_tier_q07_heatmap.csv",
        "Routine products by city tier (Q07)",
        "Breaks down the current product stack (shampoo, conditioner, serums, masks) for every city tier.",
    ),
    (
        "city_tier_q12_heatmap.csv",
        "Language demand by city tier (Q12)",
        "Language stack expectations across city tiers for content, packaging, and support.",
    ),
    (
        "city_tier_q23_heatmap.csv",
        "Channel discovery by city tier (Q23)",
        "Marketplace vs brand.com vs social shop preference splits for each city tier.",
    ),
    (
        "consumption_tier_q15_heatmap.csv",
        "Refill triggers by consumption tier (Q15)",
        "Highlights which refill incentives (discounts, eco transparency, convenience) resonate by income/consumption tier.",
    ),
    (
        "consumption_tier_q16_heatmap.csv",
        "Value proof asks by consumption tier (Q16)",
        "Maps the importance of visible results, trusted ingredients, and salon performance by tier.",
    ),
    (
        "consumption_tier_q33_heatmap.csv",
        "Launch offer appeal by consumption tier (Q33)",
        "Shows which launch incentives (deluxe minis, starter kits, subscriptions) spike by tier.",
    ),
]
visual_gallery = []
for filename, title, description in visual_targets:
    path = VISUAL_INPUTS_DIR / filename
    if not path.exists():
        continue
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            label = row[reader.fieldnames[0]]
            values = {}
            for col in reader.fieldnames[1:]:
                val = row[col]
                try:
                    values[col] = float(val)
                except (ValueError, TypeError):
                    values[col] = None
            rows.append({"label": label, "values": values})
        visual_gallery.append(
            {
                "title": title,
                "description": description,
                "file": str(path),
                "columns": reader.fieldnames[1:],
                "rows": rows,
            }
        )


def build_playbook_entries():
    entries = []
    for md_file in sorted(PLAYBOOKS_DIR.glob("*.md")):
        content = md_file.read_text()
        parsed = parse_playbook_markdown(content)
        playbook_id = md_file.stem
        segments = []
        for idx, segment in enumerate(parsed["segments"]):
            fields = segment["fields"]
            labels = {col["key"]: col["label"] for col in segment["columns"]}
            segments.append(
                {
                    "id": f"{playbook_id}_{idx + 1}",
                    "fields": fields,
                    "labels": labels,
                }
            )
        entries.append(
            {
                "id": playbook_id,
                "name": md_file.stem.replace("_", " ").title(),
                "path": str(md_file),
                "title": parsed["title"] or md_file.stem.replace("_", " ").title(),
                "stat_cues": parsed["stat_cues"],
                "segments": segments,
            }
        )
    return entries


playbook_entries = build_playbook_entries()

if METHODOLOGY_PATH.exists():
    methodology_excerpt = "\n".join(METHODOLOGY_PATH.read_text().splitlines()[:200])
else:
    methodology_excerpt = ""

prompt_samples = []
if PROMPT_OUTPUTS_DIR.exists():
    for path in sorted(PROMPT_OUTPUTS_DIR.glob("*"))[:20]:
        if path.is_file():
            prompt_samples.append({"name": path.name, "path": str(path)})

script_inventory = []
if SCRIPTS_DIR.exists():
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        script_inventory.append({"name": path.name, "path": str(path)})

resources = {"prompt_samples": prompt_samples, "scripts": script_inventory}

insight_matrix_markdown = read_text_if_exists(INSIGHT_MATRIX_MD_PATH)
stat_clinic_markdown = read_text_if_exists(STAT_CLINIC_PATH)
macro_guardrails_markdown = read_text_if_exists(MACRO_GUARDRAILS_PATH)
leadership_summary_markdown = read_text_if_exists(LEADERSHIP_SUMMARY_PATH)


def parse_insight_matrix_sections(md_text: str):
    sections = []
    if not md_text:
        return sections
    current = None
    def clean(text: str) -> str:
        return re.sub(r"\[[^\]]+\]", "", text or "").strip()
    for raw_line in md_text.splitlines():
        line = clean(raw_line)
        if line.startswith("## "):
            if current:
                sections.append(current)
            current = {
                "title": line[3:].strip(),
                "bullets": [],
                "why": "",
                "action": "",
            }
            continue
        if not current:
            continue
        if line.startswith("- "):
            current["bullets"].append(line[2:].strip())
            continue
        if line.startswith("**Why it matters:**"):
            current["why"] = clean(line.split("**Why it matters:**", 1)[1])
            continue
        if line.startswith("**Action cue:"):
            current["action"] = clean(
                line.split("**Action cue:", 1)[1]
            ).strip(" *")
            continue
    if current:
        sections.append(current)
    return sections


insight_sections = parse_insight_matrix_sections(insight_matrix_markdown)

hero = {
    "respondent_count": respondent_total,
    "persona_count": respondent_total,
    "city_tier_mix": {
        k: {"count": v, "percent": round((v / respondent_total) * 100, 1)}
        for k, v in segment_counts["city_tier"].items()
    },
    "gender_mix": {
        k: {"count": v, "percent": round((v / respondent_total) * 100, 1)}
        for k, v in gender_counts.items()
    },
    "avg_haircare_spend": round(
        metric_stats.get("pricing_monthly_spend_value", {}).get("sum", 0)
        / metric_stats.get("pricing_monthly_spend_value", {}).get("count", 1),
        2,
    )
    if metric_stats.get("pricing_monthly_spend_value", {}).get("count")
    else None,
    "avg_wash_cadence": round(
        metric_stats.get("cadence_weekly_wash_score", {}).get("sum", 0)
        / metric_stats.get("cadence_weekly_wash_score", {}).get("count", 1),
        2,
    )
    if metric_stats.get("cadence_weekly_wash_score", {}).get("count")
    else None,
    "sample_freshness": datetime.fromtimestamp(last_updated).isoformat(),
    "source_files": source_files,
}

segment_distribution = {
    field: {
        k: {"count": v, "percent": round((v / respondent_total) * 100, 1)}
        for k, v in counts.items()
    }
    for field, counts in segment_counts.items()
}

payload = {
    "meta": hero,
    "segments": segment_distribution,
    "cluster_context": cluster_context,
    "personas": persona_entries,
    "questions": question_data,
    "metrics": metric_summary,
    "insights": insight_entries,
    "chi_square": chi_highlights,
    "anova": anova_highlights,
    "segment_deltas": segment_deltas,
    "visual_gallery": visual_gallery,
    "playbooks": playbook_entries,
    "insight_sections": insight_sections,
    "insight_matrix_markdown": insight_matrix_markdown,
    "stat_clinic": stat_clinic_markdown,
    "macro_guardrails": macro_guardrails_markdown,
    "leadership_summary": leadership_summary_markdown,
    "methodology_excerpt": methodology_excerpt,
    "resources": resources,
}

output_js = "window.SEEKHO_DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n"
(ASSETS_DIR / "dashboard_data.js").write_text(output_js)
print("dashboard_data.js refreshed with payload keys:", list(payload.keys()))
