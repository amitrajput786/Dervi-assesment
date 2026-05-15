"""
Deriv Social Media Monitoring Pipeline
Stages: INIT -> POSTS_LOADED -> MULTILINGUAL_PREPROCESSING_COMPLETE ->
        POSTS_CLASSIFIED -> NARRATIVES_DETECTED -> RISK_SCORES_COMPUTED ->
        ESCALATIONS_SELECTED -> ROUTING_COMPLETE -> RESPONSE_DRAFTS_GENERATED ->
        VALIDATION_COMPLETE -> RESULTS_FINALISED
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-3-flash-preview"
LLM_LOG = []

# ─── CONTROLLED VOCABULARIES ──────────────────────────────────────────────────
ALLOWED_SENTIMENTS          = {"positive", "negative", "neutral", "mixed"}
ALLOWED_TOPICS              = {"withdrawal", "account_suspension", "spread_pricing",
                                "product_feedback", "regulatory", "technical",
                                "deposit", "kyc", "general"}
ALLOWED_URGENCY             = {"critical", "high", "medium", "low"}
ALLOWED_TEAMS               = {"Customer Support", "Legal", "Compliance",
                                "PR/Comms", "Product", "Engineering", "Finance"}
ALLOWED_NARRATIVE_STRENGTHS = {"strong", "moderate", "weak"}

# ─── PIPELINE STATE ───────────────────────────────────────────────────────────
PIPELINE_STAGE = "INIT"

def set_stage(stage: str):
    global PIPELINE_STAGE
    PIPELINE_STAGE = stage
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage}")
    print(f"{'='*60}")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()

def llm_call(stage: str, prompt: str, input_artifacts: list, output_artifact: str) -> str:
    """Make LLM call using native google-genai SDK and log it."""
    print(f"  LLM call -> {stage} ...")
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    result = response.text

    LLM_LOG.append({
        "stage":            stage,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "provider":         "google",
        "model":            MODEL,
        "prompt_hash":      hash_prompt(prompt),
        "input_artifacts":  input_artifacts,
        "output_artifact":  output_artifact
    })
    return result

def parse_json_from_llm(raw: str) -> any:
    """Strip markdown code fences and parse JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)

def validate_vocab(value, allowed, field, post_id):
    """Validate and fix controlled vocabulary."""
    if value not in allowed:
        lower = value.lower().strip()
        if lower in allowed:
            return lower
        print(f"  WARNING: {post_id} {field}='{value}' not in vocab, defaulting")
        return list(allowed)[0]
    return value

def get_engagement_score(engagement: dict) -> float:
    """Compute raw engagement from any platform's engagement fields."""
    return (
        engagement.get("likes", 0) +
        engagement.get("reposts", 0) * 2 +
        engagement.get("comments", 0) * 1.5 +
        engagement.get("replies", 0) * 1.5 +
        engagement.get("upvotes", 0) +
        engagement.get("helpful_votes", 0) +
        engagement.get("reactions", 0)
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD POSTS
# ══════════════════════════════════════════════════════════════════════════════
def stage_load_posts():
    set_stage("POSTS_LOADED")
    posts = load_json("posts.json")
    print(f"  Loaded {len(posts)} posts")
    return posts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MULTILINGUAL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def stage_multilingual_preprocessing(posts: list) -> list:
    set_stage("MULTILINGUAL_PREPROCESSING_COMPLETE")

    non_english_ids = []
    for post in posts:
        text = post["text"]
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii > len(text) * 0.1:
            non_english_ids.append(post["id"])

    if "P10" not in non_english_ids:
        non_english_ids.append("P10")

    print(f"  Non-English posts detected: {non_english_ids}")

    non_english_posts = [p for p in posts if p["id"] in non_english_ids]

    if non_english_posts:
        translation_prompt = f"""You are a translation assistant. Detect the language of each post and translate it to English.

Posts to translate:
{json.dumps([{"id": p["id"], "text": p["text"]} for p in non_english_posts], indent=2)}

Return ONLY a JSON array with this exact schema for each post:
[
  {{
    "post_id": "string",
    "original_language": "ISO 639-1 code like ms, ar, pt",
    "translated_text": "English translation here"
  }}
]

No markdown, no explanation, just the JSON array."""

        raw = llm_call(
            stage="multilingual_preprocessing",
            prompt=translation_prompt,
            input_artifacts=["posts.json"],
            output_artifact="preprocessed_posts.json"
        )
        translations = {t["post_id"]: t for t in parse_json_from_llm(raw)}
    else:
        translations = {}

    preprocessed = []
    for post in posts:
        if post["id"] in translations:
            t = translations[post["id"]]
            preprocessed.append({
                "post_id":                post["id"],
                "original_text":          post["text"],
                "text_for_classification": t["translated_text"],
                "original_language":      t["original_language"],
                "translated":             True
            })
        else:
            preprocessed.append({
                "post_id":                post["id"],
                "original_text":          post["text"],
                "text_for_classification": post["text"],
                "original_language":      "en",
                "translated":             False
            })

    save_json(preprocessed, "preprocessed_posts.json")
    print(f"  Preprocessed {len(preprocessed)} posts")
    return preprocessed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — POST CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def stage_classify_posts(posts: list, preprocessed: list) -> list:
    set_stage("POSTS_CLASSIFIED")

    pre_lookup = {p["post_id"]: p for p in preprocessed}

    classification_prompt = f"""You are a social media analyst for Deriv, a fintech trading platform.
Classify each post using ONLY the controlled vocabularies provided below.

CONTROLLED VOCABULARIES:
- sentiment: positive, negative, neutral, mixed
- topic: withdrawal, account_suspension, spread_pricing, product_feedback, regulatory, technical, deposit, kyc, general
- urgency: critical, high, medium, low

URGENCY GUIDE:
- critical: legal threats, regulator complaints, large locked funds, legal action threatened
- high: withdrawal delays over 7 days, account suspension without legal threat, multiple users affected
- medium: pricing concerns, technical issues, KYC friction, deposit failures
- low: general questions, positive feedback, competitor comparisons

POSTS TO CLASSIFY:
{json.dumps([{
    "post_id": p["post_id"],
    "platform": next(x["platform"] for x in posts if x["id"] == p["post_id"]),
    "text": p["text_for_classification"],
    "timestamp": next(x["timestamp"] for x in posts if x["id"] == p["post_id"])
} for p in preprocessed], indent=2)}

Return ONLY a JSON array. One object per post with EXACTLY this schema:
[
  {{
    "post_id": "string",
    "sentiment": "positive|negative|neutral|mixed",
    "topic": "withdrawal|account_suspension|spread_pricing|product_feedback|regulatory|technical|deposit|kyc|general",
    "urgency": "critical|high|medium|low",
    "contains_legal_threat": true or false,
    "contains_competitor_mention": true or false,
    "original_language": "string",
    "translated": true or false
  }}
]

No markdown, no explanation. Only the JSON array."""

    raw = llm_call(
        stage="post_classification",
        prompt=classification_prompt,
        input_artifacts=["preprocessed_posts.json"],
        output_artifact="classified_posts.json"
    )

    classified = parse_json_from_llm(raw)

    for item in classified:
        pid = item["post_id"]
        item["sentiment"] = validate_vocab(item["sentiment"], ALLOWED_SENTIMENTS, "sentiment", pid)
        item["topic"]     = validate_vocab(item["topic"],     ALLOWED_TOPICS,     "topic",     pid)
        item["urgency"]   = validate_vocab(item["urgency"],   ALLOWED_URGENCY,    "urgency",   pid)
        pre = pre_lookup.get(pid, {})
        item["original_language"] = pre.get("original_language", "en")
        item["translated"]        = pre.get("translated", False)

    save_json(classified, "classified_posts.json")
    print(f"  Classified {len(classified)} posts")
    return classified


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — NARRATIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def stage_detect_narratives(posts: list, classified: list, preprocessed: list) -> list:
    set_stage("NARRATIVES_DETECTED")

    pre_lookup  = {p["post_id"]: p for p in preprocessed}
    post_lookup = {p["id"]: p     for p in posts}

    enriched = []
    for item in classified:
        pid      = item["post_id"]
        original = post_lookup.get(pid, {})
        pre      = pre_lookup.get(pid, {})
        enriched.append({
            "post_id":                 pid,
            "platform":                original.get("platform", ""),
            "timestamp":               original.get("timestamp", ""),
            "engagement_summary":      original.get("engagement", {}),
            "text_for_classification": pre.get("text_for_classification", ""),
            "sentiment":               item["sentiment"],
            "topic":                   item["topic"],
            "urgency":                 item["urgency"],
            "contains_legal_threat":   item["contains_legal_threat"]
        })

    narrative_prompt = f"""You are a social media intelligence analyst for Deriv.
Analyze the classified posts below and identify EMERGING NARRATIVES.
A narrative is a cluster of posts that together suggest a systemic issue, even if individual posts look minor.

CLASSIFIED POSTS (Stage 1 output):
{json.dumps(enriched, indent=2)}

INSTRUCTIONS:
- Identify AT LEAST 3 distinct narratives
- Consider timing, platform spread, engagement weight, and topic clustering
- narrative_strength must be one of: strong, moderate, weak
- estimated_hours_until_trending: integer estimate based on engagement velocity

Return ONLY a JSON array with EXACTLY this schema:
[
  {{
    "narrative_id": "N01",
    "title": "short descriptive title",
    "supporting_post_ids": ["P01", "P07"],
    "narrative_strength": "strong|moderate|weak",
    "estimated_hours_until_trending": 6,
    "recommended_action": "specific action for Deriv team"
  }}
]

No markdown, no explanation. Only the JSON array."""

    raw = llm_call(
        stage="narrative_detection",
        prompt=narrative_prompt,
        input_artifacts=["classified_posts.json", "preprocessed_posts.json"],
        output_artifact="narratives.json"
    )

    narratives = parse_json_from_llm(raw)

    for n in narratives:
        n["narrative_strength"] = validate_vocab(
            n["narrative_strength"], ALLOWED_NARRATIVE_STRENGTHS, "narrative_strength", n["narrative_id"]
        )

    save_json(narratives, "narratives.json")
    print(f"  Detected {len(narratives)} narratives")
    return narratives


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RISK SCORING (DETERMINISTIC — NO LLM)
# ══════════════════════════════════════════════════════════════════════════════
def stage_compute_risk_scores(posts: list, classified: list, narratives: list) -> list:
    set_stage("RISK_SCORES_COMPUTED")

    BASE_RISK = {"critical": 40, "high": 25, "medium": 10, "low": 3}

    narrative_membership = {}
    for n in narratives:
        for pid in n["supporting_post_ids"]:
            narrative_membership[pid] = narrative_membership.get(pid, 0) + 1

    raw_engagements = {p["id"]: get_engagement_score(p["engagement"]) for p in posts}

    min_eng   = min(raw_engagements.values())
    max_eng   = max(raw_engagements.values())
    eng_range = max_eng - min_eng if max_eng != min_eng else 1

    def normalise(raw):
        return 1.0 + ((raw - min_eng) / eng_range) * 2.0

    risk_scores = []
    for item in classified:
        pid             = item["post_id"]
        base            = BASE_RISK.get(item["urgency"], 3)
        raw_eng         = raw_engagements.get(pid, 0)
        eng_multiplier  = normalise(raw_eng)
        legal_bonus     = 20 if item.get("contains_legal_threat") else 0
        narrative_bonus = 15 * narrative_membership.get(pid, 0)
        risk_score      = round((base * eng_multiplier) + legal_bonus + narrative_bonus, 2)

        risk_scores.append({
            "post_id":               pid,
            "urgency":               item["urgency"],
            "sentiment":             item["sentiment"],
            "topic":                 item["topic"],
            "contains_legal_threat": item["contains_legal_threat"],
            "raw_engagement":        raw_eng,
            "engagement_multiplier": round(eng_multiplier, 3),
            "base_risk":             base,
            "legal_bonus":           legal_bonus,
            "narrative_bonus":       narrative_bonus,
            "risk_score":            risk_score,
            "narrative_count":       narrative_membership.get(pid, 0)
        })

    risk_scores.sort(key=lambda x: x["risk_score"], reverse=True)
    for i, item in enumerate(risk_scores):
        item["escalate"] = i < 5

    save_json(risk_scores, "risk_scores.json")

    print(f"  Risk scores computed. Top 5 for escalation:")
    for r in [r for r in risk_scores if r["escalate"]]:
        print(f"    {r['post_id']} -> score={r['risk_score']}  urgency={r['urgency']}")

    return risk_scores


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — ESCALATION ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def stage_escalation_routing(posts: list, risk_scores: list, narratives: list) -> list:
    set_stage("ROUTING_COMPLETE")

    post_lookup = {p["id"]: p for p in posts}
    top5        = [r for r in risk_scores if r["escalate"]]

    top5_enriched = []
    for r in top5:
        pid      = r["post_id"]
        original = post_lookup.get(pid, {})
        top5_enriched.append({
            **r,
            "platform":  original.get("platform", ""),
            "text":      original.get("text", ""),
            "timestamp": original.get("timestamp", "")
        })

    routing_prompt = f"""You are a crisis management coordinator for Deriv.
Route the following high-risk social media posts to the appropriate internal teams.

TOP 5 HIGH-RISK POSTS:
{json.dumps(top5_enriched, indent=2)}

DETECTED NARRATIVES:
{json.dumps(narratives, indent=2)}

ALLOWED INTERNAL TEAMS (use EXACTLY these names):
Customer Support, Legal, Compliance, PR/Comms, Product, Engineering, Finance

ROUTING GUIDE:
- Legal: legal threats, chargeback mentions, regulator complaints
- Compliance: KYC issues, regulatory questions, account suspensions
- Customer Support: withdrawal delays, deposit failures, account issues
- PR/Comms: high-visibility posts, trending narratives, brand reputation
- Finance: large locked funds, payment failures
- Product: platform UI/UX issues, feature complaints
- Engineering: technical bugs, execution engine problems, bot issues

Assign ALL relevant teams per post. Write briefing_note as a concise Slack-style internal update.

Return ONLY a JSON array with EXACTLY this schema:
[
  {{
    "post_id": "string",
    "teams": ["Team1", "Team2"],
    "briefing_note": "concise Slack-style internal briefing"
  }}
]

No markdown, no explanation. Only the JSON array."""

    raw = llm_call(
        stage="escalation_routing",
        prompt=routing_prompt,
        input_artifacts=["risk_scores.json", "narratives.json"],
        output_artifact="escalation_routing.json"
    )

    routing = parse_json_from_llm(raw)

    for item in routing:
        item["teams"] = [t for t in item["teams"] if t in ALLOWED_TEAMS]
        if not item["teams"]:
            item["teams"] = ["Customer Support"]

    save_json(routing, "escalation_routing.json")
    print(f"  Routed {len(routing)} escalations")
    return routing


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — PUBLIC RESPONSE DRAFTS
# ══════════════════════════════════════════════════════════════════════════════
def stage_draft_responses(posts: list, classified: list) -> list:
    set_stage("RESPONSE_DRAFTS_GENERATED")

    post_lookup      = {p["id"]: p for p in posts}
    response_targets = [
        item for item in classified
        if item["urgency"] == "critical" or item.get("contains_legal_threat")
    ]

    if not response_targets:
        print("  No posts require public response drafts")
        with open("response_drafts.md", "w") as f:
            f.write("# Response Drafts\n\nNo critical posts requiring response at this time.\n")
        return []

    enriched_targets = []
    for item in response_targets:
        pid      = item["post_id"]
        original = post_lookup.get(pid, {})
        enriched_targets.append({
            **item,
            "platform":  original.get("platform", ""),
            "text":      original.get("text", ""),
            "timestamp": original.get("timestamp", "")
        })

    drafts_prompt = f"""You are a senior communications manager at Deriv.
Draft public-facing responses for the following high-urgency social media posts.

POSTS REQUIRING RESPONSE:
{json.dumps(enriched_targets, indent=2)}

STRICT RULES:
1. Acknowledge the issue empathetically
2. Do NOT admit liability or fault
3. Do NOT disclose any account-specific details
4. Provide clear next steps for the user
5. Match platform tone: Twitter = concise under 280 chars, Reddit = detailed, Trustpilot = formal
6. send_gate_note must specify what internal confirmation is required before posting

Return ONLY a JSON array with EXACTLY this schema:
[
  {{
    "post_id": "string",
    "platform": "string",
    "draft_response": "the public-facing response text",
    "send_gate_note": "what internal info or approval is required before sending"
  }}
]

No markdown, no explanation. Only the JSON array."""

    raw = llm_call(
        stage="response_drafting",
        prompt=drafts_prompt,
        input_artifacts=["classified_posts.json"],
        output_artifact="response_drafts.md"
    )

    drafts = parse_json_from_llm(raw)

    md_lines = [
        "# Deriv Public Response Drafts\n",
        f"Generated: {datetime.now().isoformat()}\n",
        "---\n"
    ]
    for d in drafts:
        md_lines.append(f"## Post {d['post_id']} | {d['platform']}\n")
        md_lines.append(f"**Draft Response:**\n\n{d['draft_response']}\n")
        md_lines.append(f"\n**SEND GATE:** {d['send_gate_note']}\n")
        md_lines.append("\n---\n")

    with open("response_drafts.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"  Drafted responses for {len(drafts)} posts")
    return drafts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SENTIMENT TREND (SHOULD ATTEMPT)
# ══════════════════════════════════════════════════════════════════════════════
def stage_sentiment_trend(posts: list, classified: list):
    set_stage("SENTIMENT_TREND")

    cls_lookup   = {c["post_id"]: c for c in classified}
    sorted_posts = sorted(posts, key=lambda p: p["timestamp"])

    timeline = []
    for post in sorted_posts:
        pid = post["id"]
        cls = cls_lookup.get(pid, {})
        timeline.append({
            "post_id":   pid,
            "timestamp": post["timestamp"],
            "sentiment": cls.get("sentiment", "neutral"),
            "urgency":   cls.get("urgency", "low"),
            "topic":     cls.get("topic", "general")
        })

    from collections import Counter
    sentiment_counts = Counter(t["sentiment"] for t in timeline)

    inflection = None
    window = 3
    for i in range(len(timeline) - window):
        window_sentiments = [timeline[j]["sentiment"] for j in range(i, i + window)]
        if window_sentiments.count("negative") >= 2 and inflection is None:
            inflection = {
                "at_post_id":  timeline[i]["post_id"],
                "timestamp":   timeline[i]["timestamp"],
                "description": f"Negative cluster: {window_sentiments.count('negative')}/{window} posts negative"
            }

    trend_data = {
        "total_posts":            len(timeline),
        "sentiment_distribution": dict(sentiment_counts),
        "timeline":               timeline,
        "inflection_point":       inflection,
        "sentiment_arc":          "predominantly_negative"
                                  if sentiment_counts.get("negative", 0) > len(timeline) * 0.4
                                  else "mixed"
    }

    save_json(trend_data, "sentiment_trend.json")
    print("  Sentiment trend saved")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — COMPETITOR SIGNALS (SHOULD ATTEMPT)
# ══════════════════════════════════════════════════════════════════════════════
def stage_competitor_signals(posts: list, classified: list):
    set_stage("COMPETITOR_SIGNALS")

    cls_lookup = {c["post_id"]: c for c in classified}
    switching_keywords = [
        "alternative", "alternatives", "moved to", "switching",
        "proper platforms", "look at", "other broker", "instead"
    ]

    signals = []
    for post in posts:
        text_lower = post["text"].lower()
        cls        = cls_lookup.get(post["id"], {})

        if (any(kw in text_lower for kw in switching_keywords) or
                cls.get("contains_competitor_mention") or
                (cls.get("sentiment") == "negative" and cls.get("urgency") in ["high", "critical"])):

            trigger = next((kw for kw in switching_keywords if kw in text_lower), "negative experience")
            signals.append({
                "post_id":                post["id"],
                "implied_competitor_type": "regulated forex broker"
                                           if any(w in text_lower for w in ["forex", "spread", "pip"])
                                           else "general trading platform",
                "switching_trigger":      trigger,
                "retention_argument":     "Highlight Deriv's synthetic indices, competitive spreads, and improved support SLAs"
            })

    save_json(signals, "competitor_signals.json")
    print(f"  Found {len(signals)} competitor signals")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — CRISIS SEVERITY RATING (STRETCH)
# ══════════════════════════════════════════════════════════════════════════════
def stage_crisis_rating(classified: list, narratives: list, risk_scores: list):
    set_stage("CRISIS_RATING")

    legal_posts       = [c["post_id"] for c in classified if c.get("contains_legal_threat")]
    critical_posts    = [c["post_id"] for c in classified if c["urgency"] == "critical"]
    strong_narratives = [n for n in narratives if n["narrative_strength"] == "strong"]
    top_scores        = risk_scores[:5]

    if len(legal_posts) >= 2 or len(strong_narratives) >= 2:
        rating        = "red"
        justification = "Multiple legal threats and strong trending narratives detected"
    elif len(legal_posts) >= 1 or len(strong_narratives) >= 1:
        rating        = "orange"
        justification = "Legal threat present or strong narrative emerging"
    elif len(critical_posts) >= 3:
        rating        = "yellow"
        justification = "Multiple critical urgency posts without legal threats"
    else:
        rating        = "green"
        justification = "No immediate crisis indicators"

    posture_map = {
        "red":    "Immediate PR/Comms activation. CEO-level awareness. Proactive public statement within 2 hours.",
        "orange": "PR/Comms on standby. Legal review all flagged accounts. Prepare holding statement.",
        "yellow": "Monitor closely. Customer Support surge resourcing. Internal briefing to leadership.",
        "green":  "Standard monitoring. No immediate action required."
    }

    save_json({
        "crisis_severity_rating": rating,
        "justification":          justification,
        "legal_threat_post_ids":  legal_posts,
        "critical_post_ids":      critical_posts,
        "strong_narrative_ids":   [n["narrative_id"] for n in strong_narratives],
        "top_risk_post_ids":      [r["post_id"] for r in top_scores],
        "recommended_posture":    posture_map.get(rating, "Monitor")
    }, "crisis_rating.json")

    print(f"  Crisis rating: {rating.upper()}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 11 — 24-HOUR MONITORING PLAN (STRETCH)
# ══════════════════════════════════════════════════════════════════════════════
def stage_monitoring_plan(narratives: list, classified: list):
    set_stage("MONITORING_PLAN")

    keywords = set()
    for n in narratives:
        for word in n["title"].lower().split():
            if len(word) > 4:
                keywords.add(word)
    keywords.update([
        "deriv withdrawal", "deriv suspension", "deriv scam",
        "deriv spread", "deriv kyc", "derivscam", "deriv bot",
        "#DerivScam", "#ConsumerRights"
    ])

    platforms = [
        "Twitter/X",
        "Reddit r/Forex",
        "Reddit r/binaryoptions",
        "Trustpilot",
        "Telegram (public group)",
        "Facebook (Nigeria/Malaysia Forex groups)"
    ]

    md_lines = [
        "# 24-Hour Monitoring Plan",
        f"\nGenerated: {datetime.now().isoformat()}\n",
        "---",
        "\n## Keywords to Track\n",
        "\n".join(f"- `{kw}`" for kw in sorted(keywords)),
        "\n\n## Platforms to Prioritise\n",
        "\n".join(f"{i+1}. {p}" for i, p in enumerate(platforms)),
        "\n\n## Escalation Signals\n",
        "- New posts with `#DerivScam` gaining >50 engagements within 1 hour",
        "- Any post mentioning regulator filing or legal action",
        "- Cluster of 3+ posts on same topic within a 2-hour window",
        "- Withdrawal/suspension complaints crossing 10 posts/hour",
        "\n\n## De-escalation Signals\n",
        "- Positive resolution posts appearing after support contact",
        "- Engagement on negative posts dropping below 10/hour",
        "- No new legal threat posts for 4+ consecutive hours",
        "\n\n## Owner Teams & SLAs\n",
        "| Signal Type | Owner Team | Response SLA |",
        "|---|---|---|",
        "| Legal threats | Legal + PR/Comms | 1 hour |",
        "| Account suspension cluster | Compliance + Customer Support | 2 hours |",
        "| Spread/pricing complaints | Product + PR/Comms | 4 hours |",
        "| Deposit/withdrawal failures | Finance + Engineering | 2 hours |",
        "| General negative sentiment | PR/Comms | 6 hours |"
    ]

    with open("monitoring_plan.md", "w") as f:
        f.write("\n".join(md_lines))
    print("  Monitoring plan saved")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE LLM CALL LOG
# ══════════════════════════════════════════════════════════════════════════════
def save_llm_log():
    with open("llm_calls.jsonl", "w") as f:
        for record in LLM_LOG:
            f.write(json.dumps(record) + "\n")
    print(f"\n  LLM call log: {len(LLM_LOG)} records -> llm_calls.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline():
    print("\n" + "█"*60)
    print("  DERIV SOCIAL MEDIA MONITORING PIPELINE")
    print("  Model: gemini-2.0-flash | Provider: Google")
    print("█"*60)

    set_stage("INIT")

    posts        = stage_load_posts()
    preprocessed = stage_multilingual_preprocessing(posts)
    classified   = stage_classify_posts(posts, preprocessed)
    narratives   = stage_detect_narratives(posts, classified, preprocessed)
    risk_scores  = stage_compute_risk_scores(posts, classified, narratives)

    set_stage("ESCALATIONS_SELECTED")

    routing = stage_escalation_routing(posts, risk_scores, narratives)
    drafts  = stage_draft_responses(posts, classified)

    stage_sentiment_trend(posts, classified)
    stage_competitor_signals(posts, classified)
    stage_crisis_rating(classified, narratives, risk_scores)
    stage_monitoring_plan(narratives, classified)

    save_llm_log()

    set_stage("RESULTS_FINALISED")

    print("\n✓ Pipeline complete. All artifacts saved.\n")
    artifacts = [
        "posts.json", "preprocessed_posts.json", "classified_posts.json",
        "narratives.json", "risk_scores.json", "escalation_routing.json",
        "response_drafts.md", "sentiment_trend.json", "competitor_signals.json",
        "crisis_rating.json", "monitoring_plan.md", "llm_calls.jsonl"
    ]
    for a in artifacts:
        status = "✓" if os.path.exists(a) else "✗ MISSING"
        print(f"  {status}  {a}")


if __name__ == "__main__":
    run_pipeline()