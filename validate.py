"""
Validation script for Deriv Social Media Monitoring Pipeline.
Run: python validate.py
"""

import json
import os
import sys

ERRORS = []
PASSES = []

def check(condition, message):
    if condition:
        PASSES.append(f"  ✓ {message}")
    else:
        ERRORS.append(f"  ✗ {message}")

def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        ERRORS.append(f"  ✗ {path} is not valid JSON: {e}")
        return None

# ─── CONTROLLED VOCABULARIES ──────────────────────────────────────────────────
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
ALLOWED_TOPICS = {"withdrawal", "account_suspension", "spread_pricing",
                  "product_feedback", "regulatory", "technical",
                  "deposit", "kyc", "general"}
ALLOWED_URGENCY = {"critical", "high", "medium", "low"}
ALLOWED_TEAMS = {"Customer Support", "Legal", "Compliance",
                 "PR/Comms", "Product", "Engineering", "Finance"}
ALLOWED_NARRATIVE_STRENGTHS = {"strong", "moderate", "weak"}

REQUIRED_ARTIFACTS = [
    "posts.json",
    "preprocessed_posts.json",
    "classified_posts.json",
    "narratives.json",
    "risk_scores.json",
    "escalation_routing.json",
    "response_drafts.md",
    "llm_calls.jsonl"
]

OPTIONAL_ARTIFACTS = [
    "sentiment_trend.json",
    "competitor_signals.json",
    "crisis_rating.json",
    "monitoring_plan.md"
]

REQUIRED_LLM_STAGES = {
    "multilingual_preprocessing",
    "post_classification",
    "narrative_detection",
    "escalation_routing",
    "response_drafting"
}

print("\n" + "="*60)
print("  DERIV PIPELINE VALIDATION")
print("="*60)

# ─── 1. REQUIRED ARTIFACTS EXIST ─────────────────────────────────────────────
print("\n[1] Required Artifacts")
for artifact in REQUIRED_ARTIFACTS:
    check(os.path.exists(artifact), f"{artifact} exists")

print("\n[2] Optional Artifacts")
for artifact in OPTIONAL_ARTIFACTS:
    exists = os.path.exists(artifact)
    status = "✓" if exists else "○"
    print(f"  {status} {artifact} {'(present)' if exists else '(not attempted)'}")

# ─── 2. JSON VALIDITY ─────────────────────────────────────────────────────────
print("\n[3] JSON Validity")
posts = load_json_safe("posts.json")
preprocessed = load_json_safe("preprocessed_posts.json")
classified = load_json_safe("classified_posts.json")
narratives = load_json_safe("narratives.json")
risk_scores = load_json_safe("risk_scores.json")
routing = load_json_safe("escalation_routing.json")

for name, data in [("posts", posts), ("preprocessed", preprocessed),
                   ("classified", classified), ("narratives", narratives),
                   ("risk_scores", risk_scores), ("routing", routing)]:
    check(data is not None, f"{name}.json is valid JSON")

# ─── 3. ALL POSTS PROCESSED ───────────────────────────────────────────────────
print("\n[4] Post Coverage")
if posts and preprocessed and classified:
    post_ids = {p["id"] for p in posts}
    pre_ids = {p["post_id"] for p in preprocessed}
    cls_ids = {p["post_id"] for p in classified}

    check(post_ids == pre_ids, f"All {len(post_ids)} posts preprocessed")
    check(post_ids == cls_ids, f"All {len(post_ids)} posts classified")

# ─── 4. NON-ENGLISH POSTS PRESERVED ──────────────────────────────────────────
print("\n[5] Multilingual Handling")
if preprocessed:
    translated = [p for p in preprocessed if p.get("translated")]
    for p in translated:
        check(
            p.get("original_text") and p.get("text_for_classification") and
            p["original_text"] != p["text_for_classification"],
            f"{p['post_id']}: original text preserved and translated text differs"
        )
    # P10 must be translated
    p10 = next((p for p in preprocessed if p["post_id"] == "P10"), None)
    if p10:
        check(p10.get("translated") == True, "P10 (Malay) is marked as translated")
        check(p10.get("original_language") != "en", "P10 original language is not English")

# ─── 5. CONTROLLED VOCABULARIES ──────────────────────────────────────────────
print("\n[6] Controlled Vocabularies")
if classified:
    bad_sentiment = [c for c in classified if c.get("sentiment") not in ALLOWED_SENTIMENTS]
    bad_topic = [c for c in classified if c.get("topic") not in ALLOWED_TOPICS]
    bad_urgency = [c for c in classified if c.get("urgency") not in ALLOWED_URGENCY]

    check(len(bad_sentiment) == 0, f"All sentiments use controlled vocabulary (bad: {[b['post_id'] for b in bad_sentiment]})")
    check(len(bad_topic) == 0, f"All topics use controlled vocabulary (bad: {[b['post_id'] for b in bad_topic]})")
    check(len(bad_urgency) == 0, f"All urgency values use controlled vocabulary (bad: {[b['post_id'] for b in bad_urgency]})")

if narratives:
    bad_strength = [n for n in narratives if n.get("narrative_strength") not in ALLOWED_NARRATIVE_STRENGTHS]
    check(len(bad_strength) == 0, f"All narrative strengths use controlled vocabulary")
    check(len(narratives) >= 3, f"At least 3 narratives detected (found {len(narratives)})")

if routing:
    for item in routing:
        bad_teams = [t for t in item.get("teams", []) if t not in ALLOWED_TEAMS]
        check(len(bad_teams) == 0, f"{item['post_id']}: teams use controlled vocabulary")

# ─── 6. NARRATIVE DETECTION USED CLASSIFIED DATA ─────────────────────────────
print("\n[7] Pipeline Stage Ordering")
llm_stages = []
if os.path.exists("llm_calls.jsonl"):
    with open("llm_calls.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    llm_stages.append(json.loads(line))
                except:
                    pass

    stage_names = [s["stage"] for s in llm_stages]

    # narrative detection must come after classification
    if "post_classification" in stage_names and "narrative_detection" in stage_names:
        cls_idx = stage_names.index("post_classification")
        nar_idx = stage_names.index("narrative_detection")
        check(cls_idx < nar_idx, "Narrative detection runs after post classification")

    # narrative input must include classified posts
    for s in llm_stages:
        if s["stage"] == "narrative_detection":
            check(
                "classified_posts.json" in s.get("input_artifacts", []),
                "Narrative detection uses classified_posts.json as input"
            )

# ─── 7. RISK SCORES COMPUTED AFTER NARRATIVES ────────────────────────────────
print("\n[8] Risk Score Integrity")
if risk_scores:
    check(
        all("risk_score" in r for r in risk_scores),
        "All posts have risk scores"
    )
    check(
        all("engagement_multiplier" in r for r in risk_scores),
        "Engagement multiplier computed for all posts"
    )
    # Top 5 flagged
    escalated = [r for r in risk_scores if r.get("escalate")]
    check(len(escalated) == 5, f"Exactly 5 posts flagged for escalation (found {len(escalated)})")

    # Verify top 5 are highest scores
    sorted_scores = sorted(risk_scores, key=lambda x: x["risk_score"], reverse=True)
    top5_ids = {r["post_id"] for r in sorted_scores[:5]}
    escalated_ids = {r["post_id"] for r in escalated}
    check(top5_ids == escalated_ids, "Escalated posts are the top 5 by risk score")

# ─── 8. ROUTING USES TOP 5 ────────────────────────────────────────────────────
print("\n[9] Escalation Routing")
if routing and risk_scores:
    escalated_ids = {r["post_id"] for r in risk_scores if r.get("escalate")}
    routed_ids = {r["post_id"] for r in routing}
    check(
        routed_ids.issubset(escalated_ids),
        f"All routed posts are from escalated set"
    )
    for item in routing:
        check(
            bool(item.get("briefing_note")),
            f"{item['post_id']}: has briefing note"
        )

# ─── 9. RESPONSE DRAFTS ───────────────────────────────────────────────────────
print("\n[10] Response Drafts")
if classified and os.path.exists("response_drafts.md"):
    critical_or_legal = [
        c["post_id"] for c in classified
        if c.get("urgency") == "critical" or c.get("contains_legal_threat")
    ]
    with open("response_drafts.md") as f:
        draft_content = f.read()

    for pid in critical_or_legal:
        check(pid in draft_content, f"{pid}: draft response exists in response_drafts.md")

    check("SEND GATE" in draft_content or "send_gate" in draft_content.lower(),
          "Send gate notes present in response drafts")

# ─── 10. LLM CALL LOG ─────────────────────────────────────────────────────────
print("\n[11] LLM Call Log")
if llm_stages:
    logged_stages = {s["stage"] for s in llm_stages}
    for required_stage in REQUIRED_LLM_STAGES:
        check(required_stage in logged_stages, f"LLM log contains stage: {required_stage}")

    for s in llm_stages:
        check(
            all(k in s for k in ["stage", "timestamp", "provider", "model", "prompt_hash"]),
            f"Stage '{s['stage']}' log record has all required fields"
        )

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  VALIDATION SUMMARY")
print("="*60)
for p in PASSES:
    print(p)
if ERRORS:
    print("\nFAILED CHECKS:")
    for e in ERRORS:
        print(e)
    print(f"\n  {len(PASSES)} passed, {len(ERRORS)} failed")
    sys.exit(1)
else:
    print(f"\n  All {len(PASSES)} checks passed ✓")
    sys.exit(0)
