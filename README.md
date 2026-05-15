# Deriv Social Media Monitoring Pipeline

A replayable multi-stage AI pipeline that ingests social media posts about Deriv,
classifies sentiment and topics, detects emerging narratives, computes deterministic
risk scores, routes escalations to internal teams, and drafts public responses with
human send gates.

---

## Tech Stack

- **Language:** Python 3.10+
- **LLM:** `gemini-2.0-flash` via Google Gemini API (`google-genai` SDK)
- **Dependencies:** `google-genai`, `python-dotenv`

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/amitrajput786/Dervi-assesment.git
cd Dervi-assesment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env
```

Get your free Gemini API key at: https://aistudio.google.com

---

## Run Pipeline

```bash
python pipeline.py
```

## Validate Output

```bash
python validate.py
```

---

## Pipeline Stages

| Stage | Description | Output Artifact |
|-------|-------------|-----------------|
| `INIT` | Pipeline initialised | — |
| `POSTS_LOADED` | Read posts.json from disk | — |
| `MULTILINGUAL_PREPROCESSING_COMPLETE` | Detect & translate non-English posts (e.g. P10 Malay) | `preprocessed_posts.json` |
| `POSTS_CLASSIFIED` | LLM classifies sentiment, topic, urgency per post | `classified_posts.json` |
| `NARRATIVES_DETECTED` | LLM identifies emerging narrative clusters from classified data | `narratives.json` |
| `RISK_SCORES_COMPUTED` | Deterministic Python formula scores each post | `risk_scores.json` |
| `ESCALATIONS_SELECTED` | Top 5 posts flagged by risk score | `risk_scores.json` |
| `ROUTING_COMPLETE` | LLM routes escalations to internal teams with briefing notes | `escalation_routing.json` |
| `RESPONSE_DRAFTS_GENERATED` | LLM drafts public responses with human send gates | `response_drafts.md` |
| `RESULTS_FINALISED` | All artifacts saved and verified | — |

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `posts.json` | Input social media posts |
| `preprocessed_posts.json` | Translated posts with original text preserved |
| `classified_posts.json` | Sentiment, topic, urgency per post |
| `narratives.json` | Emerging narrative clusters with trend estimates |
| `risk_scores.json` | Deterministic risk scores, top 5 flagged for escalation |
| `escalation_routing.json` | Team routing with Slack-style briefing notes |
| `response_drafts.md` | Public response drafts with send gate requirements |
| `sentiment_trend.json` | Sentiment arc and inflection point analysis |
| `competitor_signals.json` | Posts suggesting users considering alternatives |
| `crisis_rating.json` | Overall crisis severity: green / yellow / orange / red |
| `monitoring_plan.md` | 24-hour keyword and platform monitoring plan |
| `llm_calls.jsonl` | Audit log of all LLM calls with prompt hashes |

---

## Controlled Vocabularies

All LLM outputs are validated against these allowed values in code:

- **Sentiment:** `positive` `negative` `neutral` `mixed`
- **Topic:** `withdrawal` `account_suspension` `spread_pricing` `product_feedback` `regulatory` `technical` `deposit` `kyc` `general`
- **Urgency:** `critical` `high` `medium` `low`
- **Teams:** `Customer Support` `Legal` `Compliance` `PR/Comms` `Product` `Engineering` `Finance`
- **Narrative strength:** `strong` `moderate` `weak`

---

## Risk Scoring Formula

Risk scoring is **fully deterministic** — no LLM involved:

```
base_risk:
  critical = 40 | high = 25 | medium = 10 | low = 3

raw_engagement = likes + reposts*2 + comments*1.5 + replies*1.5
               + upvotes + helpful_votes + reactions

engagement_multiplier = normalised 1.0 to 3.0 across all posts

legal_threat_bonus    = 20 if contains_legal_threat
narrative_bonus       = 15 per narrative the post belongs to

risk_score = (base * engagement_multiplier) + legal_threat_bonus + narrative_bonus
```

Top 5 posts by risk score are flagged for escalation.

---

## LLM Call Stages

Separate LLM calls are made for each stage and logged to `llm_calls.jsonl`:

1. `multilingual_preprocessing` — translate non-English posts
2. `post_classification` — classify all posts in one call
3. `narrative_detection` — detect narratives from classified data
4. `escalation_routing` — route top 5 posts to internal teams
5. `response_drafting` — draft public responses for critical/legal posts

---

## Key Design Decisions

- **Replayable:** Delete all output files and re-run — pipeline regenerates everything
- **Staged:** Each stage saves its artifact before the next stage begins
- **Deterministic risk scoring:** Formula in pure Python, never delegated to LLM
- **Vocabulary enforcement:** Invalid LLM outputs are corrected in code
- **Multilingual:** Non-English posts translated before classification, original text preserved
- **Human send gates:** Every public response draft includes approval requirements before posting
- **Evaluator-ready:** Replace `posts.json` with any equivalent dataset — pipeline adapts

---

## Project Structure

```
Deriv-assesment/
├── pipeline.py              # Main pipeline (11 stages)
├── validate.py              # Validation checks
├── posts.json               # Input data
├── requirements.txt         # Dependencies
├── README.md                # This file
├── .env.example             # API key template
├── .gitignore               # Excludes .env
└── [generated artifacts]    # Created on run
```

---

## About

Built as part of a technical assessment for Deriv — an AI Engineer role.
The pipeline demonstrates multi-stage LLM orchestration, deterministic scoring,
controlled vocabulary validation, multilingual handling, and crisis communication workflows.