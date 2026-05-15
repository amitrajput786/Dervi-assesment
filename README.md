# Deriv Social Media Monitoring Pipeline

A replayable multi-stage pipeline that ingests social media posts about Deriv,
classifies sentiment and topics, detects emerging narratives, computes risk scores,
routes escalations, and drafts public responses.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your-key-here" > .env
```

## Run Pipeline

```bash
python pipeline.py
```

## Validate Output

```bash
python validate.py
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| POSTS_LOADED | Read posts.json from disk | — |
| MULTILINGUAL_PREPROCESSING_COMPLETE | Detect & translate non-English posts | preprocessed_posts.json |
| POSTS_CLASSIFIED | LLM classifies sentiment, topic, urgency | classified_posts.json |
| NARRATIVES_DETECTED | LLM identifies emerging narrative clusters | narratives.json |
| RISK_SCORES_COMPUTED | Deterministic formula scores each post | risk_scores.json |
| ESCALATIONS_SELECTED | Top 5 posts flagged | risk_scores.json |
| ROUTING_COMPLETE | LLM routes escalations to internal teams | escalation_routing.json |
| RESPONSE_DRAFTS_GENERATED | LLM drafts public responses with send gates | response_drafts.md |
| RESULTS_FINALISED | All artifacts saved | llm_calls.jsonl |

## Generated Artifacts

- `posts.json` — input posts
- `preprocessed_posts.json` — translated posts
- `classified_posts.json` — sentiment/topic/urgency per post
- `narratives.json` — emerging narrative clusters
- `risk_scores.json` — deterministic risk scores, top 5 flagged
- `escalation_routing.json` — team routing with briefing notes
- `response_drafts.md` — public response drafts with send gates
- `sentiment_trend.json` — sentiment arc over time
- `competitor_signals.json` — competitor switching signals
- `crisis_rating.json` — overall crisis severity (green/yellow/orange/red)
- `monitoring_plan.md` — 24-hour monitoring plan
- `llm_calls.jsonl` — audit log of all LLM calls

## Notes

- Risk scoring is fully deterministic (pure Python math, no LLM)
- All LLM outputs are validated against controlled vocabularies
- Pipeline is fully replayable — delete artifacts and re-run
- Replace posts.json with any equivalent dataset using the same schema
