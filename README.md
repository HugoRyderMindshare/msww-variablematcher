# msww-variablematcher

Match survey variables between SPSS SAV files using embeddings and Gemini verification.

## Overview

For each **target** variable the matcher:

1. Embeds question text via `gemini-embedding-001` and ranks candidates by cosine similarity
2. Filters candidates by a dtype mask (categorical ↔ categorical only)
3. Sends the target + shortlisted candidates to Gemini via a BigQuery batch job
4. The LLM picks the best match (or none) and designs a recode if the value scales differ
5. Recodes are applied, then both surveys are filtered and reordered to matched columns only

## Installation

```bash
pip install msww-variablematcher
```

## Requirements

- Python 3.11+
- Google Cloud Platform project with Vertex AI and BigQuery APIs enabled
- Environment variables (see [Configuration](#configuration))

## Quick Start

```python
import pyreadstat
from variablematcher import Survey, VariableMatcher

# Load surveys
df_t, meta_t = pyreadstat.read_sav("target.sav")
target = Survey.from_sav(df_t, meta_t)

df_c, meta_c = pyreadstat.read_sav("candidate.sav")
candidate = Survey.from_sav(df_c, meta_c)

# Fit and predict
result = VariableMatcher().fit(target, candidate).predict()

# Inspect matches
for m in result.matches:
    if m.is_match:
        print(f"{m.target_variable} -> {m.candidate_variable}")

# Export filtered/ordered surveys
df_out, meta_out = result.target.to_sav()
pyreadstat.write_sav(df_out, "target_matched.sav",
                     column_labels=meta_out.column_labels,
                     variable_value_labels=meta_out.variable_value_labels)
```

## Configuration

GCP settings are read from environment variables — no config object is passed to the API.

```bash
export GCP_PROJECT_ID="your-project"
export GCP_LOCATION="europe-west2"
export BQ_DATASET="variable_matcher"      # BigQuery dataset for batch jobs
export GCP_MODEL_NAME="gemini-2.5-flash"  # optional, default gemini-2.5-flash
export GCP_POLL_INTERVAL="30"             # optional, seconds between batch polls
```

## Core Classes

### Survey

Container for survey data and metadata. The single source of truth is the
`(DataFrame, pyreadstat metadata)` pair — `variables` is rebuilt from
metadata on every access.

```python
from variablematcher import Survey
import pyreadstat

df, meta = pyreadstat.read_sav("data.sav")
survey = Survey.from_sav(df, meta, name="wave_1")

# Variables are derived from metadata
for var in survey.variables:
    print(f"{var.code}: {var.question}")
    if var.values:
        for val in var.values:
            print(f"  {val.code}: {val.statement}")

# Filter
categorical = survey.filter_variables(has_values=True)
age_vars = survey.filter_variables(label_contains="age")

# Variables without a question are excluded from matching
survey.no_question_variables  # tuple of variable codes

# Export
df_out, meta_out = survey.to_sav()
```

### VariableMatcher

Fit / predict API. Encodes surveys, builds a similarity matrix, runs LLM
verification, applies recodes, and filters both surveys to matched columns.

```python
from variablematcher import VariableMatcher

matcher = VariableMatcher(
    min_ratio=0.8,  # candidates must score >= 80% of best similarity
    top_k=5,        # max candidates per target variable
)

# fit: embed questions, compute cosine similarity, apply dtype mask
# predict: shortlist candidates, LLM batch, apply recodes, filter & order
result = matcher.fit(target, candidate).predict()
```

### MatchResult

Contains both filtered, positionally aligned surveys and per-variable match metadata.

```python
result.target      # Survey — recoded, filtered, and ordered to matched columns
result.candidate   # Survey — recoded, filtered, and ordered to matched columns
result.matches     # list[VariableMatch] — per-variable LLM decisions
```

### VariableMatch

Per-variable outcome from the LLM verification step.

```python
for m in result.matches:
    print(m.target_variable)       # str — target variable code
    print(m.candidate_variable)    # str | None — matched candidate code
    print(m.is_match)              # bool — whether a match was confirmed
    print(m.target_groups)         # dict | None — recode groups for target
    print(m.candidate_groups)      # dict | None — recode groups for candidate
```

## Pipeline Flow

```
Target SAV ──> Survey.from_sav() ──┐
                                   ├──> fit() ──> predict() ──> MatchResult
Candidate SAV ──> Survey.from_sav() ┘                              │
                                                          ┌────────┴────────┐
                                                     result.target    result.candidate
                                                     (filtered &      (filtered &
                                                      ordered)         ordered)
                                                          │                 │
                                                     .to_sav()         .to_sav()
```

1. **fit** — Embed variable questions, compute cosine similarity, zero out cross-dtype pairs
2. **predict** — Shortlist top-k candidates, send to Gemini via BQ batch, parse recode specs, apply recodes, filter and order both surveys to matched columns
3. **Export** — `to_sav()` returns the filtered, positionally aligned data and metadata
