# msww-variablematcher

A Python package for matching survey variables between SPSS SAV files using embeddings and Google Cloud AI (Gemini) verification.

## Overview

For each **target** variable, the matcher:

1. Finds the top-k candidates by embedding similarity
2. Sends the target + candidates to the LLM in a single pass
3. The LLM picks the best match (or none), decides if a recode is needed, and designs the recode if so

## Installation

```bash
pip install msww-variablematcher
```

Or install from source:

```bash
git clone https://github.com/msww/variablematcher.git
cd variablematcher
pip install -e .
```

## Requirements

- Python 3.11+
- Google Cloud Platform project with:
  - Vertex AI API enabled
  - BigQuery API enabled
  - Appropriate IAM permissions

## Quick Start

```python
from variablematcher import Survey, VariableMatcher, GCPConfig
import pyreadstat

# Configure GCP
config = GCPConfig(project_id='your-gcp-project')

# Load surveys from SAV files
df_target, meta_target = pyreadstat.read_sav('target_data.sav')
target = Survey.from_sav(df_target, meta_target)

df_cand, meta_cand = pyreadstat.read_sav('candidate_data.sav')
candidate = Survey.from_sav(df_cand, meta_cand)

# Fit (encode + similarity matrix) then predict (LLM + recodes)
matcher = VariableMatcher(gcp_config=config)
result = matcher.fit(target, candidate).predict()

# Inspect per-variable match results
for m in result.matches:
    print(f"\nTarget: {m.target_variable}")
    print(f"  Matched to: {m.candidate_variable}")
    print(f"  Similarity: {m.similarity_score:.3f}")
    print(f"  Confidence: {m.match_confidence:.3f}")
    print(f"  Needs recode: {m.needs_recode}")

# Export recoded surveys back to .sav
df_t, meta_t = result.target.to_sav()
pyreadstat.write_sav(df_t, 'target_recoded.sav',
                     column_labels=meta_t.column_labels,
                     variable_value_labels=meta_t.variable_value_labels)

df_c, meta_c = result.candidate.to_sav()
pyreadstat.write_sav(df_c, 'candidate_recoded.sav',
                     column_labels=meta_c.column_labels,
                     variable_value_labels=meta_c.variable_value_labels)
```

## Configuration

### Environment Variables

```bash
export GCP_PROJECT_ID="your-gcp-project"
export GCP_LOCATION="us-central1"  # or your preferred region
export BQ_DATASET="variable_matcher"  # BigQuery dataset for batch jobs
```

### Programmatic Configuration

```python
from variablematcher import GCPConfig

config = GCPConfig(
    project_id='your-gcp-project',
    location='us-central1',
    bq_dataset='variable_matcher',
    model_name='gemini-2.5-flash',  # Gemini model for verification
    embedding_model='text-embedding-005',  # Embedding model
    embedding_dimensions=768,
    poll_interval=30,  # Seconds between batch job status checks
)
```

## Core Classes

### Survey

Container for survey data and metadata loaded from SAV files.

```python
from variablematcher import Survey
import pyreadstat

# Load from SAV file (stores both DataFrame and metadata)
df, meta = pyreadstat.read_sav('data.sav')
survey = Survey.from_sav(df, meta)

# Create from dictionary (no DataFrame — to_sav() will raise)
survey = Survey.from_dict([
    {'Q1_Age': {1: '18-24', 2: '25-34', 3: '35-44'}},
    {'Q2_Gender': ['Male', 'Female', 'Other']},
])

# Access variables
for var in survey.variables:
    print(f"{var.name}: {var.label}")
    if var.values:
        for val in var.values:
            print(f"  {val.code}: {val.label}")

# Filter variables
categorical_vars = survey.filter_variables(has_values=True)
age_vars = survey.filter_variables(label_contains='age')

# Export to SAV (returns DataFrame + metadata in read format)
df_out, meta_out = survey.to_sav()
pyreadstat.write_sav(df_out, 'output.sav',
                     column_labels=meta_out.column_labels,
                     variable_value_labels=meta_out.variable_value_labels)
```

### VariableMatcher

Main class for matching target variables against a candidate survey. Uses a `fit` / `predict` API.

```python
from variablematcher import VariableMatcher, GCPConfig

config = GCPConfig(project_id='my-project')
matcher = VariableMatcher(
    gcp_config=config,
    min_ratio=0.8,  # Candidates must score >= 80% of best similarity
    top_k=5,  # Number of candidates per target variable
    include_values_in_encoding=True,  # Include value labels in embeddings
)

# fit: encode surveys + compute similarity matrix
matcher.fit(target, candidate)

# predict: select candidates, run LLM, apply recodes, return result
result = matcher.predict()

# Or chain: result = matcher.fit(target, candidate).predict()

# Match specific target variables only
result = matcher.fit(
    target, candidate, variable_names=['Q1_Age', 'Q2_Gender']
).predict()
```

### MatchResult

Contains both (possibly recoded) surveys and per-variable match metadata.

```python
result = matcher.fit(target, candidate).predict()

# The recoded surveys
result.target     # Survey (recoded if needed)
result.candidate  # Survey (recoded if needed)

# Per-variable match decisions
for m in result.matches:
    print(f"Target: {m.target_variable}")
    print(f"  Candidate: {m.candidate_variable}")
    print(f"  Is match: {m.is_match}")
    print(f"  Similarity: {m.similarity_score:.3f}")
    print(f"  Confidence: {m.match_confidence:.3f}")
    if m.needs_recode:
        print(f"  Standardised label: {m.standardised_label}")
        print(f"  Target recodes: {m.target_recodes}")
        print(f"  Candidate recodes: {m.candidate_recodes}")

# Convenience filters
result.matched    # list[VariableMatch] — only successful matches
result.unmatched  # list[VariableMatch] — unmatched target variables
```

### EmbeddingEncoder

Generates embeddings for variables and values using the Google GenAI API.

```python
from variablematcher import EmbeddingEncoder, GCPConfig

config = GCPConfig(project_id='my-project')
encoder = EmbeddingEncoder(
    gcp_config=config,
    batch_size=250,  # Max texts per API call
)

# Encode a survey (returns dict of variable name → embedding)
embeddings = encoder.encode_survey(survey)

# Encode specific texts
embeddings = encoder.encode_texts([
    'What is your age?',
    'How old are you?',
    'Age group',
])

# Encode variables
embeddings = encoder.encode_variables(survey.variables)
```

## Pipeline Flow

1. **Load Surveys**: Load SAV files into Survey objects (DataFrame + metadata)
2. **fit()**: Encode both surveys, compute cosine similarity matrix
3. **predict()**: Select top-k candidates per target, run LLM batch job, apply recodes to both surveys
4. **Export**: Call `to_sav()` on result.target / result.candidate

```
Target SAV ──> Survey.from_sav() ──┐
                                   ├──> fit() ──> predict() ──> MatchResult
Candidate SAV ──> Survey.from_sav() ┘                              │
                                                          ┌────────┴────────┐
                                                     result.target    result.candidate
                                                          │                 │
                                                     .to_sav()         .to_sav()
```
