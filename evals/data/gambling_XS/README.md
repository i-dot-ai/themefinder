# Gambling XS Dataset

**This dataset contains synthetic data for evaluation purposes only.**

## Overview

- **Size**: XS (100 responses)
- **Topic**: Gambling advertising regulation
- **Questions**: 2 question parts

## Structure

```
gambling_XS/
├── inputs/
│   ├── question_part_1/
│   │   ├── question.json      # Survey question text
│   │   └── responses.jsonl    # 100 synthetic responses
│   ├── question_part_2/
│   │   ├── question.json
│   │   └── responses.jsonl
│   └── respondents.jsonl      # Demographic data
└── outputs/
    └── mapping/
        └── 2025-07-22/
            ├── question_part_1/
            │   ├── themes.json           # Ground truth themes
            │   ├── sentiment.jsonl       # Expected positions
            │   ├── mapping.jsonl         # Expected theme mappings
            │   └── detail_detection.jsonl
            └── question_part_2/
                └── ...
```

## Usage

Used by the evaluation scripts in `evals/` to test ThemeFinder pipeline stages:

- **Generation**: Creates themes from responses
- **Sentiment**: Classifies response positions (AGREE/DISAGREE)
- **Mapping**: Assigns themes to responses
- **Condensation**: Reduces theme count
- **Refinement**: Improves theme descriptions

## Field Conventions

All files use ThemeFinder field naming:

| Field | Description |
|-------|-------------|
| `response_id` | Unique response identifier |
| `response` | Response text |
| `topic_id` | Theme identifier (A, B, C...) |
| `topic_label` | Short theme name |
| `topic_description` | Detailed theme description |
| `topic` | Combined `label: description` |
| `position` | Sentiment (AGREE/DISAGREE) |
| `labels` | List of assigned theme IDs |
