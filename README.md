# Darukaa.Earth — AI-Powered Biodiversity Intelligence Assistant

Darukaa.Earth is an AI-powered environmental intelligence assistant that analyzes environmental conditions and provides evidence-backed recommendations for improving soil health, biodiversity, water availability, and habitat quality.

## Features

- Environmental assessment using:
  - Soil Organic Carbon (%)
  - Annual Rainfall (mm)
  - Temperature (°C)
  - Soil Moisture
  - Land Use / Land Cover
- Scientific knowledge retrieval from a structured environmental knowledge base
- Multi-metric environmental reasoning
- Evidence-backed environmental recommendations
- Biodiversity data integration using the GBIF API
- Natural-language conversational interaction
- Impacted metrics, confidence level, and time horizon in every response

## Multi-Metric Reasoning

The system connects multiple environmental variables to identify relationships between environmental conditions, instead of evaluating each one in isolation.

```
Soil Carbon → Water Retention → Crop Resilience → Biodiversity

Rainfall → Soil Moisture → Vegetation Survival

Monoculture → Reduced Habitat Diversity → Biodiversity Pressure

Soil Health → Vegetation Quality → Habitat Diversity
```

## Environmental Thresholds

The prototype uses threshold-based conditions to identify environmental stressors.

| Condition | Threshold |
|---|---|
| Very Low Soil Organic Carbon | < 0.3% |
| Low Soil Organic Carbon | 0.3% – < 0.5% |
| Low Annual Rainfall | < 800 mm |
| High Temperature | ≥ 30°C |
| Low Soil Moisture | Moisture = Low |

## Confidence Logic

Confidence is determined by the number of detected environmental stressors and relationships between them.

| Evidence Strength | Confidence |
|---|---|
| ≥ 4 | High |
| 2–3 | Moderate |
| < 2 | Limited |

## Knowledge Retrieval

The system uses TF-IDF vectorization and cosine similarity to retrieve relevant environmental knowledge for each query.

```
Environmental Query
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Relevant Knowledge
        ↓
Environmental Reasoning
```

The knowledge base covers:

- Soil Organic Carbon
- Crop Diversification
- Agroforestry
- Water and Soil Management
- Biodiversity and Ecosystem Services
- Pollinator Habitat
- Habitat Fragmentation
- Soil and Water Conservation
- Sustainable Agriculture
- Native Vegetation

## Biodiversity Integration

The system integrates GBIF biodiversity occurrence data to provide regional biodiversity context.

Retrieved information includes:

- Total occurrence records
- Sampled records
- Unique species in the sample
- Sample species names

Note: GBIF species counts are used as regional biodiversity context and are not treated as site-specific species richness for the user's land.

## How It Works

1. **User input** — The user provides environmental data (soil carbon, rainfall, temperature, moisture, land use, country) and asks a question in plain language.
2. **Problem detection** — The system checks the values against defined thresholds to identify stressors such as low soil carbon, water stress, or low habitat diversity.
3. **Multi-metric reasoning** — Detected problems are connected into cause-and-effect chains rather than treated separately.
4. **Knowledge retrieval** — The system searches its scientific knowledge base (FAO, IPBES) for the most relevant information.
5. **Biodiversity context** — The system queries GBIF for regional biodiversity occurrence data based on the country provided.
6. **Recommendation** — All of the above is combined into a single response containing the detected problems, recommendations, reasoning, impacted metrics, confidence level, time horizon, and sources.

## Technology Stack

- Python
- Gradio
- Scikit-learn
- NumPy
- Requests
- GBIF API
- Transformers / PyTorch

## Data Sources

| Source | Role |
|---|---|
| FAO | Agricultural and soil knowledge |
| IPBES | Biodiversity and ecosystem knowledge |
| GBIF | Real biodiversity occurrence data |

## Project Goal

To combine AI, scientific knowledge retrieval, environmental variables, and biodiversity data to provide explainable, evidence-backed environmental recommendations — not generic AI-generated advice.
