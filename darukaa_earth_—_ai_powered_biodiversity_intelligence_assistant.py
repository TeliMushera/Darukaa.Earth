!pip -q install -U gradio requests scikit-learn transformers sentencepiece

import requests
import numpy as np
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
import torch

print("Libraries loaded.")

device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model="google/flan-t5-small",
    device=device
)

print("AI model loaded.")

knowledge = [
    {
        "title": "Soil Organic Carbon",
        "source": "FAO",
        "text": """
        Soil organic carbon improves soil structure, nutrient cycling, water retention,
        microbial activity and long-term soil health. Increasing organic matter through
        crop residues, compost, cover crops and diversified vegetation can improve soil
        quality and resilience.
        """
    },
    {
        "title": "Crop Diversification",
        "source": "FAO",
        "text": """
        Crop diversification through crop rotation, intercropping and mixed cropping
        can improve soil health, reduce dependence on a single crop and increase
        agricultural biodiversity. Diverse vegetation can provide habitat and food
        resources for beneficial organisms.
        """
    },
    {
        "title": "Agroforestry",
        "source": "FAO",
        "text": """
        Agroforestry combines trees with crops or livestock. It can improve soil
        organic matter, reduce erosion, improve microclimate conditions, support
        biodiversity and provide additional ecological services.
        """
    },
    {
        "title": "Water and Soil Management",
        "source": "FAO",
        "text": """
        Mulching, maintaining vegetation cover, reducing bare soil and improving
        water management can reduce evaporation and erosion and help maintain
        soil moisture, particularly under water-limited conditions.
        """
    },
    {
        "title": "Biodiversity and Ecosystem Services",
        "source": "IPBES",
        "text": """
        Biodiversity supports ecosystem functions and services including pollination,
        nutrient cycling, soil formation and ecosystem resilience. Habitat degradation
        and loss of habitat diversity can reduce biodiversity and ecosystem functions.
        """
    },
    {
        "title": "Pollinator Habitat",
        "source": "FAO",
        "text": """
        Flowering plants, native vegetation and habitat diversity can support
        pollinators and other beneficial organisms. Maintaining diverse vegetation
        around agricultural land can improve ecological habitat.
        """
    },
    {
        "title": "Habitat Fragmentation",
        "source": "IPBES",
        "text": """
        Conversion of natural habitat into simplified land-use systems can reduce
        habitat availability and connectivity. Maintaining vegetation patches,
        ecological corridors and diverse field margins can support biodiversity.
        """
    },
    {
        "title": "Soil and Water Conservation",
        "source": "FAO",
        "text": """
        Vegetative cover, residue retention, reduced soil disturbance and appropriate
        water management can reduce erosion, conserve soil moisture and improve
        long-term land productivity.
        """
    },
    {
        "title": "Sustainable Agriculture",
        "source": "FAO",
        "text": """
        Sustainable agricultural systems combine productivity with conservation of
        soil, water and biodiversity. Integrated practices can improve resilience
        to climate variability and environmental stress.
        """
    },
    {
        "title": "Native Vegetation",
        "source": "IPBES",
        "text": """
        Maintaining or restoring native vegetation can provide habitat for species,
        improve ecological connectivity and support ecosystem functions. Native
        vegetation can also contribute to soil and water conservation.
        """
    }
]

documents = [item["text"] for item in knowledge]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(documents)

print(f"Knowledge documents indexed: {len(knowledge)}")

def retrieve_knowledge(query, top_k=4):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "title": knowledge[index]["title"],
            "source": knowledge[index]["source"],
            "score": float(scores[index]),
            "text": knowledge[index]["text"].strip()
        })

    return results

def get_gbif_data(country_code):
    try:
        url = "https://api.gbif.org/v1/occurrence/search"

        params = {
            "country": country_code.upper(),
            "occurrenceStatus": "PRESENT",
            "hasCoordinate": "true",
            "limit": 300
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        records = data.get("results", [])

        species = set()

        for record in records:
            scientific_name = record.get("scientificName")

            if scientific_name:
                species.add(scientific_name)

        return {
            "total_records": data.get("count", 0),
            "sample_records": len(records),
            "unique_species_in_sample": len(species),
            "sample_species": sorted(list(species))[:20]
        }

    except Exception as e:
        return {
            "total_records": 0,
            "sample_records": 0,
            "unique_species_in_sample": 0,
            "sample_species": [],
            "error": str(e)
        }

def calculate_reasoning(
    soil_carbon,
    rainfall,
    temperature,
    soil_moisture,
    land_use
):
    land = land_use.lower()
    moisture = soil_moisture.lower()

    problems = []
    recommendations = []
    metrics = []
    interactions = []

    # -----------------------------
    # ENVIRONMENTAL CONDITION ANALYSIS
    # -----------------------------

    low_carbon = soil_carbon < 0.5
    very_low_carbon = soil_carbon < 0.3

    low_rainfall = rainfall < 800
    very_low_rainfall = rainfall < 500

    low_moisture = moisture == "low"
    high_moisture = moisture == "high"

    monoculture = (
        "monoculture" in land
        or "single crop" in land
    )

    # -----------------------------
    # SOIL CARBON
    # -----------------------------

    if very_low_carbon:
        problems.append(
            "Very low soil organic carbon"
        )

        recommendations.append(
            "Increase organic matter using crop residues, compost or cover crops."
        )

        metrics.extend([
            "Soil organic carbon",
            "Soil structure",
            "Nutrient cycling"
        ])

    elif low_carbon:
        problems.append(
            "Low soil organic carbon"
        )

        recommendations.append(
            "Retain crop residues and introduce cover crops to gradually increase soil organic matter."
        )

        metrics.extend([
            "Soil organic carbon",
            "Soil health"
        ])

    # -----------------------------
    # WATER CONDITIONS
    # -----------------------------

    if low_rainfall and low_moisture:
        problems.append(
            "Low rainfall combined with low soil moisture"
        )

        recommendations.append(
            "Use mulch, ground cover and water-conservation practices to reduce moisture loss."
        )

        metrics.extend([
            "Soil moisture",
            "Water availability"
        ])

        interactions.append(
            "Rainfall → soil moisture → vegetation survival"
        )

    elif low_rainfall and high_moisture:
        problems.append(
            "Low rainfall but currently high soil moisture"
        )

        interactions.append(
            "Rainfall → soil moisture"
        )

    elif low_moisture:
        problems.append(
            "Low soil moisture"
        )

        recommendations.append(
            "Maintain vegetation cover and organic residues to help conserve soil moisture."
        )

        metrics.append("Soil moisture")

        interactions.append(
            "Soil organic matter → water retention → soil moisture"
        )

    # -----------------------------
    # LAND USE / BIODIVERSITY
    # -----------------------------

    if monoculture:
        problems.append(
            "Low land-use diversity due to monoculture"
        )

        recommendations.append(
            "Introduce crop rotation, intercropping and diverse flowering or native vegetation."
        )

        metrics.extend([
            "Habitat diversity",
            "Agricultural biodiversity"
        ])

        interactions.append(
            "Monoculture → reduced habitat diversity → biodiversity pressure"
        )

    # -----------------------------
    # TEMPERATURE
    # -----------------------------

    if temperature >= 30:
        problems.append(
            "Higher temperature may increase environmental stress"
        )

        recommendations.append(
            "Use locally appropriate heat-tolerant vegetation and maintain vegetation cover."
        )

        metrics.append("Temperature stress")

    # -----------------------------
    # MULTI-METRIC INTERACTIONS
    # -----------------------------

    if low_carbon and low_rainfall and monoculture:
        interactions.append(
            "Soil carbon → water retention → crop resilience → biodiversity"
        )

    if low_carbon and monoculture:
        interactions.append(
            "Soil health → vegetation quality → habitat diversity → biodiversity"
        )

    if low_rainfall and monoculture:
        interactions.append(
            "Rainfall → vegetation stress → habitat availability → biodiversity"
        )

    # -----------------------------
    # FALLBACK
    # -----------------------------

    if not recommendations:
        recommendations.append(
            "Maintain soil organic matter, vegetation diversity and regular biodiversity monitoring."
        )

        metrics.extend([
            "Soil health",
            "Vegetation diversity",
            "Biodiversity"
        ])

        interactions.append(
            "Soil health ↔ vegetation diversity ↔ biodiversity"
        )

    recommendations = list(dict.fromkeys(recommendations))
    metrics = list(dict.fromkeys(metrics))
    interactions = list(dict.fromkeys(interactions))
    problems = list(dict.fromkeys(problems))

    # -----------------------------
    # DYNAMIC CONFIDENCE
    # -----------------------------

    evidence_strength = 0

    if low_carbon:
        evidence_strength += 1

    if low_rainfall:
        evidence_strength += 1

    if low_moisture:
        evidence_strength += 1

    if monoculture:
        evidence_strength += 1

    if len(interactions) >= 2:
        evidence_strength += 1

    if evidence_strength >= 4:
        confidence = "High"
    elif evidence_strength >= 2:
        confidence = "Moderate"
    else:
        confidence = "Limited"

    return {
        "problems": problems,
        "recommendations": recommendations,
        "metrics": metrics,
        "interactions": interactions,
        "confidence": confidence
    }

def generate_ai_explanation(
    question,
    environmental_state,
    reasoning,
    evidence
):
    soil_carbon = environmental_state["soil organic carbon (%)"]
    rainfall = environmental_state["annual rainfall (mm)"]
    temperature = environmental_state["temperature (°C)"]
    moisture = environmental_state["soil moisture"]
    land_use = environmental_state["land use"]

    explanations = []

    # Soil carbon
    if soil_carbon < 0.3:
        explanations.append(
            f"The soil organic carbon is very low at {soil_carbon}%, "
            "so increasing organic matter can help improve soil structure, "
            "nutrient cycling and water retention."
        )

    elif soil_carbon < 0.5:
        explanations.append(
            f"The soil organic carbon is relatively low at {soil_carbon}%, "
            "so adding organic residues and cover crops can help improve soil health."
        )

    # Water
    if rainfall < 800 and moisture.lower() == "low":
        explanations.append(
            f"Annual rainfall is only {rainfall} mm and soil moisture is low, "
            "which indicates water stress; mulch, ground cover and residue retention "
            "can help reduce moisture loss."
        )

    elif rainfall < 800 and moisture.lower() == "high":
        explanations.append(
            f"Although rainfall is relatively low at {rainfall} mm, "
            "soil moisture is currently high, so immediate water stress is not "
            "the main concern."
        )

    elif moisture.lower() == "low":
        explanations.append(
            "Low soil moisture indicates water stress, making vegetation cover "
            "and moisture-conservation practices important."
        )

    # Land use
    if "monoculture" in land_use.lower() or "single crop" in land_use.lower():
        explanations.append(
            f"The current land use ({land_use}) has low crop diversity, "
            "which can provide fewer habitat types and ecological resources. "
            "Crop rotation, intercropping and diverse vegetation can improve habitat diversity."
        )

    # Temperature
    if temperature >= 30:
        explanations.append(
            f"The temperature is relatively high at {temperature}°C, "
            "so maintaining vegetation cover and using locally appropriate "
            "heat-tolerant vegetation can improve resilience."
        )

    # If nothing specific was detected
    if not explanations:
        explanations.append(
            "The current environmental conditions do not show a strong individual stressor. "
            "Maintaining soil quality, vegetation diversity and regular biodiversity monitoring "
            "can support long-term ecosystem health."
        )

    # Combine into a clean explanation
    explanation = " ".join(explanations)

    return explanation

def analyze_environment(
    question,
    soil_carbon,
    rainfall,
    temperature,
    soil_moisture,
    land_use,
    country_code,
    history
):
    try:
        soil_carbon = float(soil_carbon)
        rainfall = float(rainfall)
        temperature = float(temperature)

    except:
        return (
            history or [],
            "Please enter valid numeric values for soil carbon, rainfall and temperature."
        )

    # --------------------------------
    # CHECK QUESTION
    # --------------------------------

    if not question or not question.strip():
        question = "What should I change to improve soil health and biodiversity?"

    # --------------------------------
    # ENVIRONMENTAL STATE
    # --------------------------------

    environmental_state = {
        "soil organic carbon (%)": soil_carbon,
        "annual rainfall (mm)": rainfall,
        "temperature (°C)": temperature,
        "soil moisture": soil_moisture,
        "land use": land_use,
        "country": country_code.upper()
    }

    # --------------------------------
    # RETRIEVAL QUERY
    # --------------------------------

    query = f"""
    Soil organic carbon {soil_carbon}
    rainfall {rainfall}
    temperature {temperature}
    soil moisture {soil_moisture}
    land use {land_use}
    biodiversity habitat soil water
    {question}
    """

    evidence = retrieve_knowledge(
        query,
        top_k=4
    )

    # --------------------------------
    # GBIF
    # --------------------------------

    gbif = get_gbif_data(
        country_code
    )

    # --------------------------------
    # MULTI-METRIC REASONING
    # --------------------------------

    reasoning = calculate_reasoning(
        soil_carbon,
        rainfall,
        temperature,
        soil_moisture,
        land_use
    )

    # --------------------------------
    # AI EXPLANATION
    # --------------------------------

    ai_explanation = generate_ai_explanation(
        question,
        environmental_state,
        reasoning,
        evidence
    )

    # --------------------------------
    # FORMAT SECTIONS
    # --------------------------------

    problem_text = "\n".join(
        f"- {item}"
        for item in reasoning["problems"]
    )

    recommendation_text = "\n".join(
        f"- {item}"
        for item in reasoning["recommendations"]
    )

    metric_text = "\n".join(
        f"- {item}"
        for item in reasoning["metrics"]
    )

    interaction_text = "\n".join(
        f"- {item}"
        for item in reasoning["interactions"]
    )

    evidence_text = "\n".join(
        f"- **{item['title']}** — {item['source']} "
        f"(retrieval score: {item['score']:.2f})"
        for item in evidence
    )

    # --------------------------------
    # GBIF SPECIES
    # --------------------------------

    species = gbif.get(
        "sample_species",
        []
    )

    if species:
        species_text = ", ".join(
            species[:10]
        )
    else:
        species_text = "No species names returned."

    # --------------------------------
    # FINAL ANSWER
    # --------------------------------

    answer = f"""
## Environmental Assessment

**Question:** {question}

### Detected Environmental Conditions

{problem_text}

### Recommendation

{recommendation_text}

### Why This Recommendation?

{ai_explanation}

### Multi-Metric Reasoning

{interaction_text}

### Impacted Metrics

{metric_text}

### Time Horizon

- **Short term:** Monitor soil moisture and vegetation-cover changes.
- **Medium term:** Monitor soil organic carbon and habitat changes.
- **Long term:** Monitor biodiversity through repeated observations.

### Confidence

**{reasoning["confidence"]}**

### Scientific Evidence Retrieved

{evidence_text}

### GBIF Biodiversity Data

- **Country:** {country_code.upper()}
- **Total occurrence records:** {gbif.get("total_records", 0):,}
- **Records sampled:** {gbif.get("sample_records", 0)}
- **Unique species in sample:** {gbif.get("unique_species_in_sample", 0)}
- **Sample species:** {species_text}

### Environmental Inputs

- Soil Organic Carbon: {soil_carbon}%
- Annual Rainfall: {rainfall} mm
- Temperature: {temperature} °C
- Soil Moisture: {soil_moisture}
- Land Use: {land_use}
- Country: {country_code.upper()}
"""

    # --------------------------------
    # GRADIO MESSAGE FORMAT
    # --------------------------------

    if history is None:
        history = []

    history = list(history)

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return history, ""

import gradio as gr

css = """
body {
    background: #0b0f0d;
}

.gradio-container {
    background: #0b0f0d !important;
    color: #e5e7eb !important;
}

h1, h2, h3 {
    color: #f5f5f5 !important;
}

textarea, input {
    background: #111714 !important;
    color: #f5f5f5 !important;
    border: 1px solid #303a34 !important;
}

button {
    border-radius: 8px !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(title="Darukaa.Earth") as demo:

    gr.Markdown(
        """
        # 🌱 Darukaa.Earth
        ### AI Biodiversity Intelligence Assistant
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("### Environmental Conditions")

            soil_carbon = gr.Number(
                label="Soil Organic Carbon (%)",
                value=0.3
            )

            rainfall = gr.Number(
                label="Annual Rainfall (mm)",
                value=600
            )

            temperature = gr.Number(
                label="Temperature (°C)",
                value=28
            )

            soil_moisture = gr.Dropdown(
                choices=["Low", "Medium", "High"],
                value="Low",
                label="Soil Moisture"
            )

            land_use = gr.Textbox(
                label="Land Use / Land Cover",
                value="Wheat monoculture"
            )

            country_code = gr.Textbox(
                label="Country Code",
                value="IN"
            )

        with gr.Column(scale=2):

            chatbot = gr.Chatbot(
                label="Environmental Intelligence",
                height=600
            )

            question = gr.Textbox(
                label="Ask a question",
                placeholder="Example: Biodiversity is declining on my land. What should I change?",
                lines=3
            )

            analyze_button = gr.Button(
                "Analyze Environment",
                variant="primary"
            )

            clear_button = gr.Button(
                "Clear Conversation"
            )

    analyze_button.click(
        fn=analyze_environment,
        inputs=[
            question,
            soil_carbon,
            rainfall,
            temperature,
            soil_moisture,
            land_use,
            country_code,
            chatbot
        ],
        outputs=[
            chatbot,
            question
        ]
    )

    question.submit(
        fn=analyze_environment,
        inputs=[
            question,
            soil_carbon,
            rainfall,
            temperature,
            soil_moisture,
            land_use,
            country_code,
            chatbot
        ],
        outputs=[
            chatbot,
            question
        ]
    )

    clear_button.click(
        fn=lambda: [],
        inputs=[],
        outputs=chatbot
    )

print("Darukaa.Earth UI created successfully.")

demo.launch(
    share=True,
    debug=True
)
