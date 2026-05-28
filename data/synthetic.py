"""
Synthetic cross-context retrieval dataset for Phase 0.

Generates multi-window samples where:
  - Window `fact_window` contains a planted fact
  - Window `query_window` contains a question about that fact
  - All other windows contain filler/distractor text

The model must maintain the fact across context window boundaries
via the astrocytic state. This directly tests cross-context retrieval.
"""

import random
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# --- Fact templates: (fact_statement, question, answer_key) ---
# Each template has named slots that get filled with random entities.

FACT_TEMPLATES = [
    # Simple attribute facts
    {
        "fact": "The capital of {country} is {city}.",
        "question": "What is the capital of {country}?",
        "answer": "{city}",
        "slots": {
            "country": [
                "Valdoria", "Nexrath", "Calmeren", "Thrynos", "Belvasse",
                "Ondrimay", "Korveth", "Zelaphine", "Dystara", "Morventh",
                "Quinthelm", "Arcandia", "Sylphora", "Drenmoor", "Ithacron",
            ],
            "city": [
                "Ironhaven", "Crystalford", "Moonspire", "Goldmere", "Starwich",
                "Thorncastle", "Frostholm", "Brightwater", "Shadowdale", "Silverpeak",
                "Stormgate", "Windreach", "Deepshore", "Flamecrest", "Oakenwall",
            ],
        },
    },
    # Numeric facts
    {
        "fact": "The population of {city} is exactly {number} people.",
        "question": "What is the exact population of {city}?",
        "answer": "{number}",
        "slots": {
            "city": [
                "Velmoor", "Astragon", "Pyralis", "Zendara", "Crestfall",
                "Luminos", "Drakvale", "Solvenia", "Mythral", "Tundaris",
            ],
            "number": [
                "42,847", "183,291", "7,634", "521,008", "93,156",
                "1,247,833", "68,429", "312,765", "29,841", "876,502",
            ],
        },
    },
    # Discovery/event facts
    {
        "fact": "In {year}, Professor {person} discovered {thing} in the {location} laboratory.",
        "question": "What did Professor {person} discover in {year}?",
        "answer": "{thing}",
        "slots": {
            "year": ["2019", "2021", "2023", "1987", "2005", "1999", "2011", "2017"],
            "person": [
                "Aldric Wren", "Sable Voss", "Torin Hex", "Lyra Caine", "Dex Morrow",
                "Iris Blackthorn", "Cael Frost", "Mira Solenne", "Orion Drake", "Thea Ashford",
            ],
            "thing": [
                "a new isotope of xenon", "a self-replicating polymer",
                "a superconducting crystal", "gravitational wave harmonics",
                "a protein that reverses aging", "quantum-entangled molecules",
                "a photon with negative mass", "room-temperature fusion catalyst",
                "a bio-luminescent neural compound", "magnetic monopole particles",
            ],
            "location": [
                "Northwind", "Deepcore", "Skyreach", "Blackmesa",
                "Clearwater", "Stonebridge", "Highpoint", "Ironforge",
            ],
        },
    },
    # Color/property facts
    {
        "fact": "The {object} belonging to {person} is colored {color} and weighs {weight}.",
        "question": "What color is the {object} belonging to {person}?",
        "answer": "{color}",
        "slots": {
            "object": [
                "ancient medallion", "crystal sphere", "enchanted book",
                "ceremonial blade", "astronomical compass", "sealed container",
                "mechanical bird", "stone tablet", "glass prism", "iron key",
            ],
            "person": [
                "Lord Vanten", "Captain Elara", "Dr. Nomis", "Chief Ashoka",
                "Elder Brynn", "Agent Corvus", "Scholar Thane", "Keeper Lyris",
            ],
            "color": [
                "deep crimson", "midnight blue", "emerald green", "burnished gold",
                "obsidian black", "pearl white", "amber orange", "violet purple",
            ],
            "weight": [
                "3.7 kilograms", "850 grams", "12.1 kilograms", "210 grams",
                "1.4 kilograms", "5.8 kilograms", "670 grams", "2.3 kilograms",
            ],
        },
    },
    # Code/identifier facts
    {
        "fact": "The access code for the {facility} system is {code}.",
        "question": "What is the access code for the {facility} system?",
        "answer": "{code}",
        "slots": {
            "facility": [
                "Arcturus", "Blacksite Delta", "Cerberus", "Dawnstar",
                "Eclipse", "Firewall", "Gemini", "Helios",
                "Icarus", "Javelin", "Keystone", "Lighthouse",
            ],
            "code": [
                "ALPHA-7749", "BRAVO-3821", "SIGMA-9064", "OMEGA-1157",
                "DELTA-4493", "ECHO-6628", "FOXTROT-2205", "GAMMA-8831",
                "KILO-5517", "LIMA-3349", "NOVEMBER-7782", "TANGO-9901",
            ],
        },
    },
]


# --- Filler text corpus ---
# Topically diverse paragraphs that serve as distractors between fact and query.

FILLER_PARAGRAPHS = [
    "The development of sustainable energy sources has become one of the most pressing challenges of the modern era. Solar panels and wind turbines continue to improve in efficiency, while battery storage technology advances rapidly. Researchers are exploring novel materials for energy harvesting, including perovskite solar cells and thermoelectric generators that convert waste heat into electricity.",

    "Marine biology has revealed extraordinary adaptations in deep-sea organisms. Creatures living near hydrothermal vents thrive in temperatures exceeding 400 degrees Celsius and pressures that would crush most life forms. Bioluminescence, used for communication and predation, illuminates the perpetual darkness of the abyssal zone in spectacular displays of evolved chemistry.",

    "The history of cartography reflects humanity's evolving understanding of geography. From Ptolemy's early world maps to satellite imagery, each era brought new techniques and revelations. Medieval portolan charts guided Mediterranean sailors with surprising accuracy, while the Mercator projection, introduced in 1569, revolutionized navigation despite its well-known distortion of landmass sizes near the poles.",

    "Advances in material science have produced substances with remarkable properties. Aerogel, sometimes called frozen smoke, is the lightest solid material known, consisting of up to 99.8 percent air. Graphene, a single layer of carbon atoms, is stronger than steel yet incredibly flexible, and its electrical conductivity makes it a promising candidate for next-generation electronics.",

    "The study of ancient languages provides windows into lost civilizations. Linear A, used by the Minoans on Crete, remains undeciphered despite decades of scholarly effort. In contrast, the decipherment of Egyptian hieroglyphs through the Rosetta Stone unlocked thousands of years of recorded history, transforming our understanding of one of the world's oldest civilizations.",

    "Astronomical observations have recently challenged established theories about galaxy formation. The discovery of massive galaxies existing just 300 million years after the Big Bang suggests that early universe conditions were more conducive to rapid structure formation than previously modeled. These findings have prompted revisions to cosmological simulations and dark matter theories.",

    "The field of mycology continues to uncover the critical role of fungi in ecosystems. Mycorrhizal networks, sometimes called the wood wide web, facilitate nutrient exchange between trees across entire forests. Recent studies have shown that these fungal networks can transmit chemical warning signals between plants, effectively creating a biological communication system beneath the soil surface.",

    "Architecture in earthquake-prone regions requires innovative structural engineering. Base isolation systems, which decouple a building from ground motion using flexible bearings, have proven highly effective in Japan and New Zealand. Newer approaches include shape-memory alloy reinforcements that can return to their original form after seismic deformation, potentially creating self-healing structures.",

    "The psychology of decision-making reveals systematic biases that affect human judgment. Prospect theory, developed by Kahneman and Tversky, demonstrated that people evaluate losses and gains asymmetrically, weighing potential losses roughly twice as heavily as equivalent gains. This loss aversion shapes behavior in domains from financial investment to medical treatment choices.",

    "Polar expeditions in the early twentieth century pushed human endurance to its limits. Shackleton's Endurance expedition of 1914 became a legendary survival story when pack ice crushed their ship, stranding the crew on Antarctic ice floes for months. Their eventual rescue, achieved without a single loss of life, remains one of history's most remarkable feats of leadership.",

    "Computational fluid dynamics has transformed aerodynamic design across industries. Modern simulations can model turbulent airflow around complex geometries with millions of computational cells, reducing the need for expensive wind tunnel testing. Formula One teams now generate thousands of design iterations digitally before committing to physical prototypes.",

    "The evolution of writing systems traces a path from pictographic representation to abstract alphabets. Sumerian cuneiform, one of the earliest known writing systems, began as simple pictographs pressed into wet clay tablets around 3400 BCE. Over centuries, these images were simplified into the wedge-shaped marks that give cuneiform its name, eventually representing syllables rather than whole words.",

    "Quantum computing promises to revolutionize specific computational domains while leaving others largely unaffected. Shor's algorithm threatens current cryptographic systems by efficiently factoring large numbers, while Grover's algorithm offers quadratic speedups for unstructured search problems. However, for many everyday computing tasks, classical processors will remain more practical and efficient.",

    "Volcanic activity shapes landscapes on timescales both geological and human. The eruption of Mount Tambora in 1815 ejected so much sulfur dioxide into the stratosphere that global temperatures dropped by several degrees, causing the year without a summer in 1816. Crop failures across Europe and North America led to widespread famine and migration.",

    "The domestication of plants fundamentally altered human civilization. The Fertile Crescent's wild grasses were gradually selected for larger seeds and easier harvesting over thousands of years, eventually producing recognizable wheat and barley varieties. This agricultural revolution enabled permanent settlements, population growth, and the development of complex societies.",

    "Neuroscience has identified distinct memory systems operating through different brain structures. The hippocampus is essential for forming new episodic memories, while the cerebellum handles procedural learning such as motor skills. Working memory, maintained in prefrontal cortex circuits, provides the temporary storage and manipulation of information needed for complex cognitive tasks.",

    "The chemistry of cooking involves Maillard reactions, caramelization, and protein denaturation. When amino acids and reducing sugars react at temperatures above 140 degrees Celsius, hundreds of different flavor compounds are produced, giving browned foods their characteristic taste and aroma. Understanding these reactions allows chefs to precisely control flavor development.",

    "Tidal energy represents an underutilized renewable resource with predictable output. Unlike solar and wind power, tidal patterns follow lunar cycles that can be calculated centuries in advance. The Rance Tidal Power Station in France, operational since 1966, demonstrates the long-term viability of this approach, generating 540 gigawatt-hours annually.",

    "The mathematics of chaos theory reveals sensitive dependence on initial conditions in deterministic systems. Edward Lorenz discovered this in 1961 while running weather simulations, finding that rounding a variable from six decimal places to three produced dramatically different forecasts. This butterfly effect fundamentally limits long-term prediction in complex nonlinear systems.",

    "Ethnomusicology documents the incredible diversity of musical traditions worldwide. The gamelan orchestras of Indonesia use bronze metallophones tuned to scales that differ significantly from Western equal temperament. Throat singing traditions of Tuva and Mongolia produce multiple simultaneous pitches from a single vocalist through manipulation of vocal tract resonances.",
]


@dataclass
class CrossContextSample:
    """A single cross-context retrieval sample."""
    windows: List[str]
    fact: str
    question: str
    answer: str
    fact_window: int
    query_window: int
    distance: int  # number of windows between fact and query
    template_idx: int


def _fill_template(template: dict) -> Tuple[str, str, str, dict]:
    """Fill a template with randomly chosen slot values."""
    slots = template["slots"]
    chosen = {}
    for slot_name, options in slots.items():
        chosen[slot_name] = random.choice(options)

    fact = template["fact"].format(**chosen)
    question = template["question"].format(**chosen)
    answer = template["answer"].format(**chosen)

    return fact, question, answer, chosen


def _get_filler_text(target_words: int) -> str:
    """
    Generate filler text of approximately target_words length
    by concatenating random paragraphs.
    """
    paragraphs = []
    word_count = 0
    shuffled = random.sample(FILLER_PARAGRAPHS, len(FILLER_PARAGRAPHS))

    for para in shuffled:
        paragraphs.append(para)
        word_count += len(para.split())
        if word_count >= target_words:
            break

    # If we need more, repeat with reshuffling
    while word_count < target_words:
        para = random.choice(FILLER_PARAGRAPHS)
        paragraphs.append(para)
        word_count += len(para.split())

    text = "\n\n".join(paragraphs)
    # Trim to approximate target
    words = text.split()
    return " ".join(words[:target_words])


def generate_cross_context_sample(
    n_windows: int = 5,
    words_per_window: int = 500,
    fact_window: int = 0,
    query_window: int = -1,
    template_idx: Optional[int] = None,
) -> CrossContextSample:
    """
    Generate a single multi-window retrieval sample.

    Args:
        n_windows: Total number of context windows
        words_per_window: Approximate words per window
        fact_window: Which window contains the planted fact (0-indexed)
        query_window: Which window has the question (-1 = last)
        template_idx: Which fact template to use (None = random)

    Returns:
        CrossContextSample with all windows, fact, question, answer
    """
    # Resolve negative indexing
    query_window = query_window % n_windows

    # Pick template
    if template_idx is None:
        template_idx = random.randint(0, len(FACT_TEMPLATES) - 1)
    template = FACT_TEMPLATES[template_idx]

    # Fill the template
    fact, question, answer, _ = _fill_template(template)

    # Build windows
    windows = []
    for i in range(n_windows):
        if i == fact_window:
            # Plant the fact somewhere in this window's text
            filler_before = _get_filler_text(words_per_window // 3)
            filler_after = _get_filler_text(words_per_window // 3)
            text = f"{filler_before}\n\n{fact}\n\n{filler_after}"
        elif i == query_window:
            # Place the question at the end of this window
            # Use less filler to ensure question+answer fit within token limits
            filler = _get_filler_text(min(words_per_window - 30, 150))
            text = f"{filler}\n\nQuestion: {question}\nAnswer: {answer}"
        else:
            text = _get_filler_text(words_per_window)
        windows.append(text)

    return CrossContextSample(
        windows=windows,
        fact=fact,
        question=question,
        answer=answer,
        fact_window=fact_window,
        query_window=query_window,
        distance=query_window - fact_window,
        template_idx=template_idx,
    )


def generate_dataset(
    n_samples: int = 1000,
    n_windows: int = 5,
    words_per_window: int = 500,
    vary_distance: bool = True,
    max_distance: Optional[int] = None,
    seed: int = 42,
) -> List[CrossContextSample]:
    """
    Generate a full dataset of cross-context retrieval samples.

    Args:
        n_samples: Number of samples to generate
        n_windows: Total context windows per sample
        words_per_window: Approximate words per window
        vary_distance: If True, vary the fact-query distance across samples
        max_distance: Maximum distance (defaults to n_windows - 1)
        seed: Random seed for reproducibility

    Returns:
        List of CrossContextSample
    """
    random.seed(seed)
    max_dist = max_distance or (n_windows - 1)

    samples = []
    for i in range(n_samples):
        if vary_distance:
            # Distribute distances evenly across samples
            # Query is always the LAST window; fact_window varies to create distance
            distance = (i % max_dist) + 1
            effective_windows = n_windows
            query_window = effective_windows - 1
            fact_window = query_window - distance
            if fact_window < 0:
                # Need more windows to accommodate this distance
                effective_windows = distance + 1
                fact_window = 0
                query_window = effective_windows - 1
        else:
            fact_window = 0
            query_window = n_windows - 1
            effective_windows = n_windows

        sample = generate_cross_context_sample(
            n_windows=effective_windows,
            words_per_window=words_per_window,
            fact_window=fact_window,
            query_window=query_window,
        )
        samples.append(sample)

    return samples


def save_dataset(samples: List[CrossContextSample], path: str) -> None:
    """Save dataset to JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = []
    for s in samples:
        data.append({
            'windows': s.windows,
            'fact': s.fact,
            'question': s.question,
            'answer': s.answer,
            'fact_window': s.fact_window,
            'query_window': s.query_window,
            'distance': s.distance,
            'template_idx': s.template_idx,
        })
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> List[CrossContextSample]:
    """Load dataset from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [CrossContextSample(**d) for d in data]


if __name__ == "__main__":
    # Quick test: generate a small dataset and print stats
    samples = generate_dataset(n_samples=20, n_windows=5, words_per_window=200)
    print(f"Generated {len(samples)} samples")
    print(f"Distance distribution: {[s.distance for s in samples]}")
    print(f"\n--- Example sample (distance={samples[0].distance}) ---")
    print(f"Fact: {samples[0].fact}")
    print(f"Question: {samples[0].question}")
    print(f"Answer: {samples[0].answer}")
    print(f"Fact in window {samples[0].fact_window}, query in window {samples[0].query_window}")
    print(f"Window lengths (words): {[len(w.split()) for w in samples[0].windows]}")
