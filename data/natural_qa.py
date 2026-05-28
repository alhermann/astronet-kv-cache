"""
Natural-language cross-context evaluation for AstroNet.

Two evaluation scenarios that move beyond template-based synthetic facts:

  Scenario A  – Multi-Session Dialogue
    Simulates a user chatting with an assistant across multiple sessions.
    Personal facts ("I just adopted a golden retriever named Max") are
    introduced in early sessions; questions about those facts appear in
    later sessions.  Distractor sessions mention similar entities (other
    pets, other names) so the model cannot succeed by shallow heuristics.

  Scenario B  – Chunked Document QA
    Generates long articles (science, history, biography, …) split into
    ~200-word chunks.  Questions about facts that appeared in an earlier
    chunk must be answered from memory alone.  Other chunks contain
    topically related distractors (similar names, dates, quantities).

Both generators return List[CrossContextSample] compatible with the
existing training and evaluation loops.

Design principles
-----------------
* No internet / external datasets required: all text is procedurally
  generated from curated templates that produce *natural-sounding* prose
  with controlled ground truth.
* Diverse sentence structures (facts are embedded inside flowing
  paragraphs, not isolated "The X of Y is Z" statements).
* Paraphrased questions (multiple phrasings per fact type, chosen at
  random).
* Distractor facts in filler windows (same entity category, different
  values).
* Multi-token answers.
"""

from __future__ import annotations

import random
import json
import os
import copy
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Re-use the canonical sample type.
from data.synthetic import CrossContextSample


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO A  –  Multi-Session Dialogue
# ═══════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Fact pool: each entry defines one *personal fact* that a user might
#  mention in conversation.  Every entry carries:
#    category   – for distractor selection (pick distractors from same cat.)
#    fact_templates  – list of ways the user might *state* the fact
#    question_templates – list of ways someone might *ask* about it
#    answer     – canonical answer (multi-token friendly)
#    slots      – dict of slot-name -> list-of-possible-values
#    distractor_slots – slot overrides for generating distractors
# ---------------------------------------------------------------------------

DIALOGUE_FACT_POOL: List[Dict[str, Any]] = [
    # ── pets ──────────────────────────────────────────────────────────
    {
        "category": "pet",
        "fact_templates": [
            "Oh, exciting news — I just adopted a {breed} and named {pronoun} {pet_name}. {pronoun_cap} is about {age} old and already rules the house.",
            "So I finally got a pet! A {breed} called {pet_name}. {pronoun_cap}'s {age} old and absolutely full of energy.",
            "Big life update: {pet_name}, my new {breed}, came home last week. At {age}, {pronoun}'s already chewed through two pairs of shoes.",
            "I wanted to tell you — we brought home a {breed} named {pet_name}. {pronoun_cap}'s only {age} but {pronoun}'s settled right in.",
        ],
        "question_templates": [
            "What's the name of my dog?",
            "What did I name my pet?",
            "Can you remind me of my pet's name?",
            "I mentioned getting a new pet a while ago — what was the name?",
            "What's my dog called again?",
            "Do you remember the name I gave my new pet?",
            "I brought home a new pet recently — what did I call it?",
            "What was the name of the dog I told you about?",
            "Hey, what's my pet's name? I can't believe I'm blanking.",
        ],
        "answer": "{pet_name}",
        "slots": {
            "breed": [
                "golden retriever", "border collie", "German shepherd",
                "Labrador", "beagle", "Australian shepherd", "husky",
                "poodle", "Bernese mountain dog", "Shiba Inu",
            ],
            "pet_name": [
                "Max", "Luna", "Charlie", "Bella", "Cooper",
                "Daisy", "Milo", "Sadie", "Rocky", "Rosie",
                "Tucker", "Penny", "Zeus", "Willow", "Bear",
            ],
            "pronoun": ["he", "she"],
            "pronoun_cap": ["He", "She"],
            "age": [
                "eight weeks", "three months", "six months",
                "ten weeks", "four months", "five months",
            ],
        },
        "distractor_slots": {
            "pet_name": [
                "Buddy", "Coco", "Duke", "Ruby", "Scout",
                "Maple", "Oscar", "Hazel", "Finn", "Olive",
                "Thor", "Ivy", "Rex", "Nala", "Gus",
            ],
            "breed": [
                "corgi", "Dalmatian", "French bulldog",
                "Rottweiler", "Great Dane", "boxer",
                "whippet", "Akita", "Samoyed", "vizsla",
            ],
        },
    },
    # ── address / moving ──────────────────────────────────────────────
    {
        "category": "address",
        "fact_templates": [
            "We finally closed on the new place — we're at {address} now, in the {neighborhood} neighborhood. It's a {home_type} with a nice backyard.",
            "Just moved! Our new address is {address}, right in {neighborhood}. We got a lovely {home_type} that needed barely any work.",
            "After months of searching, we settled on a {home_type} at {address} in {neighborhood}. The move was exhausting but totally worth it.",
            "I've been meaning to share — we're living at {address} now. It's a charming {home_type} in {neighborhood}, and we love it so far.",
        ],
        "question_templates": [
            "What's my new address?",
            "Where did I move to recently?",
            "Can you remind me of the address I mentioned?",
            "I told you about my new place — what was the address?",
            "What street do I live on now?",
        ],
        "answer": "{address}",
        "slots": {
            "address": [
                "742 Evergreen Terrace", "31 Maple Court", "158 Birchwood Lane",
                "2204 Oakridge Drive", "89 Harbour Street", "17 Pinecrest Avenue",
                "463 Cedar Boulevard", "1105 Willowbrook Road", "56 Stonegate Way",
                "320 Elm Park Circle",
            ],
            "neighborhood": [
                "Westlake", "Riverside", "Kensington", "Maplewood",
                "Silverton", "Brookfield", "Eastgate", "Lakeshore",
                "Hillcrest", "Fairview",
            ],
            "home_type": [
                "three-bedroom bungalow", "renovated Victorian",
                "two-story colonial", "cozy craftsman",
                "modern townhouse", "split-level ranch",
                "mid-century modern", "Cape Cod cottage",
            ],
        },
        "distractor_slots": {
            "address": [
                "910 Rosewood Place", "47 Summit Drive", "283 Lakeside Court",
                "1502 Chestnut Lane", "66 Foxglove Avenue", "774 Bayview Road",
                "205 Aspen Circle", "391 Clover Street", "128 Thornhill Way",
                "843 Magnolia Terrace",
            ],
        },
    },
    # ── job / career ──────────────────────────────────────────────────
    {
        "category": "job",
        "fact_templates": [
            "Great news — I accepted an offer as a {job_title} at {company}! I start on {start_date} and I'll be working in their {department} division.",
            "Career update: I'm going to be the new {job_title} at {company}. My first day is {start_date}. I'll be joining the {department} team.",
            "I just signed the contract — {job_title} at {company}, starting {start_date}. I'm really excited about the {department} group there.",
            "Wanted to share that I got the job! I'll be a {job_title} at {company}, with a {start_date} start date in the {department} department.",
        ],
        "question_templates": [
            "What company did I say I'm joining?",
            "Where am I starting my new job?",
            "Can you remind me which company hired me?",
            "I told you about a new job — what was the company?",
            "What's the name of my new employer?",
            "Which company offered me the position?",
            "I signed a contract recently — what was the company name?",
            "Do you recall the employer I mentioned in our earlier chat?",
            "What firm did I say I'm moving to?",
        ],
        "answer": "{company}",
        "slots": {
            "job_title": [
                "senior data engineer", "product manager",
                "research scientist", "lead designer",
                "software architect", "marketing director",
                "operations analyst", "principal engineer",
                "clinical researcher", "technical writer",
            ],
            "company": [
                "Meridian Labs", "Northstar Analytics", "Solace Therapeutics",
                "Vertex Dynamics", "Cascade Systems", "BlueShift AI",
                "Orion Biotech", "Pinnacle Robotics", "Helix Genomics",
                "Stratos Aerospace",
            ],
            "start_date": [
                "March 3rd", "April 15th", "January 8th", "September 1st",
                "June 20th", "November 12th", "July 7th", "February 24th",
            ],
            "department": [
                "research and development", "data science",
                "machine learning", "cloud infrastructure",
                "product engineering", "clinical trials",
                "advanced concepts", "strategy and operations",
            ],
        },
        "distractor_slots": {
            "company": [
                "Arclight Engineering", "Crestwood Partners", "Lumina Health",
                "Quantum Bridge", "Redwood Ventures", "Summit Digital",
                "Tidewater Group", "Vanguard Innovations", "Zenith Software",
                "Cobalt Microsystems",
            ],
        },
    },
    # ── birthday / event ──────────────────────────────────────────────
    {
        "category": "event",
        "fact_templates": [
            "Just a heads-up — my {relation}'s birthday is coming up on {date}. I'm planning a {party_type} and I've already ordered a {gift}.",
            "I need to remember this: {relation}'s birthday is {date}. I want to throw a {party_type} and I'm thinking of getting {pronoun_obj} a {gift}.",
            "Mark your calendar — {date} is my {relation}'s birthday. We're doing a {party_type} this year, and I found the perfect {gift} as a present.",
            "So my {relation} turns another year older on {date}. I'm organizing a {party_type} and I've picked out a beautiful {gift} for {pronoun_obj}.",
        ],
        "question_templates": [
            "When is my {relation}'s birthday?",
            "What date did I say my {relation}'s birthday falls on?",
            "Can you remind me of the birthday I mentioned?",
            "I told you about an upcoming birthday — what was the date?",
            "When's the birthday party I'm planning?",
        ],
        "answer": "{date}",
        "slots": {
            "relation": [
                "sister", "brother", "mom", "dad",
                "best friend", "partner", "daughter", "son",
            ],
            "date": [
                "March 14th", "July 22nd", "November 5th", "February 28th",
                "August 9th", "October 31st", "April 18th", "December 12th",
                "June 3rd", "January 27th", "May 16th", "September 23rd",
            ],
            "party_type": [
                "surprise dinner", "backyard barbecue", "rooftop party",
                "beach picnic", "escape room outing", "wine tasting evening",
                "brunch celebration", "game night party",
            ],
            "gift": [
                "handmade photo album", "vintage watch",
                "custom star map", "leather-bound journal",
                "weekend getaway package", "noise-cancelling headphones",
                "signed first-edition novel", "artisan chocolate box",
            ],
            "pronoun_obj": ["him", "her", "them"],
        },
        "distractor_slots": {
            "date": [
                "January 3rd", "March 29th", "May 7th", "July 14th",
                "September 1st", "November 19th", "April 25th", "August 30th",
                "October 11th", "December 6th", "February 14th", "June 21st",
            ],
        },
    },
    # ── travel / vacation ─────────────────────────────────────────────
    {
        "category": "travel",
        "fact_templates": [
            "I booked our vacation! We're flying to {destination} on {travel_date}. We're staying at the {hotel} for {duration} — I can't wait to see {landmark}.",
            "Travel plans are set: {destination} on {travel_date}. The {hotel} had great reviews and it's right near {landmark}. We'll be there for {duration}.",
            "Guess what — we're going to {destination}! Flights are booked for {travel_date}, {duration} at the {hotel}. First stop is definitely {landmark}.",
            "Exciting trip coming up: {destination}, departing {travel_date}. I reserved {duration} at the {hotel}. Everyone says {landmark} is a must-see.",
        ],
        "question_templates": [
            "Where am I going on vacation?",
            "What travel destination did I mention?",
            "Can you remind me where I'm traveling to?",
            "I told you about an upcoming trip — where to?",
            "What's the destination of that vacation I booked?",
            "Do you remember the place I said I was flying to?",
            "I was excited about a trip — where was it again?",
            "Which destination did I book flights to?",
            "What country am I visiting on that vacation I planned?",
        ],
        "answer": "{destination}",
        "slots": {
            "destination": [
                "Kyoto, Japan", "Reykjavik, Iceland", "Queenstown, New Zealand",
                "Dubrovnik, Croatia", "Banff, Canada", "Patagonia, Argentina",
                "Tromsø, Norway", "Cusco, Peru", "Hallstatt, Austria",
                "Hoi An, Vietnam",
            ],
            "travel_date": [
                "June 12th", "August 3rd", "December 20th", "March 5th",
                "September 15th", "April 28th", "October 8th", "January 19th",
            ],
            "hotel": [
                "Grand Sakura Inn", "Aurora Lodge", "Lakeside Retreat",
                "Old Town Residence", "Mountain View Hotel", "Patagonia Base Camp",
                "Northern Lights Cabin", "Valley Heritage Guesthouse",
                "Alpine Chalet", "River Lantern Hotel",
            ],
            "duration": [
                "ten days", "two weeks", "eight days", "twelve days",
                "nine days", "one week", "eleven days", "sixteen days",
            ],
            "landmark": [
                "Fushimi Inari Shrine", "the Northern Lights", "Milford Sound",
                "the Old City walls", "Lake Louise", "the Perito Moreno Glacier",
                "the Arctic Cathedral", "Machu Picchu",
                "the salt mines", "the Japanese Covered Bridge",
            ],
        },
        "distractor_slots": {
            "destination": [
                "Lisbon, Portugal", "Chiang Mai, Thailand",
                "Cape Town, South Africa", "Edinburgh, Scotland",
                "Marrakech, Morocco", "Bruges, Belgium",
                "Santorini, Greece", "Cartagena, Colombia",
                "Ljubljana, Slovenia", "Lofoten, Norway",
            ],
        },
    },
    # ── medical / health ──────────────────────────────────────────────
    {
        "category": "medical",
        "fact_templates": [
            "Had my annual check-up yesterday. Dr. {doctor_name} said my {metric} is {value}, which {assessment}. I have a follow-up on {followup_date}.",
            "Quick health update — Dr. {doctor_name} ran some tests and my {metric} came back at {value}. {assessment_cap}. The next appointment is {followup_date}.",
            "So I saw Dr. {doctor_name} today. Turns out my {metric} is at {value}. {assessment_cap}, according to the doctor. Follow-up scheduled for {followup_date}.",
            "Medical news: Dr. {doctor_name} checked my {metric} and it's {value} — {assessment}. I'm going back on {followup_date} for another look.",
        ],
        "question_templates": [
            "What's the name of my doctor?",
            "Which doctor did I see recently?",
            "Who's the doctor I mentioned from my check-up?",
            "Can you remind me of my doctor's name?",
            "I told you about a medical appointment — who was the doctor?",
        ],
        "answer": "Dr. {doctor_name}",
        "slots": {
            "doctor_name": [
                "Ananya Patel", "Marcus Chen", "Sofia Ramirez",
                "James Okonkwo", "Helena Vasquez", "David Tanaka",
                "Elena Petrova", "Robert Achebe", "Claire Fontaine",
                "Samuel Johansson",
            ],
            "metric": [
                "cholesterol", "blood pressure", "vitamin D level",
                "iron count", "thyroid level", "blood sugar",
                "triglycerides", "hemoglobin A1C",
            ],
            "value": [
                "slightly elevated", "within normal range",
                "a little low", "borderline high",
                "perfectly normal", "on the high side",
                "a bit below average", "right where it should be",
            ],
            "assessment": [
                "is nothing to worry about for now",
                "means I should adjust my diet a bit",
                "could use some attention but isn't urgent",
                "is actually an improvement from last year",
            ],
            "assessment_cap": [
                "It's nothing to worry about for now",
                "It means I should adjust my diet a bit",
                "It could use some attention but isn't urgent",
                "It's actually an improvement from last year",
            ],
            "followup_date": [
                "April 10th", "June 25th", "September 3rd", "January 14th",
                "March 21st", "August 7th", "November 30th", "February 18th",
            ],
        },
        "distractor_slots": {
            "doctor_name": [
                "Priya Sharma", "William Torres", "Ingrid Hoffman",
                "Thomas Nakamura", "Beatrice Okello", "Liam Durand",
                "Aisha Karim", "Felix Lindgren", "Rosa Gutierrez",
                "Dmitri Volkov",
            ],
        },
    },
    # ── book / recommendation ─────────────────────────────────────────
    {
        "category": "book",
        "fact_templates": [
            "I just finished reading \"{book_title}\" by {author}. It's a {genre} novel that completely blew me away — the part about {plot_element} was unforgettable.",
            "Book recommendation: \"{book_title}\" by {author}. It's {genre} and it's one of the best things I've read in years. The {plot_element} storyline is incredible.",
            "You have to read \"{book_title}\" — {author} wrote it and it's this amazing {genre} story. I was hooked from the moment {plot_element} came into play.",
            "Just put down \"{book_title}\" by {author}. What a {genre} masterpiece. The way {author} handles {plot_element} is genuinely brilliant.",
        ],
        "question_templates": [
            "What was the book I recommended?",
            "What's the title of the book I was raving about?",
            "Can you remind me of the book I just finished?",
            "I mentioned a great book recently — what was the title?",
            "What book did I tell you about?",
        ],
        "answer": "{book_title}",
        "slots": {
            "book_title": [
                "The Cartographer of Forgotten Stars",
                "A Silence Between Tides",
                "The Glass Meridian",
                "Beneath the Iron Canopy",
                "When the Rivers Burned",
                "The Last Apothecary of Venn",
                "Orbital Decay",
                "The Bone Garden Manuscripts",
                "Fractured Longitude",
                "A Thousand Paper Lanterns",
            ],
            "author": [
                "Elena Vargas", "Tobias Lindqvist", "Adaeze Nwosu",
                "Haruki Brennan", "Saoirse Chakrabarti", "Nikolai Petrov",
                "Carmen Delacroix", "Yusuf Arendse", "Ingrid Hallström",
                "Ronan Achebe",
            ],
            "genre": [
                "literary fiction", "speculative fiction", "historical fiction",
                "science fiction", "magical realism", "psychological thriller",
                "dystopian", "gothic", "philosophical", "noir",
            ],
            "plot_element": [
                "the unreliable narrator twist",
                "the multi-generational family saga",
                "the slow unraveling of the protagonist's memory",
                "the parallel timelines converging",
                "the exploration of collective grief",
                "the ethical dilemma at the story's core",
                "the climate catastrophe backdrop",
                "the tension between duty and desire",
            ],
        },
        "distractor_slots": {
            "book_title": [
                "The Clockmaker's Daughter",
                "Maps of the Interior",
                "The Sunlit Abyss",
                "Salt and Iron",
                "The Paper Republic",
                "After the Seventh Wave",
                "The Murmur Engine",
                "A Cathedral of Thorns",
                "The Frequency of Light",
                "Echoes Along the Silk Road",
            ],
        },
    },
    # ── recipe / cooking ──────────────────────────────────────────────
    {
        "category": "recipe",
        "fact_templates": [
            "I tried making {dish} last night using my {source}'s recipe. The secret ingredient is {ingredient} — you add it right at the end and it transforms the whole dish. Took about {time} total.",
            "Made the most incredible {dish} yesterday! It's a family recipe from my {source}. The trick is adding {ingredient} at the very end. The whole thing takes roughly {time}.",
            "You won't believe how good the {dish} turned out. My {source}'s recipe calls for {ingredient} as a finishing touch, and it's a game-changer. About {time} from start to finish.",
            "Food update: I finally nailed {dish}! My {source}'s version uses {ingredient} as the key ingredient. Start to finish, it's about {time}.",
        ],
        "question_templates": [
            "What was the secret ingredient in that dish I made?",
            "I told you about a recipe — what was the special ingredient?",
            "Can you remind me of the key ingredient I mentioned?",
            "What was the finishing ingredient in the recipe I described?",
            "I made something with a secret ingredient — what was it?",
        ],
        "answer": "{ingredient}",
        "slots": {
            "dish": [
                "mushroom risotto", "lamb tagine", "Thai green curry",
                "cioppino", "shakshuka", "duck confit",
                "pozole rojo", "chicken tikka masala",
                "bouillabaisse", "bibimbap",
            ],
            "source": [
                "grandmother", "grandfather", "aunt",
                "college roommate", "neighbor in Seville",
                "mother-in-law", "uncle", "old chef friend",
            ],
            "ingredient": [
                "a splash of aged sherry vinegar",
                "smoked paprika butter",
                "a teaspoon of fish sauce",
                "freshly grated nutmeg",
                "a drizzle of truffle oil",
                "toasted sesame paste",
                "preserved lemon zest",
                "a pinch of saffron threads",
                "brown butter with sage",
                "roasted garlic confit",
            ],
            "time": [
                "forty-five minutes", "an hour and a half",
                "just under two hours", "about ninety minutes",
                "roughly an hour", "two and a half hours",
            ],
        },
        "distractor_slots": {
            "ingredient": [
                "a dash of pomegranate molasses",
                "caramelized shallots",
                "a squeeze of yuzu juice",
                "crumbled goat cheese",
                "harissa paste",
                "miso glaze",
                "sumac and za'atar",
                "anchovy butter",
                "chipotle in adobo",
                "black garlic paste",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
#  Chitchat / filler turns for dialogue sessions.
#  Each is a (user_msg, assistant_msg) pair.
# ---------------------------------------------------------------------------

DIALOGUE_CHITCHAT: List[Tuple[str, str]] = [
    (
        "What's the weather going to be like this weekend?",
        "It looks like Saturday will be partly cloudy with a high of 18°C, and Sunday should be sunny and a bit warmer. Great weather for getting outside!"
    ),
    (
        "I've been trying to sleep better. Any advice?",
        "A few things that tend to help: keep a consistent sleep schedule, avoid screens for an hour before bed, and keep the bedroom cool — around 18°C seems optimal for most people."
    ),
    (
        "Can you explain what a black hole actually is?",
        "A black hole is a region of spacetime where gravity is so intense that nothing, not even light, can escape once it crosses the event horizon. They form when massive stars collapse at the end of their life cycle."
    ),
    (
        "What's a good movie to watch tonight?",
        "If you're in the mood for something thought-provoking, I'd suggest Arrival — it's a brilliant sci-fi film about linguistics and perception of time. For something lighter, The Grand Budapest Hotel is always a delight."
    ),
    (
        "I've been feeling stressed at work lately.",
        "I'm sorry to hear that. It might help to identify the specific sources of stress and tackle them one at a time. Even short breaks during the day, like a ten-minute walk, can significantly reduce stress levels."
    ),
    (
        "Tell me something interesting about octopuses.",
        "Octopuses have three hearts, blue blood, and each of their eight arms has a degree of autonomous control with its own cluster of neurons. They can also change color and texture in milliseconds for camouflage."
    ),
    (
        "How does sourdough bread actually work?",
        "Sourdough relies on a symbiotic culture of wild yeast and lactobacilli bacteria. The bacteria produce lactic and acetic acids, giving the bread its tangy flavor, while the yeast handles the leavening. It's essentially a tiny ecosystem."
    ),
    (
        "I'm trying to learn a new language. Tips?",
        "Consistency beats intensity — even fifteen minutes a day is better than a three-hour session once a week. Try to immerse yourself through podcasts or shows in the target language, and don't be afraid of making mistakes."
    ),
    (
        "What's the deal with quantum computing?",
        "Quantum computers use qubits that can exist in superpositions of states, allowing them to explore many solutions simultaneously. They're not faster at everything, but for certain problems like factoring large numbers or simulating molecules, they can be exponentially faster."
    ),
    (
        "I went for a run this morning and it felt amazing.",
        "That's great! Running releases endorphins and can improve mood for hours afterward. If you're getting into a routine, try varying your pace — alternating between intervals and steady runs keeps things interesting and builds fitness faster."
    ),
    (
        "What should I make for dinner?",
        "How about a simple stir-fry? You can use whatever vegetables you have on hand, toss them in a wok with some garlic and ginger, add soy sauce and a splash of sesame oil. Serve over rice and you've got a meal in twenty minutes."
    ),
    (
        "I've been thinking about learning guitar.",
        "Guitar is a wonderful instrument to pick up. Start with a few basic chords — G, C, D, and E minor will let you play hundreds of songs. Focus on clean chord transitions before worrying about speed. There are excellent free tutorials online."
    ),
    (
        "Can you recommend a podcast?",
        "If you enjoy storytelling, Radiolab does an incredible job weaving science and narrative together. For history, Hardcore History by Dan Carlin is gripping — his series on World War I is particularly remarkable."
    ),
    (
        "Why is the sky blue?",
        "It's due to Rayleigh scattering. Sunlight contains all colors, but shorter wavelengths like blue and violet scatter more when they hit gas molecules in the atmosphere. Our eyes are more sensitive to blue than violet, so the sky appears blue rather than purple."
    ),
    (
        "I just discovered a great coffee shop near my office.",
        "Nice find! A good coffee spot near work can really improve the day. Do they roast their own beans? That's usually a sign of a place that takes their coffee seriously."
    ),
    (
        "What's the most remote place on Earth?",
        "Point Nemo in the South Pacific is the farthest point from any land — about 2,688 kilometers from the nearest shore. It's so remote that the closest humans are often the astronauts on the International Space Station overhead."
    ),
]


def _fill_dialogue_fact(
    fact_entry: Dict[str, Any],
    slot_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, Dict[str, str]]:
    """
    Pick random slot values for a dialogue fact entry and return
    (fact_statement, question, answer, chosen_values).
    """
    chosen: Dict[str, str] = {}
    for slot_name, options in fact_entry["slots"].items():
        if slot_overrides and slot_name in slot_overrides:
            chosen[slot_name] = slot_overrides[slot_name]
        else:
            chosen[slot_name] = random.choice(options)

    # Ensure pronoun consistency
    if "pronoun" in chosen:
        if chosen["pronoun"] == "he":
            chosen["pronoun_cap"] = "He"
            chosen.setdefault("pronoun_obj", "him")
        else:
            chosen["pronoun_cap"] = "She"
            chosen.setdefault("pronoun_obj", "her")

    fact_text = random.choice(fact_entry["fact_templates"]).format(**chosen)
    question_text = random.choice(fact_entry["question_templates"]).format(**chosen)
    answer_text = fact_entry["answer"].format(**chosen)

    return fact_text, question_text, answer_text, chosen


def _make_dialogue_session(
    user_lines: List[str],
    assistant_lines: List[str],
) -> str:
    """
    Render a dialogue session as a natural multi-turn conversation string.
    """
    lines: List[str] = []
    for u, a in zip(user_lines, assistant_lines):
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    return "\n\n".join(lines)


def _make_distractor_dialogue(
    fact_entry: Dict[str, Any],
    chosen_values: Optional[Dict[str, str]] = None,
    n_turns: int = 3,
) -> str:
    """
    Build a distractor session: a few chitchat turns plus a mention of
    a *different* entity from the same category (e.g., a different pet
    name, a different company).  This is what makes the task hard.

    chosen_values: the slot values used for the real answer, so we
    can exclude them from distractor selection (overlapping pools).
    """
    # Generate distractor fact with values from the SAME pool minus the answer
    distractor_overrides: Dict[str, str] = {}
    for slot_name in fact_entry.get("distractor_slots", {}):
        pool = list(fact_entry["slots"].get(slot_name, []))
        # Exclude the real answer value to ensure the distractor differs
        if chosen_values and slot_name in chosen_values:
            pool = [v for v in pool if v != chosen_values[slot_name]]
        if pool:
            distractor_overrides[slot_name] = random.choice(pool)

    dist_text, _, _, _ = _fill_dialogue_fact(fact_entry, distractor_overrides)

    # Wrap it in a realistic session
    chitchat_pairs = random.sample(
        DIALOGUE_CHITCHAT, min(n_turns, len(DIALOGUE_CHITCHAT))
    )

    user_lines: List[str] = []
    assistant_lines: List[str] = []

    # First turn or two: chitchat
    for u, a in chitchat_pairs[:max(1, n_turns - 1)]:
        user_lines.append(u)
        assistant_lines.append(a)

    # Embed distractor fact as a casual mention
    user_lines.append(dist_text)
    assistant_lines.append(
        "That's great to hear! Thanks for sharing. Let me know if you'd like to talk more about it."
    )

    # Maybe one more chitchat turn
    if n_turns > 2 and len(chitchat_pairs) > n_turns - 1:
        u, a = chitchat_pairs[n_turns - 1]
        user_lines.append(u)
        assistant_lines.append(a)

    return _make_dialogue_session(user_lines, assistant_lines)


def _make_chitchat_session(n_turns: int = 3) -> str:
    """Build a pure chitchat session with no planted facts."""
    pairs = random.sample(
        DIALOGUE_CHITCHAT, min(n_turns, len(DIALOGUE_CHITCHAT))
    )
    user_lines = [u for u, _ in pairs]
    assistant_lines = [a for _, a in pairs]
    return _make_dialogue_session(user_lines, assistant_lines)


def generate_dialogue_sample(
    n_windows: int = 5,
    fact_window: int = 0,
    query_window: int = -1,
    fact_idx: Optional[int] = None,
) -> CrossContextSample:
    """
    Generate a single multi-session dialogue evaluation sample.

    The fact is embedded naturally in session `fact_window`.
    Distractor sessions contain similar entities from the same category.
    The question appears at the end of `query_window`.

    Args:
        n_windows:    Total number of dialogue sessions (context windows)
        fact_window:  Which session contains the target fact
        query_window: Which session asks the question (-1 = last)
        fact_idx:     Index into DIALOGUE_FACT_POOL (None = random)

    Returns:
        CrossContextSample compatible with existing eval loops.
    """
    query_window = query_window % n_windows

    # Select fact
    if fact_idx is None:
        fact_idx = random.randint(0, len(DIALOGUE_FACT_POOL) - 1)
    fact_entry = DIALOGUE_FACT_POOL[fact_idx]

    # Generate the target fact
    fact_text, question_text, answer_text, chosen = _fill_dialogue_fact(fact_entry)

    windows: List[str] = []

    for i in range(n_windows):
        if i == fact_window:
            # ── session with target fact ──
            chitchat_pairs = random.sample(
                DIALOGUE_CHITCHAT,
                min(3, len(DIALOGUE_CHITCHAT)),
            )
            user_lines: List[str] = []
            assistant_lines: List[str] = []

            # One or two chitchat turns before the fact
            n_before = random.randint(1, 2)
            for u, a in chitchat_pairs[:n_before]:
                user_lines.append(u)
                assistant_lines.append(a)

            # The fact itself
            user_lines.append(fact_text)
            assistant_lines.append(
                random.choice([
                    "That's wonderful news! I'll remember that. Is there anything else you'd like to share?",
                    "Thanks for telling me! I've made a note of that. What else is going on?",
                    "How exciting! I appreciate you sharing. Anything else on your mind?",
                    "Great to hear! I'll keep that in mind. Want to talk about anything else?",
                ])
            )

            # Maybe one more chitchat turn after
            if len(chitchat_pairs) > n_before:
                u, a = chitchat_pairs[n_before]
                user_lines.append(u)
                assistant_lines.append(a)

            windows.append(_make_dialogue_session(user_lines, assistant_lines))

        elif i == query_window:
            # ── session with the question ──
            chitchat_pairs = random.sample(
                DIALOGUE_CHITCHAT,
                min(2, len(DIALOGUE_CHITCHAT)),
            )
            user_lines_q: List[str] = []
            assistant_lines_q: List[str] = []

            # A chitchat turn before the question
            for u, a in chitchat_pairs[:1]:
                user_lines_q.append(u)
                assistant_lines_q.append(a)

            # Build session text with chitchat, then append the question
            # using the standard "Question: ...\nAnswer: ..." format so that
            # eval functions can truncate at the marker.
            session_text = _make_dialogue_session(user_lines_q, assistant_lines_q)
            session_text += (
                f"\n\nQuestion: {question_text}\n"
                f"Answer: {answer_text}"
            )
            windows.append(session_text)

        else:
            # ── filler session: ALL filler windows get distractors ──
            # This makes the task harder (multiple confounders)
            windows.append(
                _make_distractor_dialogue(
                    fact_entry,
                    chosen_values=chosen,
                    n_turns=3,
                )
            )

    return CrossContextSample(
        windows=windows,
        fact=fact_text,
        question=question_text,
        answer=answer_text,
        fact_window=fact_window,
        query_window=query_window,
        distance=query_window - fact_window,
        template_idx=fact_idx,
    )


def generate_dialogue_dataset(
    n_samples: int = 200,
    n_windows: int = 5,
    vary_distance: bool = True,
    max_distance: Optional[int] = None,
    seed: int = 42,
) -> List[CrossContextSample]:
    """
    Generate a dataset of multi-session dialogue evaluation samples.

    Args:
        n_samples:     Number of samples
        n_windows:     Default number of sessions per sample
        vary_distance: Vary fact–query distance across samples
        max_distance:  Maximum distance (default n_windows - 1)
        seed:          Random seed

    Returns:
        List of CrossContextSample
    """
    random.seed(seed)
    max_dist = max_distance or (n_windows - 1)

    samples: List[CrossContextSample] = []
    n_fact_types = len(DIALOGUE_FACT_POOL)

    # Pre-build balanced distance assignments (shuffled)
    if vary_distance:
        distances = []
        per_dist = n_samples // max_dist
        for d in range(1, max_dist + 1):
            distances.extend([d] * per_dist)
        # Fill remainder
        while len(distances) < n_samples:
            distances.append(random.randint(1, max_dist))
        random.shuffle(distances)

    for i in range(n_samples):
        if vary_distance:
            distance = distances[i]
            query_window_idx = max(distance + 1, n_windows) - 1
            effective_windows = query_window_idx + 1
            fw = query_window_idx - distance
        else:
            effective_windows = n_windows
            fw = 0
            query_window_idx = n_windows - 1

        # Randomize fact category (not deterministic cycling)
        fact_idx = random.randint(0, n_fact_types - 1)

        sample = generate_dialogue_sample(
            n_windows=effective_windows,
            fact_window=fw,
            query_window=query_window_idx,
            fact_idx=fact_idx,
        )
        samples.append(sample)

    return samples


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO B  –  Chunked Document QA
# ═══════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Article blueprints: each blueprint defines a *topic* with:
#    - section_templates:  list of paragraph templates for building a long
#      article.  Each template has {slots} that get filled.
#    - fact_templates:  specific sentences embedding a retrievable fact,
#      designed to blend naturally into the article flow.
#    - question_templates:  paraphrased ways to ask about the fact.
#    - answer: canonical answer string.
#    - slots: dict of slot-name -> options
#    - distractor_sections: paragraph templates that mention similar-but-
#      different entities to act as confounders.
# ---------------------------------------------------------------------------

ARTICLE_BLUEPRINTS: List[Dict[str, Any]] = [
    # ── History / Exploration ─────────────────────────────────────────
    {
        "topic": "history_exploration",
        "title_template": "The {adjective} Expedition to {region}",
        "section_templates": [
            "The {adjective} expedition to {region} remains one of the most remarkable chapters in the history of exploration. Organized in {era}, the journey was driven by a desire to chart unmapped territories and establish new trade routes. Historians have long debated the expedition's true motivations, with some arguing it was primarily a scientific endeavor while others emphasize the political pressures of the time.",
            "Preparations for the voyage took the better part of two years. Supply ships were loaded with provisions calculated to sustain a crew of over two hundred men for eighteen months at sea. Navigation instruments of the period were rudimentary by modern standards, relying heavily on celestial observation and dead reckoning. The expedition's cartographer, whose detailed maps survive in fragmentary form, proved instrumental in later voyages to the same waters.",
            "The crew faced harrowing conditions during the crossing. Violent storms in the southern latitudes damaged the flagship's mainmast, forcing an unplanned stop for repairs at a remote island chain. Disease, always a threat on long voyages, claimed several lives before the ship's surgeon implemented a regimen of citrus rations that slowed the spread of scurvy. Morale was maintained through strict discipline and the promise of substantial rewards upon return.",
            "Upon reaching {region}, the expedition encountered landscapes and peoples entirely unknown to their home continent. Detailed journals kept by several officers describe lush river valleys, towering mountain ranges, and sophisticated agricultural systems maintained by the indigenous population. These accounts would later fuel both scientific curiosity and colonial ambition in equal measure.",
            "The expedition's legacy is complex. While it advanced geographical knowledge considerably and brought back botanical specimens that transformed European agriculture, it also set in motion a chain of colonial encounters with devastating consequences for indigenous communities. Modern scholarship attempts to hold both truths simultaneously, recognizing the expedition's contributions to science without ignoring its role in broader patterns of exploitation.",
        ],
        "fact_templates": [
            "The expedition was commanded by Captain {person}, a veteran navigator who had previously mapped the coastline of three continents. Under {person}'s leadership, the crew maintained remarkable discipline despite the enormous hardships they faced during the {duration} journey.",
            "At the helm of the expedition was Captain {person}, already famous for circumnavigating the southern archipelago twice. The {duration} voyage would prove to be {person}'s greatest achievement, cementing a reputation that endures to this day.",
            "Captain {person} led the expedition with a combination of iron will and genuine concern for the crew's welfare. The {duration} crossing tested even {person}'s considerable experience, but meticulous planning ensured the mission's ultimate success.",
        ],
        "question_templates": [
            "Who commanded the expedition to {region}?",
            "What was the name of the captain who led the expedition?",
            "Who led the voyage to {region}?",
            "Can you tell me who was in charge of the {region} expedition?",
        ],
        "answer": "Captain {person}",
        "slots": {
            "adjective": [
                "Storied", "Legendary", "Ill-Fated", "Ambitious",
                "Remarkable", "Pioneering", "Audacious", "Historic",
            ],
            "region": [
                "the Cerulean Coast", "the Meridian Archipelago",
                "the Obsidian Straits", "the Verdant Peninsula",
                "the Coral Expanse", "the Iron Shore",
                "the Amber Isles", "the Silver Fjords",
            ],
            "era": [
                "the late fifteenth century", "the early sixteenth century",
                "the 1580s", "the spring of 1492",
                "the turbulent 1530s", "the closing decades of the 1400s",
            ],
            "person": [
                "Elias Thorne", "Margareta Solberg", "Joaquín del Río",
                "Aleksandr Voss", "Linnéa Strand", "Daisuke Morimoto",
                "Caterina Almeida", "Henrik Rasmusen",
            ],
            "duration": [
                "fourteen-month", "eleven-month", "two-year",
                "nine-month", "eighteen-month", "sixteen-month",
            ],
        },
        "distractor_templates": [
            "Around the same period, Captain {dist_person} mounted a rival expedition to map the northern coastline. Though less celebrated, this parallel journey produced important nautical charts that remained in use for over a century. {dist_person}'s fleet was smaller — just three vessels — but managed to survey more than two thousand kilometers of previously uncharted shore.",
            "Historical records also mention Commander {dist_person}, who attempted a similar voyage several years earlier but was forced to turn back due to severe weather. Despite the failure, {dist_person}'s logbooks contained invaluable meteorological observations that helped later expeditions plan their routes more effectively.",
        ],
        "distractor_slots": {
            "dist_person": [
                "Rodrigo Ferreira", "Astrid Lindahl", "Tobias Hauck",
                "Yumiko Watanabe", "Edvard Nilsen", "Isabella Conti",
                "Bernard Leclerc", "Olga Petrov",
            ],
        },
    },
    # ── Science / Biology ─────────────────────────────────────────────
    {
        "topic": "science_biology",
        "title_template": "Recent Advances in {field}: The Discovery of {organism}",
        "section_templates": [
            "The field of {field} has undergone a remarkable transformation in recent decades, driven by advances in genomic sequencing and computational biology. What was once a discipline limited to morphological classification has become a deeply molecular science, capable of resolving evolutionary relationships with unprecedented precision. New species continue to be discovered at a surprising rate, particularly in understudied ecosystems.",
            "Fieldwork in remote regions remains essential despite technological advances. Researchers spend weeks or months in challenging environments, collecting specimens and environmental DNA samples that later undergo analysis in well-equipped laboratories. The gap between field collection and publication can span years, as thorough taxonomic description requires comparison with existing type specimens housed in museums around the world.",
            "Ecological relationships in these communities are extraordinarily complex. Symbiotic interactions between species create networks of mutual dependence that can be disrupted by even small environmental perturbations. Understanding these networks is critical for conservation planning, as the loss of a single keystone species can trigger cascading effects throughout an entire ecosystem.",
            "Funding for basic taxonomic research has historically been limited, yet its importance for biodiversity conservation cannot be overstated. Without accurate species inventories, it is impossible to assess the true extent of biodiversity loss or to prioritize conservation efforts effectively. Several international initiatives now aim to accelerate the pace of species description through standardized protocols and open-access databases.",
            "The integration of citizen science into professional research has opened new avenues for discovery. Trained volunteers contribute observations and specimens from regions that professional scientists cannot regularly access, significantly expanding the geographic coverage of biodiversity surveys. Digital platforms for sharing photographs and locality data have made these contributions more valuable than ever.",
        ],
        "fact_templates": [
            "The breakthrough came in {year} when Dr. {scientist} identified {organism} in samples collected from {habitat}. The discovery was remarkable because {organism} exhibited {trait}, a feature never before documented in this taxonomic group. Dr. {scientist}'s {year} paper describing the find has already been cited over three hundred times.",
            "Dr. {scientist}'s {year} discovery of {organism} in {habitat} sent ripples through the scientific community. What made {organism} extraordinary was its {trait}, which challenged prevailing assumptions about the evolutionary limits of the group. The original description, published in a leading journal, remains one of the most-discussed papers in the field.",
            "In {year}, Dr. {scientist} reported the existence of {organism} based on specimens from {habitat}. The species immediately attracted attention due to its {trait} — a property that had been considered theoretically possible but never observed in nature. {scientist}'s meticulous documentation left little room for doubt about the finding's validity.",
        ],
        "question_templates": [
            "Who discovered {organism}?",
            "Which scientist identified {organism}?",
            "What researcher found {organism} in {habitat}?",
            "Can you name the scientist who described {organism}?",
        ],
        "answer": "Dr. {scientist}",
        "slots": {
            "field": [
                "marine taxonomy", "fungal ecology", "entomology",
                "deep-sea biology", "cave ecology", "soil microbiology",
                "freshwater invertebrate zoology", "canopy ecology",
            ],
            "year": [
                "2019", "2021", "2023", "2018", "2022", "2020", "2017", "2024",
            ],
            "scientist": [
                "Amara Osei", "Kenji Furukawa", "Clara Eisenberg",
                "Tomás Herrera", "Astrid Lund", "Pradeep Venkatesh",
                "Miriam Bakker", "Lukas Stein",
            ],
            "organism": [
                "Luminocaris abyssalis", "Mycena crystallum",
                "Troglobius magneticus", "Dendrophila aurora",
                "Bathynomus titanicus", "Rhizopogon stellaris",
                "Neocaridina phosphorea", "Xylotrechus mirabilis",
            ],
            "habitat": [
                "a hydrothermal vent field in the South Pacific",
                "ancient limestone caves in Borneo",
                "the Amazon River basin floodplain",
                "Antarctic subglacial lakes",
                "the cloud forests of Papua New Guinea",
                "deep-sea sediment cores in the Arctic Ocean",
                "volcanic hot springs in Iceland",
                "the canopy of old-growth redwood forests",
            ],
            "trait": [
                "magnetoreception capabilities",
                "bioluminescent reproductive structures",
                "a previously unknown form of photosynthesis",
                "the ability to survive complete desiccation for decades",
                "a symbiotic relationship with archaea",
                "regenerative properties rivaling those of planaria",
                "a completely novel respiratory pigment",
                "acoustic communication in the ultrasonic range",
            ],
        },
        "distractor_templates": [
            "In a parallel discovery, Dr. {dist_scientist} reported a closely related species from an entirely different ocean basin in the same year. While superficially similar, genomic analysis revealed that the two lineages diverged over forty million years ago, representing a striking case of convergent evolution. {dist_scientist}'s find was no less significant, though it received less media attention.",
            "Contemporaneous work by Dr. {dist_scientist} in a nearby region turned up several undescribed species as well, though none as dramatic as the headline discovery. {dist_scientist}'s systematic survey of the area provided essential ecological context, showing that the broader community harbored much greater diversity than previously suspected.",
        ],
        "distractor_slots": {
            "dist_scientist": [
                "Fiona MacGregor", "Ravi Anand", "Isabelle Moreau",
                "Sven Holmqvist", "Chandra Deshpande", "Yuki Tanabe",
                "Nadia El-Amin", "Jorge Salazar",
            ],
        },
    },
    # ── Technology / Engineering ───────────────────────────────────────
    {
        "topic": "technology_engineering",
        "title_template": "The Development of the {system_name} System",
        "section_templates": [
            "The {system_name} project represents one of the most ambitious engineering undertakings in recent memory. Conceived as a response to growing demands for {application}, the system combines cutting-edge hardware design with novel software architectures to achieve performance levels that were considered impossible just a decade ago. Development began in earnest after initial feasibility studies demonstrated that the core concept was sound.",
            "Engineering challenges during the design phase were formidable. The team had to solve problems in thermal management, signal integrity, and power distribution simultaneously, with tight constraints on physical dimensions and weight. Iterative prototyping revealed unexpected interactions between subsystems that required fundamental redesigns in several critical areas. Each revision brought the team closer to the performance targets, though setbacks along the way tested both patience and funding.",
            "Testing and validation occupied nearly as much time as initial development. The system was subjected to thousands of hours of stress testing under conditions designed to exceed operational specifications by wide margins. Failure modes were catalogued and addressed systematically, with redundancy built into every critical pathway. The result is a system whose reliability metrics surpass those of its predecessors by an order of magnitude.",
            "Deployment of {system_name} has already begun in several key markets, with early adopters reporting performance improvements consistent with laboratory benchmarks. The transition from legacy systems has been smoother than anticipated, partly because the development team prioritized backward compatibility in the interface design. Training programs for operators and maintenance personnel have been rolled out in parallel.",
            "Looking ahead, the modular architecture of {system_name} is expected to support incremental upgrades over a lifespan of at least fifteen years. The development team has published a technology roadmap outlining planned enhancements in processing speed, energy efficiency, and integration with emerging standards. If realized, these improvements would extend the system's capabilities well beyond its original design envelope.",
        ],
        "fact_templates": [
            "The project was led by chief engineer {engineer}, who assembled a team of {team_size} specialists drawn from twelve countries. {engineer}'s previous work on high-reliability systems proved invaluable, as the design philosophy of building in redundancy from the ground up shaped every aspect of the {system_name} architecture. The total development cost reached {cost}.",
            "Under the direction of chief engineer {engineer}, a team of {team_size} engineers and scientists spent over four years bringing {system_name} from concept to reality. The project's {cost} budget — substantial by any measure — was justified by the system's projected twenty-year operational lifespan. {engineer} was later recognized with the field's highest engineering honor for the achievement.",
            "Chief engineer {engineer} guided the {system_name} project through its most challenging phases, coordinating a team of {team_size} people and a budget of {cost}. {engineer}'s insistence on rigorous testing protocols, even when they caused schedule delays, ultimately saved the project from several potentially catastrophic design flaws that surfaced late in development.",
        ],
        "question_templates": [
            "Who was the chief engineer of the {system_name} project?",
            "Who led the development of {system_name}?",
            "What engineer was in charge of building {system_name}?",
            "Can you name the lead engineer behind the {system_name} system?",
        ],
        "answer": "{engineer}",
        "slots": {
            "system_name": [
                "Helios-9", "Meridian Array", "Quantum Lattice",
                "Polaris Core", "Tesseract Grid", "Aegis Prime",
                "Nexus Vector", "Chronos Module",
            ],
            "application": [
                "autonomous infrastructure monitoring",
                "ultra-low-latency communications",
                "distributed energy grid management",
                "precision atmospheric modeling",
                "next-generation satellite navigation",
                "large-scale environmental sensing",
            ],
            "engineer": [
                "Rina Takahashi", "Oluwaseun Adeyemi", "Kai Lindström",
                "Mariana Costa", "Dmitri Volkov", "Annika Bauer",
                "Tariq Al-Rashid", "Freya Magnúsdóttir",
            ],
            "team_size": [
                "340", "over 500", "280", "roughly 450",
                "more than 600", "approximately 370",
            ],
            "cost": [
                "1.4 billion dollars", "2.1 billion euros",
                "870 million dollars", "3.2 billion dollars",
                "1.8 billion euros", "960 million dollars",
            ],
        },
        "distractor_templates": [
            "A competing initiative led by {dist_engineer} at a rival institution pursued a fundamentally different approach to the same problem. While ultimately less successful in achieving raw performance targets, {dist_engineer}'s design introduced several innovations in energy efficiency that were later incorporated into the {system_name} project's upgrade roadmap.",
            "Industry veteran {dist_engineer} served as an external reviewer during the critical design review phase. {dist_engineer}'s published critique of the thermal management subsystem prompted a significant redesign that, while adding six months to the schedule, dramatically improved long-term reliability. The team later acknowledged this intervention as a turning point.",
        ],
        "distractor_slots": {
            "dist_engineer": [
                "Pavel Mikhailov", "Sunita Rao", "Anders Johansson",
                "Beatriz Silva", "Hiroshi Endo", "Leila Mansouri",
                "Viktor Szabo", "Christine O'Neill",
            ],
        },
    },
    # ── Biography / Arts ──────────────────────────────────────────────
    {
        "topic": "biography_arts",
        "title_template": "The Life and Work of {artist}",
        "section_templates": [
            "Few artists of the modern era have provoked as much critical debate as {artist}. Born into modest circumstances, {artist} showed an early affinity for creative expression that would eventually reshape entire genres. Biographers have traced the roots of this sensibility to a childhood spent in close contact with nature and a family that, while not wealthy, valued intellectual curiosity above material comfort.",
            "The early career was marked by a period of intense experimentation. Working in relative obscurity, {artist} produced a body of work that drew from diverse influences — folk traditions, classical forms, and the emerging avant-garde. Critics of the time were divided: some praised the originality, while others dismissed the work as derivative or needlessly provocative. Time would vindicate the former camp decisively.",
            "Recognition came gradually and then all at once. A series of exhibitions in major cultural capitals brought widespread attention, and within a few years {artist} had become one of the most sought-after figures in the international art scene. Commissions poured in from private collectors, museums, and public institutions alike. Despite the sudden fame, {artist} maintained a disciplined creative practice, producing work at a pace that astonished peers.",
            "The middle period is generally regarded as the most productive and artistically significant. The works from this era exhibit a confidence and maturity that distinguish them from both the early experiments and the more contemplative late output. Scholars have identified a recurring set of themes — memory, displacement, the tension between permanence and impermanence — that give the oeuvre its distinctive coherence.",
            "In later years, {artist} turned increasingly to mentorship and advocacy, using the platform of fame to champion emerging artists from underrepresented backgrounds. Several of these protégés have since achieved major recognition in their own right, forming what critics sometimes call a school or movement. The legacy, still evolving, extends far beyond any single body of work.",
        ],
        "fact_templates": [
            "The defining work of {artist}'s career, \"{masterwork}\", was completed in {year} after {duration} of painstaking labor. Its debut at the {venue} drew crowds that overwhelmed the institution's capacity, and the critical response was rapturous. \"{masterwork}\" is now considered a landmark — one of those rare works that permanently altered the trajectory of its medium.",
            "{artist}'s magnum opus, \"{masterwork}\", emerged from {duration} of concentrated creative effort, finally reaching completion in {year}. When it premiered at the {venue}, audiences and critics alike recognized that something extraordinary had arrived. In the decades since, \"{masterwork}\" has been exhibited worldwide and studied in countless academic programs.",
            "By far the most celebrated of {artist}'s creations is \"{masterwork}\", unveiled in {year} at the {venue} after {duration} in the studio. The work's emotional depth, technical virtuosity, and sheer ambition secured {artist}'s place among the giants of the field. It remains the single most requested loan in the {venue}'s history.",
        ],
        "question_templates": [
            "What is the title of {artist}'s most famous work?",
            "What was {artist}'s masterpiece called?",
            "Can you name {artist}'s defining work?",
            "What work is {artist} best known for?",
        ],
        "answer": "{masterwork}",
        "slots": {
            "artist": [
                "Lena Johansson", "Kwame Asante", "Yael Mizrahi",
                "Rafael Esteban", "Hana Kimura", "Odhran Byrne",
                "Adélaïde Rousseau", "Mikhail Sorokin",
            ],
            "masterwork": [
                "The Weight of Silence",
                "Convergence in Blue and Gold",
                "An Atlas of Lost Hours",
                "The Seventh Migration",
                "Roots Beneath Still Water",
                "Threshold of the Visible",
                "The Cartography of Longing",
                "Nocturne for a Vanishing World",
            ],
            "year": [
                "1998", "2003", "2011", "1987", "2016", "1994", "2008", "2020",
            ],
            "duration": [
                "three years", "nearly five years", "eighteen months",
                "over two years", "almost four years", "seven years",
            ],
            "venue": [
                "Biennale di Venezia", "Tate Modern", "MoMA",
                "Centre Pompidou", "Guggenheim Bilbao",
                "documenta in Kassel", "Serpentine Gallery",
                "National Gallery of Art",
            ],
        },
        "distractor_templates": [
            "Contemporary {dist_artist} was often compared to {artist}, though the two pursued quite different aesthetic visions. {dist_artist}'s work was generally more minimalist in approach, favoring stark compositions and monochromatic palettes. Despite the stylistic differences, they maintained a lifelong mutual respect, occasionally collaborating on joint exhibitions.",
            "In the same generation, {dist_artist} emerged as another major voice, frequently appearing alongside {artist} in group shows and critical surveys. While {dist_artist} explored related themes, the formal vocabulary was distinct — more geometric, more architectonic. The friendly rivalry between the two pushed both toward their best work.",
        ],
        "distractor_slots": {
            "dist_artist": [
                "Tomoko Shirai", "Emeka Okafor", "Ingrid Halversen",
                "Mateo Aguilar", "Suki Park", "Ciaran Doyle",
                "Noémie Laurent", "Andrei Volkov",
            ],
        },
    },
    # ── Geography / Climate ───────────────────────────────────────────
    {
        "topic": "geography_climate",
        "title_template": "Climate and Ecology of the {region_name}",
        "section_templates": [
            "The {region_name} encompasses one of the most ecologically diverse landscapes on the planet. Spanning an area of approximately {area}, it ranges from coastal lowlands through temperate forests to alpine tundra, supporting an astonishing variety of plant and animal life. Seasonal variation is dramatic, with summer temperatures occasionally exceeding thirty degrees Celsius while winter lows routinely dip below minus twenty.",
            "Hydrological systems in the {region_name} play a central role in shaping both the landscape and the communities that depend on it. The region's rivers, fed by snowmelt from mountain glaciers, provide freshwater to millions of people downstream. Recent studies have documented accelerated glacial retreat, raising concerns about long-term water security in an area already subject to periodic droughts.",
            "Human settlement in the {region_name} dates back thousands of years, with archaeological evidence of sophisticated irrigation systems and terraced agriculture. Modern communities maintain many traditional land-use practices alongside contemporary agriculture and resource extraction. Balancing economic development with environmental stewardship remains the region's central policy challenge.",
            "Biodiversity surveys conducted over the past two decades have revealed that the {region_name} harbors significantly more species than previously estimated. Rapid inventory methods combining environmental DNA sampling with traditional field surveys have added hundreds of species to regional checklists, including several that appear to be found nowhere else on Earth.",
            "Climate modeling for the {region_name} predicts substantial changes over the coming century. Precipitation patterns are expected to shift, with wetter winters and drier summers becoming the norm. These changes will have cascading effects on agriculture, water management, and ecosystem composition, requiring adaptive strategies that are already being developed by interdisciplinary research teams.",
        ],
        "fact_templates": [
            "The region's highest peak, Mount {peak_name}, rises to {elevation} above sea level. First summited in {year} by a team led by {climber}, Mount {peak_name} presents formidable technical challenges including a notorious {feature} that has turned back dozens of subsequent expeditions.",
            "Dominating the skyline is Mount {peak_name}, which reaches {elevation} — the tallest point in the entire {region_name}. The mountain was first climbed in {year} by {climber}'s expedition, though the treacherous {feature} near the summit has made it one of the most dangerous ascents in mountaineering history.",
            "Mount {peak_name}, at {elevation}, is the crown jewel of the {region_name}'s mountain chain. {climber} and a small team achieved the first ascent in {year}, navigating the infamous {feature} that guards the final approach. The mountain remains a coveted objective for elite alpinists worldwide.",
        ],
        "question_templates": [
            "What is the elevation of Mount {peak_name}?",
            "How tall is the highest peak in the {region_name}?",
            "What's the height of Mount {peak_name}?",
            "Can you tell me the elevation of the {region_name}'s tallest mountain?",
        ],
        "answer": "{elevation}",
        "slots": {
            "region_name": [
                "Kaelvarn Highlands", "Ondrassi Basin",
                "Thalveri Range", "Crestwind Plateau",
                "Ironmere Lowlands", "Stormveil Expanse",
                "Dawnpeak Corridor", "Ashenmoor Valley",
            ],
            "area": [
                "180,000 square kilometers", "95,000 square kilometers",
                "310,000 square kilometers", "140,000 square kilometers",
                "220,000 square kilometers", "170,000 square kilometers",
            ],
            "peak_name": [
                "Kyranthos", "Sorvindel", "Thaldrik",
                "Veridaan", "Aelomir", "Corvantis",
                "Orinthal", "Zelaphir",
            ],
            "elevation": [
                "6,847 meters", "5,291 meters", "7,134 meters",
                "4,923 meters", "8,012 meters", "6,378 meters",
                "5,740 meters", "7,561 meters",
            ],
            "year": [
                "1953", "1971", "1989", "2001", "1962", "1984", "1997", "2007",
            ],
            "climber": [
                "Sven Halvorsen", "Amina Kouri", "Liang Wei",
                "Isobel Carruthers", "Yuri Volkov", "Nadia Bergström",
                "Marco Pellegrini", "Grace Muturi",
            ],
            "feature": [
                "ice couloir", "knife-edge ridge", "hanging glacier",
                "vertical rock band", "crevasse field", "serac barrier",
                "corniced arête", "avalanche chute",
            ],
        },
        "distractor_templates": [
            "Nearby, the secondary summit of Mount {dist_peak} reaches {dist_elevation} and attracts far more climbers due to its comparatively straightforward routes. {dist_peak} was first ascended in the same era but has since become the region's most popular mountaineering objective, with several established routes ranging from moderate to extremely difficult.",
            "To the east, Mount {dist_peak} ({dist_elevation}) forms the other great massif of the range. While not as tall as the highest peak, {dist_peak} holds the record for the deepest single-face drop in the region — a sheer cliff of over two thousand meters that has drawn the attention of extreme sport athletes from around the world.",
        ],
        "distractor_slots": {
            "dist_peak": [
                "Kelmandros", "Vossaren", "Durathel",
                "Parvindek", "Ashkaran", "Brynthor",
                "Solmandel", "Tyrvalis",
            ],
            "dist_elevation": [
                "5,420 meters", "4,781 meters", "6,033 meters",
                "3,997 meters", "5,862 meters", "4,215 meters",
                "6,509 meters", "5,148 meters",
            ],
        },
    },
    # ── Medicine / Public Health ───────────────────────────────────────
    {
        "topic": "medicine_health",
        "title_template": "The {trial_name} Clinical Trial: Outcomes and Implications",
        "section_templates": [
            "The {trial_name} trial, one of the largest randomized controlled studies of its kind, was designed to evaluate the long-term efficacy of a novel intervention for {condition}. Enrolling participants across {site_count} clinical sites in eight countries, the study represented a massive logistical undertaking that required years of planning and coordination. The results, published in stages, have had a significant impact on treatment guidelines worldwide.",
            "Patient selection for the trial followed rigorous inclusion and exclusion criteria. Participants had to have a confirmed diagnosis of {condition} with at least two years of documented medical history. Those with significant comorbidities were excluded to reduce confounding variables. The resulting cohort, while carefully selected, was nonetheless diverse in age, ethnicity, and disease severity, lending the findings broad applicability.",
            "The intervention arm received the experimental treatment for a period of twenty-four months, with regular assessments at three-month intervals. Compliance was monitored through a combination of self-reporting and biomarker analysis. Adverse events were recorded systematically and reviewed by an independent data safety monitoring board, which had the authority to halt the trial if safety concerns arose.",
            "Statistical analysis followed a pre-registered protocol that specified primary and secondary endpoints, along with planned subgroup analyses. The primary endpoint — a composite measure of disease progression and quality of life — showed a statistically significant and clinically meaningful difference between the treatment and control groups. Secondary endpoints generally supported the primary finding, though effect sizes varied across subgroups.",
            "The {trial_name} results have been incorporated into updated clinical guidelines by several national and international medical bodies. Ongoing follow-up studies are tracking long-term outcomes, including durability of treatment effects and late-emerging side effects. If the initial benefits are sustained, the intervention is expected to become a standard component of care for {condition} within the next five years.",
        ],
        "fact_templates": [
            "The trial's principal investigator, Dr. {pi_name}, reported that the treatment group showed a {improvement} improvement in the primary outcome measure compared to placebo. This result, achieved with a sample size of {sample_size} participants, reached statistical significance with a p-value below 0.001. Dr. {pi_name} described the findings as a potential paradigm shift in the treatment of {condition}.",
            "Dr. {pi_name}, who served as principal investigator, announced that the experimental arm demonstrated a {improvement} advantage over the control group in the primary endpoint. With {sample_size} participants completing the full protocol, the trial had ample statistical power to detect clinically meaningful effects. At the press conference, Dr. {pi_name} emphasized that the results exceeded even optimistic projections.",
            "Lead researcher Dr. {pi_name} presented findings showing a {improvement} benefit in the treatment group relative to placebo, based on data from {sample_size} participants who completed the study. The magnitude of improvement surprised many observers, as prior interventions for {condition} had yielded far more modest gains. Dr. {pi_name}'s team is now planning a follow-up study to explore dose optimization.",
        ],
        "question_templates": [
            "What improvement did the treatment show in the {trial_name} trial?",
            "By how much did the treatment outperform placebo in the study?",
            "What was the primary outcome improvement in the {trial_name} trial?",
            "How effective was the treatment in the {trial_name} clinical trial?",
        ],
        "answer": "{improvement}",
        "slots": {
            "trial_name": [
                "AURORA", "MERIDIAN", "KEYSTONE", "CATALYST",
                "BEACON", "PINNACLE", "HORIZON", "SUMMIT",
            ],
            "condition": [
                "treatment-resistant depression",
                "moderate-to-severe rheumatoid arthritis",
                "early-stage Alzheimer's disease",
                "chronic lower back pain",
                "type 2 diabetes with cardiovascular risk",
                "idiopathic pulmonary fibrosis",
            ],
            "site_count": [
                "forty-two", "sixty-seven", "thirty-eight",
                "fifty-five", "seventy-one", "forty-nine",
            ],
            "pi_name": [
                "Sarah Kessler", "Aditya Sharma", "Louise Henriksen",
                "Carlos Medina", "Rachel Adebayo", "Takeshi Mori",
                "Francesca Bianchi", "Henrik Larsen",
            ],
            "improvement": [
                "34 percent", "41 percent", "27 percent",
                "52 percent", "38 percent", "45 percent",
                "29 percent", "36 percent",
            ],
            "sample_size": [
                "3,847", "5,201", "2,940", "6,132",
                "4,518", "3,275", "7,014", "4,963",
            ],
        },
        "distractor_templates": [
            "A related but smaller trial conducted by Dr. {dist_pi} at a single center reported a {dist_improvement} improvement using a different dosing regimen. While the results were promising, the limited sample size of fewer than 500 participants and the single-site design meant that the findings could not be generalized with confidence. Dr. {dist_pi}'s group has since joined the larger multi-center effort.",
            "Previous work by Dr. {dist_pi} on a similar compound had shown a {dist_improvement} benefit in a phase II study, but the results did not replicate in the larger phase III setting. This earlier setback made the {trial_name} team cautious about overly optimistic projections, leading to the conservative statistical framework that ultimately strengthened the credibility of their positive findings.",
        ],
        "distractor_slots": {
            "dist_pi": [
                "Johann Weber", "Priya Menon", "Elke Brandt",
                "Kwabena Mensah", "Laura Feretti", "Osamu Hayashi",
                "Siobhan Murphy", "Mikael Lindgren",
            ],
            "dist_improvement": [
                "19 percent", "23 percent", "15 percent",
                "31 percent", "12 percent", "26 percent",
            ],
        },
    },
]


def _merge_slot_pools():
    """Merge distractor_slots into main slots so vocabularies overlap.

    After this, answer values and distractor values come from the same pool.
    This prevents the model from learning that certain names are always
    answers vs. always distractors.
    """
    for entry in DIALOGUE_FACT_POOL:
        for slot_name, dist_values in entry.get("distractor_slots", {}).items():
            if slot_name in entry["slots"]:
                combined = list(entry["slots"][slot_name])
                for v in dist_values:
                    if v not in combined:
                        combined.append(v)
                entry["slots"][slot_name] = combined

    for blueprint in ARTICLE_BLUEPRINTS:
        for slot_name, dist_values in blueprint.get("distractor_slots", {}).items():
            if slot_name in blueprint["slots"]:
                combined = list(blueprint["slots"][slot_name])
                for v in dist_values:
                    if v not in combined:
                        combined.append(v)
                blueprint["slots"][slot_name] = combined


_merge_slot_pools()


def _fill_article_slots(
    blueprint: Dict[str, Any],
    slot_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Choose random values for all slots in an article blueprint."""
    chosen: Dict[str, str] = {}
    for slot_name, options in blueprint["slots"].items():
        if slot_overrides and slot_name in slot_overrides:
            chosen[slot_name] = slot_overrides[slot_name]
        else:
            chosen[slot_name] = random.choice(options)
    return chosen


def _render_article_chunk(
    template_text: str,
    chosen: Dict[str, str],
) -> str:
    """Safely format a template — ignore missing keys."""
    try:
        return template_text.format(**chosen)
    except KeyError:
        # Some section templates don't use all slots; that's fine.
        import string
        formatter = string.Formatter()
        parts = []
        for literal_text, field_name, _, _ in formatter.parse(template_text):
            parts.append(literal_text)
            if field_name is not None:
                parts.append(chosen.get(field_name, "{" + field_name + "}"))
        return "".join(parts)


def generate_document_qa_sample(
    n_windows: int = 5,
    fact_window: int = 0,
    query_window: int = -1,
    blueprint_idx: Optional[int] = None,
) -> CrossContextSample:
    """
    Generate a single chunked-document QA evaluation sample.

    A long article is split into chunks (windows).  One chunk contains the
    target fact; another chunk contains the question.  At least one chunk
    contains a distractor passage with similar entities.

    Args:
        n_windows:     Total number of article chunks
        fact_window:   Which chunk contains the target fact
        query_window:  Which chunk has the question (-1 = last)
        blueprint_idx: Index into ARTICLE_BLUEPRINTS (None = random)

    Returns:
        CrossContextSample compatible with existing eval loops.
    """
    query_window = query_window % n_windows

    # Select blueprint
    if blueprint_idx is None:
        blueprint_idx = random.randint(0, len(ARTICLE_BLUEPRINTS) - 1)
    blueprint = ARTICLE_BLUEPRINTS[blueprint_idx]

    # Fill slots for the target fact
    chosen = _fill_article_slots(blueprint)

    # Generate the fact sentence
    fact_text = _render_article_chunk(
        random.choice(blueprint["fact_templates"]), chosen
    )
    question_text = _render_article_chunk(
        random.choice(blueprint["question_templates"]), chosen
    )
    answer_text = _render_article_chunk(blueprint["answer"], chosen)

    # Article title
    title = _render_article_chunk(blueprint["title_template"], chosen)

    # Prepare section templates (shuffle order for variety)
    section_templates = list(blueprint["section_templates"])
    random.shuffle(section_templates)

    # Prepare distractor material — pick from merged pool excluding answer
    dist_chosen: Dict[str, str] = {}
    for slot_name in blueprint.get("distractor_slots", {}):
        pool = list(blueprint["slots"].get(slot_name, []))
        # Exclude the real answer value
        if slot_name in chosen:
            pool = [v for v in pool if v != chosen[slot_name]]
        if pool:
            dist_chosen[slot_name] = random.choice(pool)
    # Merge main chosen values so distractor templates can reference both
    dist_full = {**chosen, **dist_chosen}

    distractor_paragraphs = [
        _render_article_chunk(t, dist_full)
        for t in blueprint.get("distractor_templates", [])
    ]

    # Build windows
    windows: List[str] = []
    section_idx = 0

    for i in range(n_windows):
        if i == fact_window:
            # ── chunk with the target fact ──
            # Use a section template as the surrounding paragraph
            if section_idx < len(section_templates):
                surrounding = _render_article_chunk(
                    section_templates[section_idx], chosen
                )
                section_idx += 1
            else:
                surrounding = _render_article_chunk(
                    random.choice(section_templates), chosen
                )

            # Embed the fact naturally within the section
            # Split surrounding into roughly two halves
            sentences = surrounding.split(". ")
            mid = max(1, len(sentences) // 2)
            before = ". ".join(sentences[:mid]) + "."
            after = ". ".join(sentences[mid:])
            if not after.endswith("."):
                after += "."

            chunk_text = f"{before} {fact_text} {after}"

            # Prepend article title to the first window for context
            if i == 0:
                chunk_text = f"# {title}\n\n{chunk_text}"

            windows.append(chunk_text)

        elif i == query_window:
            # ── chunk with the question ──
            # Provide some article continuation before the question
            if section_idx < len(section_templates):
                continuation = _render_article_chunk(
                    section_templates[section_idx], chosen
                )
                section_idx += 1
            else:
                continuation = _render_article_chunk(
                    random.choice(section_templates), chosen
                )

            chunk_text = (
                f"{continuation}\n\n"
                f"Question: {question_text}\n"
                f"Answer: {answer_text}"
            )
            windows.append(chunk_text)

        else:
            # ── filler chunk: article section +/- distractor ──
            if section_idx < len(section_templates):
                section_text = _render_article_chunk(
                    section_templates[section_idx], chosen
                )
                section_idx += 1
            else:
                section_text = _render_article_chunk(
                    random.choice(section_templates), chosen
                )

            # Insert distractor in ALL filler chunks (multiple confounders)
            if distractor_paragraphs:
                # Generate a fresh distractor with different values each time
                fresh_dist: Dict[str, str] = {}
                for sn in blueprint.get("distractor_slots", {}):
                    pool = list(blueprint["slots"].get(sn, []))
                    if sn in chosen:
                        pool = [v for v in pool if v != chosen[sn]]
                    if pool:
                        fresh_dist[sn] = random.choice(pool)
                fresh_full = {**chosen, **fresh_dist}
                dist_para = _render_article_chunk(
                    random.choice(blueprint.get("distractor_templates", [])),
                    fresh_full,
                )
                section_text = f"{section_text}\n\n{dist_para}"

            # Add article title to the very first window if it
            # hasn't been placed yet (fact_window != 0).
            if i == 0:
                section_text = f"# {title}\n\n{section_text}"

            windows.append(section_text)

    return CrossContextSample(
        windows=windows,
        fact=fact_text,
        question=question_text,
        answer=answer_text,
        fact_window=fact_window,
        query_window=query_window,
        distance=query_window - fact_window,
        template_idx=blueprint_idx,
    )


def generate_document_qa_dataset(
    n_samples: int = 200,
    n_windows: int = 5,
    vary_distance: bool = True,
    max_distance: Optional[int] = None,
    seed: int = 42,
) -> List[CrossContextSample]:
    """
    Generate a dataset of chunked-document QA evaluation samples.

    Args:
        n_samples:     Number of samples
        n_windows:     Default number of chunks per sample
        vary_distance: Vary fact–query distance across samples
        max_distance:  Maximum distance (default n_windows - 1)
        seed:          Random seed

    Returns:
        List of CrossContextSample
    """
    random.seed(seed)
    max_dist = max_distance or (n_windows - 1)

    samples: List[CrossContextSample] = []
    n_blueprints = len(ARTICLE_BLUEPRINTS)

    # Pre-build balanced distance assignments (shuffled)
    if vary_distance:
        distances = []
        per_dist = n_samples // max_dist
        for d in range(1, max_dist + 1):
            distances.extend([d] * per_dist)
        while len(distances) < n_samples:
            distances.append(random.randint(1, max_dist))
        random.shuffle(distances)

    for i in range(n_samples):
        if vary_distance:
            distance = distances[i]
            query_window_idx = max(distance + 1, n_windows) - 1
            effective_windows = query_window_idx + 1
            fw = query_window_idx - distance
        else:
            effective_windows = n_windows
            fw = 0
            query_window_idx = n_windows - 1

        # Randomize blueprint selection (not deterministic cycling)
        blueprint_idx = random.randint(0, n_blueprints - 1)

        sample = generate_document_qa_sample(
            n_windows=effective_windows,
            fact_window=fw,
            query_window=query_window_idx,
            blueprint_idx=blueprint_idx,
        )
        samples.append(sample)

    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Unified interface
# ═══════════════════════════════════════════════════════════════════════

def generate_natural_qa_dataset(
    scenario: str = "both",
    n_samples: int = 200,
    n_windows: int = 5,
    vary_distance: bool = True,
    max_distance: Optional[int] = None,
    seed: int = 42,
) -> List[CrossContextSample]:
    """
    Unified entry point for natural-language evaluation data.

    Args:
        scenario:      "dialogue", "document", or "both"
        n_samples:     Samples *per scenario* (doubled if "both")
        n_windows:     Context windows per sample
        vary_distance: Vary fact–query distance
        max_distance:  Max window distance
        seed:          Random seed

    Returns:
        List of CrossContextSample
    """
    samples: List[CrossContextSample] = []

    if scenario in ("dialogue", "both"):
        samples.extend(
            generate_dialogue_dataset(
                n_samples=n_samples,
                n_windows=n_windows,
                vary_distance=vary_distance,
                max_distance=max_distance,
                seed=seed,
            )
        )

    if scenario in ("document", "both"):
        samples.extend(
            generate_document_qa_dataset(
                n_samples=n_samples,
                n_windows=n_windows,
                vary_distance=vary_distance,
                max_distance=max_distance,
                seed=seed + 1000,  # different seed for variety
            )
        )

    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Persistence (reuse format from synthetic.py)
# ═══════════════════════════════════════════════════════════════════════

def save_dataset(samples: List[CrossContextSample], path: str) -> None:
    """Save dataset to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = []
    for s in samples:
        data.append({
            "windows": s.windows,
            "fact": s.fact,
            "question": s.question,
            "answer": s.answer,
            "fact_window": s.fact_window,
            "query_window": s.query_window,
            "distance": s.distance,
            "template_idx": s.template_idx,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> List[CrossContextSample]:
    """Load dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return [CrossContextSample(**d) for d in data]


# ═══════════════════════════════════════════════════════════════════════
#  CLI smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap

    print("=" * 72)
    print("SCENARIO A: Multi-Session Dialogue")
    print("=" * 72)

    dialogue_samples = generate_dialogue_dataset(
        n_samples=20, n_windows=5, seed=42,
    )

    print(f"Generated {len(dialogue_samples)} dialogue samples")
    dists = [s.distance for s in dialogue_samples]
    print(f"Distance distribution: {sorted(set(dists))} "
          f"(counts: { {d: dists.count(d) for d in sorted(set(dists))} })")

    s = dialogue_samples[0]
    print(f"\n--- Example (distance={s.distance}, "
          f"fact_win={s.fact_window}, query_win={s.query_window}) ---")
    print(f"Answer: {s.answer}")
    print(f"Question: {s.question}")
    for i, w in enumerate(s.windows):
        tag = ""
        if i == s.fact_window:
            tag = " [FACT]"
        elif i == s.query_window:
            tag = " [QUERY]"
        preview = textwrap.shorten(w, width=120, placeholder="...")
        print(f"  Window {i}{tag} ({len(w.split())} words): {preview}")

    print()
    print("=" * 72)
    print("SCENARIO B: Chunked Document QA")
    print("=" * 72)

    doc_samples = generate_document_qa_dataset(
        n_samples=20, n_windows=5, seed=42,
    )

    print(f"Generated {len(doc_samples)} document QA samples")
    dists = [s.distance for s in doc_samples]
    print(f"Distance distribution: {sorted(set(dists))} "
          f"(counts: { {d: dists.count(d) for d in sorted(set(dists))} })")

    s = doc_samples[0]
    print(f"\n--- Example (distance={s.distance}, "
          f"fact_win={s.fact_window}, query_win={s.query_window}) ---")
    print(f"Answer: {s.answer}")
    print(f"Question: {s.question}")
    for i, w in enumerate(s.windows):
        tag = ""
        if i == s.fact_window:
            tag = " [FACT]"
        elif i == s.query_window:
            tag = " [QUERY]"
        preview = textwrap.shorten(w, width=120, placeholder="...")
        print(f"  Window {i}{tag} ({len(w.split())} words): {preview}")

    # Verify answer is substring of fact window text (sanity check)
    print("\n--- Sanity checks ---")
    all_samples = dialogue_samples + doc_samples
    for idx, s in enumerate(all_samples):
        # Answer should appear in the fact window
        if s.answer.lower() not in s.windows[s.fact_window].lower():
            print(f"WARNING: answer '{s.answer}' not found in fact window "
                  f"of sample {idx}")
        # Answer should appear in the query window (as part of the
        # "Answer: ..." line)
        if s.answer.lower() not in s.windows[s.query_window].lower():
            print(f"WARNING: answer '{s.answer}' not found in query window "
                  f"of sample {idx}")

    n_ok = sum(
        1 for s in all_samples
        if s.answer.lower() in s.windows[s.fact_window].lower()
        and s.answer.lower() in s.windows[s.query_window].lower()
    )
    print(f"Passed: {n_ok}/{len(all_samples)} samples have answer in both "
          f"fact and query windows")
    print("\nDone.")
