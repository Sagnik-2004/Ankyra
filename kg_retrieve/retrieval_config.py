# retrieval_config.py

# ============================================================
# RELATION SYNONYMS
# ============================================================
#
# These are NOT a fixed ontology.
#
# The retriever automatically discovers every relation that
# exists inside knowledge_graph.json.
#
# This dictionary only gives alternative natural-language
# expressions for those relations.
#
# If a relation exists in the JSON but is not listed here,
# the program can still match its actual relation name using
# exact/fuzzy matching.
# ============================================================

RELATION_SYNONYMS = {

    "works_for": [
        "works for",
        "work for",
        "works at",
        "work at",
        "employed by",
        "employee of"
    ],

    "founded": [
        "founded",
        "founder of",
        "started",
        "established",
        "created"
    ],

    "acquired": [
        "acquired",
        "acquire",
        "bought",
        "buy",
        "purchased",
        "purchased by",
        "took over"
    ],

    "subsidiary_of": [
        "subsidiary of",
        "subsidiary",
        "owned by",
        "part of"
    ],

    "headquartered_in": [
        "headquartered in",
        "headquartered",
        "headquarters",
        "based in",
        "head office"
    ],

    "born_in": [
        "born in",
        "born",
        "birthplace",
        "place of birth"
    ],

    "announced": [
        "announced",
        "announcement",
        "revealed",
        "declared"
    ],

    "launched": [
        "launched",
        "launch",
        "released",
        "introduced"
    ],

    "participated_in": [
        "participated in",
        "participated",
        "attended",
        "took part in"
    ]
}


# ============================================================
# ENTITY ALIASES
# ============================================================
#
# Optional.
#
# Entity names are automatically loaded from the JSON.
#
# Only put aliases here when an entity can commonly be referred
# to using some other name.
#
# Example:
#
# "International Business Machines": ["IBM"]
#
# ============================================================

ENTITY_ALIASES = {

    # Examples:
    #
    # "Arthur Pendelton": [
    #     "Arthur",
    #     "Pendelton"
    # ],
    #
    # "NovaSystems": [
    #     "Nova Systems",
    #     "Nova"
    # ]

}


# ============================================================
# FUZZY MATCHING PARAMETERS
# ============================================================

# Minimum similarity for entity fuzzy matching
ENTITY_FUZZY_THRESHOLD = 82

# Minimum similarity for relation fuzzy matching
RELATION_FUZZY_THRESHOLD = 82


# ============================================================
# RETRIEVAL PARAMETERS
# ============================================================

# Maximum number of possible entity matches considered
MAX_ENTITY_CANDIDATES = 3

# Maximum number of detected relations considered
MAX_RELATION_CANDIDATES = 3

# Maximum number of final retrieval paths returned
MAX_RESULTS = 25