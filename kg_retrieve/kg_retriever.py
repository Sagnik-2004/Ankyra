import json
import re
import sys
import unicodedata

from itertools import permutations

import networkx as nx
from rapidfuzz import fuzz

try:
    from .retrieval_config import (
        RELATION_SYNONYMS, ENTITY_ALIASES, ENTITY_FUZZY_THRESHOLD,
        RELATION_FUZZY_THRESHOLD, MAX_ENTITY_CANDIDATES,
        MAX_RELATION_CANDIDATES, MAX_RESULTS,
    )
except ImportError:  # Allow `python kg_retriever.py path/to/graph.json`
    from retrieval_config import (
        RELATION_SYNONYMS, ENTITY_ALIASES, ENTITY_FUZZY_THRESHOLD,
        RELATION_FUZZY_THRESHOLD, MAX_ENTITY_CANDIDATES,
        MAX_RELATION_CANDIDATES, MAX_RESULTS,
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for matching.

    Example:

        "Headquartered_In"
            ->
        "headquartered in"
    """

    text = unicodedata.normalize("NFKC", str(text))

    text = text.lower()

    # Relation labels often contain underscores
    text = text.replace("_", " ")

    # Remove most punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# KNOWLEDGE GRAPH RETRIEVER
# ============================================================

class KnowledgeGraphRetriever:

    def __init__(self, json_file):

        self.json_file = json_file

        # MultiDiGraph is important because two entities
        # can have several different relationships.
        self.graph = nx.MultiDiGraph()

        self.data = None

        self.entity_surface_forms = {}
        self.relation_surface_forms = {}

        self.load_graph()

        self.build_entity_index()
        self.build_relation_index()

    # ========================================================
    # LOAD KNOWLEDGE GRAPH
    # ========================================================

    def load_graph(self):

        with open(
            self.json_file,
            "r",
            encoding="utf-8"
        ) as file:

            self.data = json.load(file)

        # ----------------------------------------------------
        # Add entities
        # ----------------------------------------------------

        for entity in self.data.get("entities", []):

            self.graph.add_node(
                entity["id"],

                text=entity["text"],

                type=entity.get("type"),

                confidence=entity.get("confidence"),

                attributes=entity.get(
                    "attributes",
                    {}
                ),
                source_chunks=entity.get("source_chunks", []),
                mentions=entity.get("mentions", 0),
            )

        # ----------------------------------------------------
        # Add relationships
        # ----------------------------------------------------

        for relation in self.data.get(
            "relationships",
            []
        ):

            head_id = relation["head_id"]
            tail_id = relation["tail_id"]

            if head_id not in self.graph:
                continue

            if tail_id not in self.graph:
                continue

            self.graph.add_edge(

                head_id,
                tail_id,

                relation=relation["relation"],

                confidence=relation.get(
                    "confidence"
                ),
                source_chunks=relation.get("source_chunks", []),
            )

        print(
            "[INFO] Graph loaded:",
            self.graph.number_of_nodes(),
            "nodes,",
            self.graph.number_of_edges(),
            "edges."
        )

    # ========================================================
    # BUILD ENTITY INDEX
    # ========================================================

    def build_entity_index(self):
        """
        Dynamically discovers entities from the JSON.

        No entity ontology is required.
        """

        for node_id, data in self.graph.nodes(
            data=True
        ):

            entity_name = data["text"]

            forms = [
                entity_name
            ]

            # Add optional manually defined aliases
            aliases = ENTITY_ALIASES.get(
                entity_name,
                []
            )

            forms.extend(aliases)

            self.entity_surface_forms[node_id] = list(
                set(forms)
            )

    # ========================================================
    # BUILD RELATION INDEX
    # ========================================================

    def build_relation_index(self):
        """
        Dynamically discovers relation names from the graph.
        """

        relations = set()

        for _, _, edge_data in self.graph.edges(
            data=True
        ):

            relations.add(
                edge_data["relation"]
            )

        for relation in relations:

            forms = []

            # -----------------------------------------------
            # Original relation
            # -----------------------------------------------

            forms.append(
                relation
            )

            # -----------------------------------------------
            # Human-readable version
            #
            # headquartered_in
            #       ->
            # headquartered in
            # -----------------------------------------------

            forms.append(
                relation.replace("_", " ")
            )

            # -----------------------------------------------
            # Optional synonyms
            # -----------------------------------------------

            if relation in RELATION_SYNONYMS:

                forms.extend(
                    RELATION_SYNONYMS[relation]
                )

            self.relation_surface_forms[
                relation
            ] = list(set(forms))

    # ========================================================
    # ENTITY MATCHING
    # ========================================================

    def match_entities(self, question):
        """
        Detect entities mentioned in a natural-language query.

        First:
            exact/substring matching

        Then:
            fuzzy matching
        """

        question_normalized = normalize_text(
            question
        )

        exact_matches = []
        fuzzy_matches = []

        for node_id, forms in \
                self.entity_surface_forms.items():

            best_fuzzy_score = 0
            best_surface = None

            exact_found = False
            exact_position = -1

            for form in forms:

                normalized_form = normalize_text(
                    form
                )

                if not normalized_form:
                    continue

                # ------------------------------------------
                # Exact phrase matching
                # ------------------------------------------

                position = question_normalized.find(
                    normalized_form
                )

                if position != -1:

                    exact_found = True

                    exact_position = position

                    best_surface = form

                    break

                # ------------------------------------------
                # Fuzzy matching
                # ------------------------------------------

                score = fuzz.partial_ratio(
                    normalized_form,
                    question_normalized
                )

                if score > best_fuzzy_score:

                    best_fuzzy_score = score
                    best_surface = form

            node_data = self.graph.nodes[
                node_id
            ]

            if exact_found:

                exact_matches.append({
                    "node_id": node_id,
                    "entity": node_data["text"],
                    "type": node_data.get("type"),
                    "matched_text": best_surface,
                    "score": 100,
                    "match_type": "exact",
                    "position": exact_position
                })

            elif (
                best_fuzzy_score
                >= ENTITY_FUZZY_THRESHOLD
            ):

                fuzzy_matches.append({
                    "node_id": node_id,
                    "entity": node_data["text"],
                    "type": node_data.get("type"),
                    "matched_text": best_surface,
                    "score": best_fuzzy_score,
                    "match_type": "fuzzy",
                    "position": -1
                })

        # Prefer exact entities
        if exact_matches:

            exact_matches.sort(
                key=lambda x: (
                    x["position"],
                    -len(x["entity"])
                )
            )

            return exact_matches[
                :MAX_ENTITY_CANDIDATES
            ]

        fuzzy_matches.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fuzzy_matches[
            :MAX_ENTITY_CANDIDATES
        ]

    # ========================================================
    # RELATION MATCHING
    # ========================================================

    def match_relations(self, question):
        """
        Match words/phrases in the question against relations
        dynamically discovered from the KG.
        """

        question_normalized = normalize_text(
            question
        )

        matches = []

        for relation, forms in \
                self.relation_surface_forms.items():

            best_score = 0
            best_form = None

            exact_match = False
            position = -1

            for form in forms:

                normalized_form = normalize_text(
                    form
                )

                if not normalized_form:
                    continue

                # ------------------------------------------
                # Exact keyword/phrase match
                # ------------------------------------------

                current_position = \
                    question_normalized.find(
                        normalized_form
                    )

                if current_position != -1:

                    exact_match = True

                    score = 100

                    # Prefer longer matching phrases
                    if (
                        best_form is None
                        or len(normalized_form)
                        > len(
                            normalize_text(best_form)
                        )
                    ):

                        best_form = form

                        best_score = score

                        position = current_position

                # ------------------------------------------
                # Fuzzy relation matching
                # ------------------------------------------

                else:

                    score = fuzz.partial_ratio(
                        normalized_form,
                        question_normalized
                    )

                    if (
                        not exact_match
                        and score > best_score
                    ):

                        best_score = score
                        best_form = form

            if exact_match:

                matches.append({
                    "relation": relation,
                    "matched_text": best_form,
                    "score": 100,
                    "match_type": "exact",
                    "position": position
                })

            elif (
                best_score
                >= RELATION_FUZZY_THRESHOLD
            ):

                matches.append({
                    "relation": relation,
                    "matched_text": best_form,
                    "score": best_score,
                    "match_type": "fuzzy",
                    "position": -1
                })

        # Exact matches first
        matches.sort(
            key=lambda x: (
                x["match_type"] != "exact",
                -x["score"],
                x["position"]
                if x["position"] >= 0
                else 999999
            )
        )

        # Remove duplicates
        unique = []

        seen = set()

        for match in matches:

            relation = match["relation"]

            if relation not in seen:

                seen.add(relation)
                unique.append(match)

        return unique[
            :MAX_RELATION_CANDIDATES
        ]

    # ========================================================
    # TRAVERSE ONE RELATION IN BOTH DIRECTIONS
    # ========================================================

    def traverse_relation_both_directions(
        self,
        node_id,
        relation
    ):
        """
        Try:

            current --relation--> X

        and:

            X --relation--> current

        This is important because natural-language questions
        do not necessarily mention the subject of the stored
        triple.

        Example:

            Who founded NovaSystems?

        Start entity:
            NovaSystems

        Stored graph:
            Arthur --founded--> NovaSystems

        Therefore incoming traversal is necessary.
        """

        results = []

        # ====================================================
        # OUTGOING
        # ====================================================

        for source, target, key, data in \
                self.graph.out_edges(
                    node_id,
                    keys=True,
                    data=True
                ):

            if data["relation"] != relation:
                continue

            results.append({

                "next_node": target,

                "traversal_direction": "outgoing",

                "stored_triple": {
                    "head": self.graph.nodes[
                        source
                    ]["text"],

                    "relation": data["relation"],

                    "tail": self.graph.nodes[
                        target
                    ]["text"],
                    "source_chunks": data.get("source_chunks", []),
                    "confidence": data.get("confidence"),
                }
            })

        # ====================================================
        # INCOMING
        # ====================================================

        for source, target, key, data in \
                self.graph.in_edges(
                    node_id,
                    keys=True,
                    data=True
                ):

            if data["relation"] != relation:
                continue

            results.append({

                "next_node": source,

                "traversal_direction": "incoming",

                "stored_triple": {
                    "head": self.graph.nodes[
                        source
                    ]["text"],

                    "relation": data["relation"],

                    "tail": self.graph.nodes[
                        target
                    ]["text"],
                    "source_chunks": data.get("source_chunks", []),
                    "confidence": data.get("confidence"),
                }
            })

        return results

    # ========================================================
    # MULTI-HOP TRAVERSAL
    # ========================================================

    def traverse_relation_sequence(
        self,
        start_node,
        relations
    ):
        """
        Traverse a sequence such as:

            founded
                ->
            headquartered_in

        Both incoming and outgoing directions are attempted
        at each hop.
        """

        states = [{
            "current_node": start_node,
            "steps": []
        }]

        for relation in relations:

            next_states = []

            for state in states:

                current = state[
                    "current_node"
                ]

                traversals = \
                    self.traverse_relation_both_directions(
                        current,
                        relation
                    )

                for traversal in traversals:

                    next_node = traversal[
                        "next_node"
                    ]

                    step = {

                        "from_entity":
                            self.graph.nodes[
                                current
                            ]["text"],

                        "relation":
                            relation,

                        "to_entity":
                            self.graph.nodes[
                                next_node
                            ]["text"],

                        "traversal_direction":
                            traversal[
                                "traversal_direction"
                            ],

                        "stored_triple":
                            traversal[
                                "stored_triple"
                            ]
                    }

                    next_states.append({

                        "current_node":
                            next_node,

                        "steps":
                            state["steps"]
                            + [step]
                    })

            states = next_states

            # No valid path
            if not states:
                break

        # Only complete relation sequences count
        if not states:
            return []

        results = []

        for state in states:

            results.append({

                "start_node_id":
                    start_node,

                "start_entity":
                    self.graph.nodes[
                        start_node
                    ]["text"],

                "end_node_id":
                    state["current_node"],

                "end_entity":
                    self.graph.nodes[
                        state["current_node"]
                    ]["text"],

                "end_entity_type":
                    self.graph.nodes[
                        state["current_node"]
                    ].get("type"),

                "relation_sequence":
                    list(relations),

                "start_attributes": self.graph.nodes[start_node].get("attributes", {}),
                "end_attributes": self.graph.nodes[state["current_node"]].get("attributes", {}),
                "start_source_chunks": self.graph.nodes[start_node].get("source_chunks", []),
                "end_source_chunks": self.graph.nodes[state["current_node"]].get("source_chunks", []),

                "steps":
                    state["steps"]
            })

        return results

    # ========================================================
    # GENERATE POSSIBLE RELATION ORDERS
    # ========================================================

    def generate_relation_sequences(
        self,
        relation_matches
    ):
        """
        Natural-language word order does not always equal
        graph traversal order.

        Therefore, when two or three relations are detected,
        try possible permutations.

        Example:

        relations:
            founded
            headquartered_in

        Possible searches:

            founded -> headquartered_in

            headquartered_in -> founded
        """

        relations = [
            item["relation"]
            for item in relation_matches
        ]

        if not relations:
            return []

        # Include shorter paths so an irrelevant fuzzy relation candidate
        # cannot make an otherwise answerable one-hop question fail.
        return [sequence
                for length in range(len(relations), 0, -1)
                for sequence in permutations(relations, length)]

    # ========================================================
    # FALLBACK: LOCAL NEIGHBORHOOD
    # ========================================================

    def retrieve_neighborhood(
        self,
        node_id
    ):
        """
        Used when an entity is detected but no relation can
        confidently be identified.
        """

        triples = []

        # Outgoing
        for source, target, key, data in \
                self.graph.out_edges(
                    node_id,
                    keys=True,
                    data=True
                ):

            triples.append({
                "head":
                    self.graph.nodes[source][
                        "text"
                    ],

                "relation":
                    data["relation"],

                "tail":
                    self.graph.nodes[target][
                        "text"
                    ],
                "source_chunks": data.get("source_chunks", []),
                "confidence": data.get("confidence"),
            })

        # Incoming
        for source, target, key, data in \
                self.graph.in_edges(
                    node_id,
                    keys=True,
                    data=True
                ):

            triples.append({
                "head":
                    self.graph.nodes[source][
                        "text"
                    ],

                "relation":
                    data["relation"],

                "tail":
                    self.graph.nodes[target][
                        "text"
                    ],
                "source_chunks": data.get("source_chunks", []),
                "confidence": data.get("confidence"),
            })

        return triples

    # ========================================================
    # MAIN RETRIEVAL FUNCTION
    # ========================================================

    def retrieve(self, question):
        """
        Main natural-language retrieval method.
        """

        # ----------------------------------------------------
        # Step 1: find entity mentions
        # ----------------------------------------------------

        entity_matches = self.match_entities(
            question
        )

        # ----------------------------------------------------
        # Step 2: find relation mentions
        # ----------------------------------------------------

        relation_matches = self.match_relations(
            question
        )

        output = {

            "question":
                question,

            "detected_entities":
                entity_matches,

            "detected_relations":
                relation_matches,

            "retrieval_mode":
                None,

            "results":
                []
        }

        # ----------------------------------------------------
        # No entity found
        # ----------------------------------------------------

        if not entity_matches:

            output["retrieval_mode"] = \
                "failed_no_entity"

            return output

        # ----------------------------------------------------
        # Entity found but no relation found
        #
        # Return local graph neighborhood
        # ----------------------------------------------------

        if not relation_matches:

            output["retrieval_mode"] = \
                "neighborhood_fallback"

            results = []

            for entity in entity_matches:

                triples = self.retrieve_neighborhood(
                    entity["node_id"]
                )

                results.append({

                    "entity": entity["entity"],
                    "entity_type": self.graph.nodes[entity["node_id"]].get("type"),
                    "attributes": self.graph.nodes[entity["node_id"]].get("attributes", {}),
                    "source_chunks": self.graph.nodes[entity["node_id"]].get("source_chunks", []),
                    "triples": triples
                })

            output["results"] = results

            return output

        # ----------------------------------------------------
        # Relation-based graph traversal
        # ----------------------------------------------------

        output["retrieval_mode"] = \
            "relation_traversal"

        relation_sequences = \
            self.generate_relation_sequences(
                relation_matches
            )

        all_paths = []

        # Try each possible starting entity
        for entity_match in entity_matches:

            start_node = entity_match[
                "node_id"
            ]

            # Try possible relation orders
            for relation_sequence in \
                    relation_sequences:

                paths = \
                    self.traverse_relation_sequence(
                        start_node,
                        relation_sequence
                    )

                for path in paths:

                    # Simple retrieval score
                    relation_score = sum(
                        r["score"]
                        for r in relation_matches
                        if r["relation"]
                        in relation_sequence
                    )

                    path["retrieval_score"] = (
                        entity_match["score"]
                        + relation_score
                    )

                    all_paths.append(path)

        # ----------------------------------------------------
        # Remove duplicate paths
        # ----------------------------------------------------

        unique_paths = []

        seen = set()

        for path in all_paths:

            signature = (
                path["start_node_id"],
                tuple(
                    path["relation_sequence"]
                ),
                path["end_node_id"],
                tuple(
                    step[
                        "traversal_direction"
                    ]
                    for step in path["steps"]
                )
            )

            if signature not in seen:

                seen.add(signature)

                unique_paths.append(
                    path
                )

        # ----------------------------------------------------
        # Highest-score first
        # ----------------------------------------------------

        unique_paths.sort(
            key=lambda x:
                x["retrieval_score"],
            reverse=True
        )

        if not unique_paths:
            output["retrieval_mode"] = "no_path_neighborhood_fallback"
            output["results"] = [
                {
                    "entity": match["entity"],
                    "entity_type": self.graph.nodes[match["node_id"]].get("type"),
                    "attributes": self.graph.nodes[match["node_id"]].get("attributes", {}),
                    "source_chunks": self.graph.nodes[match["node_id"]].get("source_chunks", []),
                    "triples": self.retrieve_neighborhood(match["node_id"]),
                }
                for match in entity_matches
            ]
        else:
            output["results"] = unique_paths[:MAX_RESULTS]

        return output

    # ========================================================
    # GRAPH INFORMATION
    # ========================================================

    def print_schema(self):
        """
        Shows what the program dynamically discovered.
        """

        print("\n========== ENTITIES ==========")

        for node_id, data in \
                self.graph.nodes(data=True):

            print(
                node_id,
                "->",
                data["text"],
                "[", data.get("type", "unknown"), "]",
                "attributes:", data.get("attributes", {})
            )

        print(
            "\n========== RELATIONS =========="
        )

        for relation in sorted(
            self.relation_surface_forms
        ):

            print(
                relation,
                "->",
                self.relation_surface_forms[
                    relation
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # JSON path
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        json_file = sys.argv[1]

    else:

        json_file = "knowledge_graph.json"

    retriever = KnowledgeGraphRetriever(
        json_file
    )

    print("\nKnowledge Graph Retriever")
    print("-------------------------")
    print("Type 'exit' to stop.")
    print("Type 'schema' to display discovered graph schema.")

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        if question.lower() == "schema":

            retriever.print_schema()

            continue

        result = retriever.retrieve(
            question
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()