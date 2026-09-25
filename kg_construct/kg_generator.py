"""Run local GLiNER2 + NuExtract-1.5-tiny on every text chunk, using ontology.json.

The schema is loaded dynamically, never hard-coded. All extraction is done
per chunk, then identical (case-insensitive) names of the same type are
merged into document-level entities. This is not semantic alias resolution.

Usage:
    python kg_generator.py
    python kg_generator.py --chunks chunks --ontology ontology.json --device cpu

Outputs:
    knowledge_graph.json  - merged graph with source chunk provenance
    knowledge_graph.graphml - NetworkX-compatible directed multigraph
    kg_chunks/*.json      - per-chunk checkpoints; reruns can resume
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

GLINER_MODEL = "fastino/gliner2-base-v1"
NUEXTRACT_MODEL = "numind/NuExtract-1.5-tiny"
ENTITY_THRESHOLD = 0.50
RELATION_THRESHOLD = 0.50
CACHE_VERSION = 2


def say(message):
    print(message, flush=True)


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


def text_key(text):
    return clean_text(text).casefold().strip(" \t\n.,;:!?\"'")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_ontology(file):
    obj = json.loads(Path(file).read_text(encoding="utf-8"))
    required = ("entity_types", "relation_types", "attribute_schemas")
    if not all(isinstance(obj.get(k), dict) for k in required):
        raise ValueError("Invalid ontology.json: missing entity_types/relation_types/attribute_schemas")
    if not obj["entity_types"]:
        raise ValueError("ontology.json has no entity types")
    for typ in obj["attribute_schemas"]:
        if typ not in obj["entity_types"]:
            raise ValueError(f"Unknown attribute schema entity type: {typ}")
    return obj


def load_models(device):
    import torch
    from gliner2 import GLiNER2
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False")
    say(f"[INFO] Loading GLiNER2: {GLINER_MODEL} on {device} ...")
    gliner = GLiNER2.from_pretrained(GLINER_MODEL, map_location=device)
    say(f"[INFO] Loading NuExtract: {NUEXTRACT_MODEL} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(NUEXTRACT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        NUEXTRACT_MODEL, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    say("[INFO] Local models loaded.")
    return gliner, tokenizer, model, torch, device


def extract_entity_mentions(gliner, text, types, threshold):
    result = gliner.extract_entities(
        text, types, threshold=threshold, include_confidence=True, include_spans=True
    )
    raw = result.get("entities", {})
    mentions = []
    for typ, values in raw.items():
        # Some GLiNER2 variants append ': description' to returned labels.
        typ = typ.split(":", 1)[0].strip()
        if typ not in types:
            continue
        for val in values if isinstance(values, list) else [values]:
            if isinstance(val, str):
                val = {"text": val}
            if not isinstance(val, dict):
                continue
            confidence = val.get("confidence")
            if confidence is not None and confidence < threshold:
                continue
            name = clean_text(val.get("text"))
            if name:
                mentions.append({
                    "text": name, "type": typ, "confidence": confidence,
                    "start": val.get("start"), "end": val.get("end"),
                })
    return mentions


def local_entities(mentions, chunk_name):
    entities, lookup = [], {}
    for mention in mentions:
        key = (mention["type"], text_key(mention["text"]))
        if key not in lookup:
            entity = {
                "id": f"e{len(entities)}", "text": mention["text"],
                "type": mention["type"], "mentions": 0,
                "confidence": None, "spans": [], "attributes": {},
            }
            lookup[key] = entity
            entities.append(entity)
        entity = lookup[key]
        entity["mentions"] += 1
        if mention["confidence"] is not None:
            entity["confidence"] = max(entity["confidence"] or 0, mention["confidence"])
        if isinstance(mention["start"], int) and isinstance(mention["end"], int):
            entity["spans"].append({"chunk": chunk_name, "start": mention["start"],
                                    "end": mention["end"]})
    return entities


def extract_relations(gliner, text, types, threshold):
    if not types:
        return []
    result = gliner.extract_relations(text, types, threshold=threshold,
                                      include_confidence=True)
    raw = result.get("relation_extraction", result.get("relations", {}))
    if not isinstance(raw, dict):
        return []
    relations = []
    for label, entries in raw.items():
        label = label.split(":", 1)[0].strip()
        if label not in types:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            confidence = None
            if isinstance(entry, dict):
                head, tail = entry.get("head"), entry.get("tail")
                confidence = entry.get("confidence")
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                head, tail = entry[:2]
                confidence = entry[2] if len(entry) > 2 else None
            else:
                continue
            head = head.get("text") if isinstance(head, dict) else head
            tail = tail.get("text") if isinstance(tail, dict) else tail
            if (not isinstance(head, str) or not isinstance(tail, str)
                    or (confidence is not None and confidence < threshold)):
                continue
            relations.append({"head": clean_text(head), "relation": label,
                              "tail": clean_text(tail), "confidence": confidence})
    return relations


def normalize_relations(raw, entities, chunk_name):
    # Never invent a node for a relation endpoint. If the same surface name
    # has multiple entity types in one chunk, refuse the ambiguous link.
    by_name = {}
    for entity in entities:
        by_name.setdefault(text_key(entity["text"]), []).append(entity)
    valid, seen = [], set()
    for relation in raw:
        heads = by_name.get(text_key(relation["head"]), [])
        tails = by_name.get(text_key(relation["tail"]), [])
        if len(heads) != 1 or len(tails) != 1 or heads[0]["id"] == tails[0]["id"]:
            continue
        key = (heads[0]["id"], relation["relation"], tails[0]["id"])
        if key in seen:
            continue
        seen.add(key)
        valid.append({
            "head_id": heads[0]["id"], "head": heads[0]["text"],
            "relation": relation["relation"],
            "tail_id": tails[0]["id"], "tail": tails[0]["text"],
            "confidence": relation["confidence"], "source_chunks": [chunk_name],
        })
    return valid


def parse_json(raw):
    # NuExtract outputs are sometimes prefixed with the original prompt or
    # output marker; take the LAST output section before decoding.
    if "<|output|>" in raw:
        raw = raw.rsplit("<|output|>", 1)[-1]
    raw = raw.replace("<|end-output|>", "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def grounded(value, text):
    """Keep only verbatim/substrings found in the same source chunk."""
    if isinstance(value, str):
        value = clean_text(value)
        if not value or value.casefold() in {"verbatim-string", "number", "date", "currency"}:
            return ""
        return value if value.casefold() in clean_text(text).casefold() else ""
    if isinstance(value, list):
        return [item for v in value if (item := grounded(v, text)) not in ("", None, [])]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Number formatting may differ; be conservative.
        return value if str(value) in text else None
    return None


def extract_entity_attributes(tokenizer, model, torch, text, entity, schema,
                              max_input_tokens, max_new_tokens):
    if not schema:
        return {}
    # One entity per template helps the 0.5B NuExtract model avoid mixing the
    # attributes of similarly typed entities in the same chunk.
    template = {"attributes": schema}
    prompt = (
        "<|input|>\n### Template:\n"
        + json.dumps(template, indent=2, ensure_ascii=False)
        + "\n### Text:\nTarget entity: " + entity["text"]
        + "\nOnly extract attributes of this target from the evidence below.\n"
        + text + "\n\n<|output|>"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    n_input = inputs["input_ids"].shape[1]
    if n_input > max_input_tokens:
        say(f"[WARN] NuExtract prompt too long for {entity['text']!r} ({n_input} tokens); skipping attributes")
        return {}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False,
                                   max_new_tokens=max_new_tokens, use_cache=True)
    continuation = generated[0, n_input:]
    output = tokenizer.decode(continuation, skip_special_tokens=False)
    parsed = parse_json(output)
    if not parsed:
        say(f"[WARN] NuExtract did not return JSON for {entity['text']!r}")
        return {}
    attrs = parsed.get("attributes", parsed)
    if not isinstance(attrs, dict):
        return {}
    cleaned = {}
    for name, placeholder in schema.items():
        value = grounded(attrs.get(name), text)
        if isinstance(placeholder, list) and not isinstance(value, list):
            value = [value] if value not in ("", None) else []
        elif placeholder is None and not isinstance(value, (int, float)):
            # Preserve extracted numerical strings only if explicitly grounded.
            value = value if isinstance(value, str) and value else None
        elif isinstance(placeholder, str) and isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if value not in (None, "", []):
            cleaned[name] = value
    return cleaned


def process_chunk(name, text, ontology, gliner, tokenizer, model, torch, args):
    types = ontology["entity_types"]
    relations = ontology["relation_types"]
    schemas = ontology["attribute_schemas"]
    say(f"  [NER] Running GLiNER2 on {name} ...")
    mentions = extract_entity_mentions(gliner, text, types, args.entity_threshold)
    entities = local_entities(mentions, name)
    say(f"  [NER] {len(mentions)} mentions / {len(entities)} entities")
    say("  [RE] Running GLiNER2 relations ...")
    raw = extract_relations(gliner, text, relations, args.relation_threshold)
    valid = normalize_relations(raw, entities, name)
    say(f"  [RE] {len(raw)} predicted / {len(valid)} linked relations")
    say(f"  [AE] Running NuExtract for {len(entities)} entities ...")
    for idx, entity in enumerate(entities, 1):
        say(f"    [{idx}/{len(entities)}] {entity['text']}")
        entity["attributes"] = extract_entity_attributes(
            tokenizer, model, torch, text, entity,
            schemas.get(entity["type"], {}),
            args.max_input_tokens, args.max_new_tokens,
        )
    return {"source_chunk": name, "entities": entities, "relationships": valid}


def merge_attributes(current, incoming):
    # Attribute values supported by different chunks are retained. Conflicts
    # are exposed as lists; the generator never invents a preferred value.
    for name, value in incoming.items():
        if value in (None, "", []):
            continue
        if name not in current:
            current[name] = value
            continue
        existing = current[name]
        if existing == value:
            continue
        a = existing if isinstance(existing, list) else [existing]
        b = value if isinstance(value, list) else [value]
        for item in b:
            if item not in a:
                a.append(item)
        current[name] = a


def merge_chunks(chunk_results, ontology_file):
    entities, relationships, by_entity, by_relation = [], [], {}, {}
    for result in chunk_results:
        local_to_global = {}
        chunk = result["source_chunk"]
        for entity in result["entities"]:
            key = (entity["type"], text_key(entity["text"]))
            if key not in by_entity:
                merged = {
                    "id": f"e{len(entities)}", "text": entity["text"],
                    "type": entity["type"], "mentions": 0,
                    "confidence": None, "source_chunks": [],
                    "spans": [], "attributes": {},
                }
                by_entity[key] = merged
                entities.append(merged)
            merged = by_entity[key]
            local_to_global[entity["id"]] = merged["id"]
            merged["mentions"] += entity["mentions"]
            if entity["confidence"] is not None:
                merged["confidence"] = max(merged["confidence"] or 0,
                                            entity["confidence"])
            if chunk not in merged["source_chunks"]:
                merged["source_chunks"].append(chunk)
            merged["spans"].extend(entity["spans"])
            merge_attributes(merged["attributes"], entity.get("attributes", {}))

        for relation in result["relationships"]:
            head = local_to_global.get(relation["head_id"])
            tail = local_to_global.get(relation["tail_id"])
            if head is None or tail is None or head == tail:
                continue
            key = (head, relation["relation"], tail)
            if key not in by_relation:
                merged = {
                    "head_id": head, "head": entities[int(head[1:])]["text"],
                    "relation": relation["relation"],
                    "tail_id": tail, "tail": entities[int(tail[1:])]["text"],
                    "confidence": relation["confidence"], "source_chunks": [],
                }
                by_relation[key] = merged
                relationships.append(merged)
            merged = by_relation[key]
            if relation["confidence"] is not None:
                merged["confidence"] = max(merged["confidence"] or 0,
                                            relation["confidence"])
            if chunk not in merged["source_chunks"]:
                merged["source_chunks"].append(chunk)
    return {
        "ontology_file": str(ontology_file),
        "source_chunks": [r["source_chunk"] for r in chunk_results],
        "entities": entities, "relationships": relationships,
    }


def export_graphml(path, graph):
    import networkx as nx
    g = nx.MultiDiGraph()
    for entity in graph["entities"]:
        g.add_node(
            entity["id"], text=entity["text"], type=entity["type"],
            mentions=entity["mentions"],
            confidence=entity["confidence"] if entity["confidence"] is not None else -1.0,
            source_chunks=json.dumps(entity["source_chunks"], ensure_ascii=False),
            attributes=json.dumps(entity["attributes"], ensure_ascii=False),
            spans=json.dumps(entity["spans"], ensure_ascii=False),
        )
    for rel in graph["relationships"]:
        g.add_edge(rel["head_id"], rel["tail_id"], relation=rel["relation"],
                   confidence=rel["confidence"] if rel["confidence"] is not None else -1.0,
                   source_chunks=json.dumps(rel["source_chunks"], ensure_ascii=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="chunks")
    parser.add_argument("--ontology", default="ontology.json")
    parser.add_argument("--output", default="knowledge_graph.json")
    parser.add_argument("--graphml", default="knowledge_graph.graphml")
    parser.add_argument("--cache-dir", default="kg_chunks")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--entity-threshold", type=float, default=ENTITY_THRESHOLD)
    parser.add_argument("--relation-threshold", type=float, default=RELATION_THRESHOLD)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--refresh", action="store_true", help="Ignore per-chunk KG checkpoints")
    args = parser.parse_args(argv)
    ontology = load_ontology(args.ontology)
    folder = Path(args.chunks)
    files = sorted(folder.glob("*.txt"))
    if not files:
        parser.error(f"No .txt files found in {folder.resolve()}")
    say(f"[INFO] Ontology: {len(ontology['entity_types'])} entity types, "
        f"{len(ontology['relation_types'])} relation types")
    say(f"[INFO] Found {len(files)} chunks")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    schema_hash = sha(json.dumps({k: ontology[k] for k in
        ("entity_types", "relation_types", "attribute_schemas")}, sort_keys=True))
    models = None  # Don't download models if every chunk is already cached.
    results = []
    for idx, file in enumerate(files, 1):
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            say(f"[{idx}/{len(files)}] SKIP empty: {file.name}")
            continue
        cache_path = cache_dir / (file.stem + "_" + sha(str(file.resolve()))[:8] + ".json")
        fingerprint = sha(json.dumps({"version": CACHE_VERSION, "text": text,
            "schema_hash": schema_hash, "gliner": GLINER_MODEL,
            "nuextract": NUEXTRACT_MODEL, "entity_threshold": args.entity_threshold,
            "relation_threshold": args.relation_threshold,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens}, sort_keys=True))
        if cache_path.exists() and not args.refresh:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                say(f"[{idx}/{len(files)}] CACHED {file.name}")
                results.append(cached["result"])
                continue
        if models is None:
            models = load_models(args.device)
        say(f"[{idx}/{len(files)}] Processing {file.name}")
        gliner, tokenizer, model, torch, _ = models
        result = process_chunk(file.name, text, ontology,
                               gliner, tokenizer, model, torch, args)
        results.append(result)
        save_json(cache_path, {"fingerprint": fingerprint, "result": result})
        say(f"[{idx}/{len(files)}] Saved checkpoint {cache_path}")

    graph = merge_chunks(results, args.ontology)
    save_json(args.output, graph)
    say(f"[DONE] JSON: {Path(args.output).resolve()}")
    export_graphml(args.graphml, graph)
    say(f"[DONE] GraphML: {Path(args.graphml).resolve()}")
    say(f"[DONE] {len(graph['entities'])} entities, {len(graph['relationships'])} relationships")


if __name__ == "__main__":
    main()
