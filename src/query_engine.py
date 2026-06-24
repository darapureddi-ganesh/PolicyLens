"""Module 4 -- Query Engine.

Resolves a natural-language query to a graph-traversal answer: no LLM, no
embeddings. Pipeline: spaCy tokenisation/lemmatisation assists keyword/phrase
matching against the fixed entity ontology; the matched entities are then
located in the NetworkX knowledge graph and their incoming/outgoing typed
edges are inspected to produce an auditable answer with exact source clause
and traversal path.
"""
from dataclasses import dataclass, field

import spacy

from entity_extractor import ENTITY_ONTOLOGY, extract_entities_from_text
from graph_builder import entity_node_id, clause_node_id

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = spacy.blank("en")
    return _nlp


@dataclass
class SourceClause:
    citation: str
    text: str
    relation: str
    clause_id: str


@dataclass
class QueryResult:
    query: str
    intent: str
    status: str                       # YES / NO / PARTIAL / LIST / UNKNOWN
    answer: str
    sources: list = field(default_factory=list)      # list[SourceClause]
    graph_path: list = field(default_factory=list)    # list[str], human-readable hops
    matched_entities: list = field(default_factory=list)


def _lemmatised(text: str) -> str:
    doc = _get_nlp()(text)
    return " ".join(tok.lemma_.lower() for tok in doc)


def _match_query_entities(query: str) -> list:
    candidates = extract_entities_from_text(query)
    lemma_text = _lemmatised(query)
    if lemma_text != query.lower():
        for key in extract_entities_from_text(lemma_text):
            if key not in candidates:
                candidates.append(key)
    return candidates


def _clauses_via_relation(graph, entity_key: str, relation: str) -> list:
    node = entity_node_id(entity_key)
    out = []
    if node not in graph:
        return out
    for pred in graph.predecessors(node):
        data = graph.get_edge_data(pred, node)
        if data.get("relation") == relation:
            cdata = graph.nodes[pred]
            out.append(SourceClause(
                citation=cdata["citation"],
                text=cdata["text"],
                relation=relation,
                clause_id=cdata["clause_id"],
            ))
    return out


def _waived_by(graph, entity_key: str) -> list:
    node = entity_node_id(entity_key)
    if node not in graph:
        return []
    return [
        graph.nodes[v]["label"]
        for _, v, d in graph.out_edges(node, data=True)
        if d.get("relation") == "WAIVED_BY"
    ]


def _list_by_relation(graph, relation: str) -> list:
    seen = set()
    out = []
    for u, v, d in graph.edges(data=True):
        if d.get("relation") == relation and u not in seen:
            seen.add(u)
            cdata = graph.nodes[u]
            out.append(SourceClause(
                citation=cdata["citation"],
                text=cdata["text"],
                relation=relation,
                clause_id=cdata["clause_id"],
            ))
    out.sort(key=lambda s: tuple(int(p) for p in s.clause_id.split(".")))
    return out


_INFO_RELATIONS = ["HAS_DEDUCTIBLE", "HAS_CONDITION", "PROVIDES_ADDON"]


def _entity_lookup(graph, query: str, entity_keys: list) -> QueryResult:
    excludes, has_coverage, waived_addons, info, path = [], [], [], [], []
    for key in entity_keys:
        label = ENTITY_ONTOLOGY[key][0]
        ex = _clauses_via_relation(graph, key, "EXCLUDES")
        cov = _clauses_via_relation(graph, key, "HAS_COVERAGE")
        if ex:
            excludes.extend(ex)
            path.append(f"entity:{label} --EXCLUDES--> {', '.join(c.citation for c in ex)}")
        if cov:
            has_coverage.extend(cov)
            path.append(f"entity:{label} --HAS_COVERAGE--> {', '.join(c.citation for c in cov)}")
        for relation in _INFO_RELATIONS:
            hits = _clauses_via_relation(graph, key, relation)
            if hits:
                info.extend(hits)
                path.append(f"entity:{label} --{relation}--> {', '.join(c.citation for c in hits)}")
        addons = _waived_by(graph, key)
        if addons:
            waived_addons.extend(addons)
            path.append(f"entity:{label} --WAIVED_BY--> {', '.join(addons)}")

    if excludes and waived_addons:
        status = "PARTIAL"
        clause = excludes[0]
        addon_str = ", ".join(sorted(set(waived_addons)))
        answer = (
            f"PARTIAL -- {clause.citation}: {clause.text} "
            f"This exclusion is waived if the following add-on is purchased: {addon_str}."
        )
        sources = excludes
    elif excludes:
        status = "NO"
        clause = excludes[0]
        answer = f"NO -- {clause.citation}: {clause.text}"
        sources = excludes
    elif has_coverage:
        status = "YES"
        clause = has_coverage[0]
        answer = f"YES -- {clause.citation}: {clause.text}"
        sources = has_coverage
    elif info:
        status = "INFO"
        clause = info[0]
        answer = f"INFO -- {clause.citation}: {clause.text}"
        sources = info
    else:
        status = "UNKNOWN"
        answer = "This policy does not contain a clause that directly addresses this query."
        sources = []

    return QueryResult(
        query=query, intent="ENTITY_LOOKUP", status=status, answer=answer,
        sources=sources, graph_path=path, matched_entities=entity_keys,
    )


def answer_query(graph, query: str) -> QueryResult:
    lowered = query.lower()
    entity_keys = _match_query_entities(query)

    if "exclu" in lowered:
        sources = _list_by_relation(graph, "EXCLUDES")
        path = [f"clause:{s.clause_id} --EXCLUDES--> entity" for s in sources]
        lines = [f"{s.citation}: {s.text}" for s in sources]
        answer = "Exclusions:\n" + "\n".join(f"- {line}" for line in lines) if lines else "No exclusions found."
        return QueryResult(query=query, intent="LIST_EXCLUSIONS", status="LIST",
                            answer=answer, sources=sources, graph_path=path,
                            matched_entities=entity_keys)

    if "deductible" in lowered:
        sources = _list_by_relation(graph, "HAS_DEDUCTIBLE")
        path = [f"clause:{s.clause_id} --HAS_DEDUCTIBLE--> entity" for s in sources]
        lines = [f"{s.citation}: {s.text}" for s in sources]
        answer = "Deductibles:\n" + "\n".join(f"- {line}" for line in lines) if lines else "No deductible clauses found."
        return QueryResult(query=query, intent="LIST_DEDUCTIBLES", status="LIST",
                            answer=answer, sources=sources, graph_path=path,
                            matched_entities=entity_keys)

    if entity_keys:
        return _entity_lookup(graph, query, entity_keys)

    return QueryResult(
        query=query, intent="UNRESOLVED", status="UNKNOWN",
        answer="No matching entity was found in this policy's knowledge graph for that query.",
        sources=[], graph_path=[], matched_entities=[],
    )


if __name__ == "__main__":
    from parser import parse_policy
    from entity_extractor import extract
    from graph_builder import build_graph

    policy = parse_policy("data/sample_motor_policy.pdf")
    graph = build_graph(extract(policy))

    for q in ["Is theft covered?", "What is excluded?", "What is the deductible?",
              "Is engine flood damage covered?"]:
        result = answer_query(graph, q)
        print("Q:", q)
        print("A:", result.answer)
        print("Path:", result.graph_path)
        print()
