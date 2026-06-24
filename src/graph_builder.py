"""Module 3 -- Graph Builder.

Builds a NetworkX DiGraph from extracted clauses. Clause nodes connect to
entity nodes with a typed relation derived from the clause's section type
(HAS_COVERAGE, EXCLUDES, HAS_DEDUCTIBLE, PROVIDES_ADDON, HAS_CONDITION,
MENTIONS). Exclusions that are waived by a purchasable add-on (e.g. engine
damage from flooding, waived by Engine Protect) get an extra entity-to-entity
WAIVED_BY edge so the query engine can answer "partial coverage" questions.
"""
import networkx as nx

from entity_extractor import ENTITY_ONTOLOGY

CLAUSE_TYPE_RELATION = {
    "COVERAGE": "HAS_COVERAGE",
    "SCOPE": "HAS_COVERAGE",
    "EXCLUSION": "EXCLUDES",
    "DEDUCTIBLE": "HAS_DEDUCTIBLE",
    "ADDON": "PROVIDES_ADDON",
    "CONDITION": "HAS_CONDITION",
}
DEFAULT_RELATION = "MENTIONS"

CLAUSE_NODE_PREFIX = "clause:"
ENTITY_NODE_PREFIX = "entity:"


def clause_node_id(clause_id: str) -> str:
    return f"{CLAUSE_NODE_PREFIX}{clause_id}"


def entity_node_id(entity_key: str) -> str:
    return f"{ENTITY_NODE_PREFIX}{entity_key}"


def build_graph(extracted_clauses) -> nx.DiGraph:
    g = nx.DiGraph()

    # entity nodes (fixed ontology, always present so queries can resolve even
    # if a given policy doesn't mention every concept)
    for key, (label, category, _) in ENTITY_ONTOLOGY.items():
        g.add_node(entity_node_id(key), node_type="entity", key=key, label=label, category=category)

    for ec in extracted_clauses:
        cnode = clause_node_id(ec.clause_id)
        g.add_node(
            cnode,
            node_type="clause",
            clause_id=ec.clause_id,
            section_number=ec.section_number,
            section_title=ec.section_title,
            clause_type=ec.clause_type,
            text=ec.text,
            amounts=ec.amounts,
            citation=f"Section {ec.section_number}, Clause {ec.clause_id.split('.')[1]}",
        )
        relation = CLAUSE_TYPE_RELATION.get(ec.clause_type, DEFAULT_RELATION)
        for entity_key in ec.entities:
            g.add_edge(cnode, entity_node_id(entity_key), relation=relation)

    # cross-link: exclusions waived by an add-on mentioned in the same clause
    for ec in extracted_clauses:
        if ec.clause_type != "EXCLUSION":
            continue
        addon_keys = [k for k in ec.entities if ENTITY_ONTOLOGY[k][1] == "addon"]
        cause_keys = [k for k in ec.entities if ENTITY_ONTOLOGY[k][1] in ("peril", "exclusion_cause")]
        for cause_key in cause_keys:
            for addon_key in addon_keys:
                g.add_edge(
                    entity_node_id(cause_key),
                    entity_node_id(addon_key),
                    relation="WAIVED_BY",
                    waiving_clause=ec.clause_id,
                )

    return g


if __name__ == "__main__":
    from parser import parse_policy
    from entity_extractor import extract

    policy = parse_policy("data/sample_motor_policy.pdf")
    extracted = extract(policy)
    graph = build_graph(extracted)
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    for u, v, data in graph.edges(data=True):
        print(u, "--", data["relation"], "-->", v)
