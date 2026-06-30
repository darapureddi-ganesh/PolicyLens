"""Dev tool: report policy clauses that have a recognized clause type
(COVERAGE/EXCLUSION/DEDUCTIBLE/ADDON/CONDITION) but matched no entity in
ENTITY_ONTOLOGY. Use this to find which phrase patterns to add to
src/entity_extractor.py when feeding in a new insurer's policy template.

Usage:
    python tools/ontology_gap_report.py path/to/policy.pdf
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse_policy  # noqa: E402
from entity_extractor import extract  # noqa: E402

RELEVANT_TYPES = {"COVERAGE", "SCOPE", "EXCLUSION", "DEDUCTIBLE", "ADDON", "CONDITION"}


def gap_report(pdf_path: str) -> None:
    policy = parse_policy(pdf_path)
    extracted = extract(policy)

    relevant = [ec for ec in extracted if ec.clause_type in RELEVANT_TYPES]
    gaps = [ec for ec in relevant if not ec.entities]
    covered = len(relevant) - len(gaps)
    coverage_pct = 100 * covered / len(relevant) if relevant else 0

    print(f"Entity coverage: {covered}/{len(relevant)} relevant clauses matched "
          f"an ontology entity ({coverage_pct:.0f}%)\n")

    if not gaps:
        print("No gaps -- every relevant clause matched at least one ontology entity.")
        return

    by_type = defaultdict(list)
    for ec in gaps:
        by_type[ec.clause_type].append(ec)

    for clause_type in sorted(by_type):
        clauses = by_type[clause_type]
        print(f"=== {clause_type} ({len(clauses)} unmatched) ===")
        for ec in clauses:
            clause = policy.get(ec.clause_id)
            citation = clause.citation if clause else f"Clause {ec.clause_id}"
            print(f"  {citation}")
            print(f"    {ec.text}\n")


def main():
    argparser = argparse.ArgumentParser(
        description="Report policy clauses with no matched ontology entity, "
                     "to help extend ENTITY_ONTOLOGY for new policy wording."
    )
    argparser.add_argument("pdf_path", help="Path to a text-layer policy PDF")
    args = argparser.parse_args()
    gap_report(args.pdf_path)


if __name__ == "__main__":
    main()
