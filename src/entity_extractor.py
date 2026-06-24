"""Module 2 -- Entity Extractor.

Rule-based entity extraction over parsed policy clauses. Uses a fixed domain
ontology spanning motor and health insurance concepts (perils, coverages,
exclusion causes, deductibles, add-ons, conditions) matched via spaCy
lemmatised tokens / phrase matching plus regex for monetary values. No
embeddings, no LLM.
"""
import re
from dataclasses import dataclass, field

import spacy

# entity_key -> (display label, category, [phrase patterns to match, lowercase])
MOTOR_ONTOLOGY = {
    "theft": ("Theft", "peril", ["theft", "stolen"]),
    "fire": ("Fire", "peril", ["fire", "explosion", "self-ignition", "lightning"]),
    "flood": ("Flood / Water Damage", "peril", ["flood", "flooding", "submersion", "cyclone"]),
    "accidental_damage": ("Accidental Damage", "peril", ["accidental external means", "own damage"]),
    "third_party_liability": ("Third Party Liability", "coverage", ["third party liability", "third parties"]),
    "personal_accident": ("Personal Accident Cover", "coverage", ["personal accident cover", "owner-driver"]),
    "drunk_driving": ("Drunk Driving", "exclusion_cause", ["drunk driving", "influence of alcohol", "influence of alcohol or drugs"]),
    "no_valid_licence": ("Driving Without Valid Licence", "exclusion_cause", ["valid and effective driving licence", "valid licence"]),
    "consequential_loss": ("Consequential Loss", "exclusion_cause", ["consequential loss", "mechanical or electrical breakdown"]),
    "war": ("War & Nuclear Risks", "exclusion_cause", ["war and nuclear", "invasion", "nuclear perils", "civil war"]),
    "racing": ("Racing", "exclusion_cause", ["racing", "speed testing", "rallying", "pace-making"]),
    "engine_damage": ("Engine Damage from Flooding", "exclusion_cause", ["engine damage", "hydrostatic lock", "water ingestion"]),
    "compulsory_deductible": ("Compulsory Deductible", "deductible", ["compulsory deductible"]),
    "voluntary_deductible": ("Voluntary Deductible", "deductible", ["voluntary deductible"]),
    "engine_protect": ("Engine Protect Add-on", "addon", ["engine protect"]),
    "zero_depreciation": ("Zero Depreciation Add-on", "addon", ["zero depreciation"]),
    "roadside_assistance": ("Roadside Assistance Add-on", "addon", ["roadside assistance"]),
    "notice_of_claim": ("Notice of Claim", "condition", ["notice of claim", "notify the company"]),
    "cancellation": ("Cancellation", "condition", ["cancellation", "cancel this policy"]),
}

HEALTH_ONTOLOGY = {
    "hospitalization": ("Hospitalisation Cover", "coverage", ["hospitalisation", "hospitalization", "in-patient", "inpatient"]),
    "day_care_procedure": ("Day Care Procedure", "coverage", ["day care", "daycare"]),
    "pre_post_hospitalization": ("Pre & Post Hospitalisation", "coverage", ["pre-hospitalisation", "post-hospitalisation", "pre hospitalisation", "post hospitalisation", "domiciliary hospitalisation"]),
    "ambulance_cover": ("Ambulance Cover", "coverage", ["ambulance"]),
    "maternity_cover": ("Maternity Cover", "coverage", ["maternity"]),
    "alternative_treatment": ("AYUSH / Alternative Treatment", "coverage", ["ayurved", "homeopath", "unani", "ayush"]),
    "critical_illness": ("Critical Illness Cover", "coverage", ["critical illness"]),
    "organ_donor": ("Organ Donor Cover", "coverage", ["organ donor"]),
    "free_health_checkup": ("Free Health Check-up", "coverage", ["free medical check-up", "health check-up", "health checkup"]),
    "pre_existing_disease": ("Pre-Existing Disease", "exclusion_cause", ["pre-existing condition", "pre-existing disease"]),
    "cosmetic_exclusion": ("Cosmetic Treatment Exclusion", "exclusion_cause", ["cosmetic surgery", "cosmetic treatment"]),
    "psychiatric_exclusion": ("Psychiatric Disorders", "exclusion_cause", ["psychiatric", "psychosomatic"]),
    "waiting_period": ("Waiting Period", "condition", ["waiting period"]),
    "no_claim_bonus": ("No Claim Bonus", "condition", ["no claim bonus", "ncb"]),
    "grace_period": ("Grace Period", "condition", ["grace period"]),
    "co_payment": ("Co-payment", "deductible", ["co-payment", "co payment", "copay"]),
    "room_rent_limit": ("Room Rent Limit", "deductible", ["room rent", "room, boarding"]),
    "sum_insured": ("Sum Insured", "deductible", ["sum insured"]),
}

ENTITY_ONTOLOGY = {**MOTOR_ONTOLOGY, **HEALTH_ONTOLOGY}

SECTION_TYPE_MAP = {
    "definitions": "DEFINITION",
    "scope of cover": "SCOPE",
    "coverage": "COVERAGE",
    "deductibles": "DEDUCTIBLE",
    "exclusions": "EXCLUSION",
    "add-on covers": "ADDON",
    "conditions": "CONDITION",
}

# fallback keyword cues used when the section title doesn't map directly
# (real-world PDFs rarely use the bundled motor template's exact headings)
TEXT_TYPE_CUES = [
    ("EXCLUSION", ["is excluded", "are excluded", "shall not be liable", "will not be liable",
                   "not covered under this policy", "does not cover", "we will not pay",
                   "company shall not", "benefits will not be available", "not be admissible"]),
    ("DEDUCTIBLE", ["deductible", "co-payment", "co payment", "room rent"]),
    ("ADDON", ["add-on", "addon", "rider", "optional cover"]),
    ("CONDITION", ["shall notify", "condition precedent", "grace period", "waiting period", "cancellation"]),
    ("COVERAGE", ["shall pay", "undertakes to pay", "agrees to pay", "will pay", "is covered",
                  "are covered", "shall be payable", "indemnify", "reimburse", "insurer shall"]),
]

MONEY_RE = re.compile(r"(?:Rs\.?|₹|INR)\s?[\d,]+(?:\.\d+)?")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")

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
class ExtractedClause:
    clause_id: str
    section_number: str
    section_title: str
    clause_type: str
    text: str
    entities: list = field(default_factory=list)   # list of entity_key
    amounts: list = field(default_factory=list)     # list of matched monetary/percent strings


def classify_clause_type(section_title: str, text: str = "") -> str:
    mapped = SECTION_TYPE_MAP.get(section_title.strip().lower())
    if mapped:
        return mapped
    lowered = text.lower()
    if lowered.startswith('"') or lowered.startswith("“") or " means " in lowered[:120]:
        return "DEFINITION"
    for clause_type, cues in TEXT_TYPE_CUES:
        if any(cue in lowered for cue in cues):
            return clause_type
    return "OTHER"


def extract_entities_from_text(text: str) -> list:
    """Return the list of ontology entity_keys whose phrase patterns appear in text."""
    lowered = text.lower()
    found = []
    for key, (_, _, patterns) in ENTITY_ONTOLOGY.items():
        if any(p in lowered for p in patterns):
            found.append(key)
    return found


def extract_amounts(text: str) -> list:
    return MONEY_RE.findall(text) + PERCENT_RE.findall(text)


def extract(parsed_policy) -> list:
    """Run extraction over every clause of a ParsedPolicy (see parser.py)."""
    _get_nlp()  # ensures spaCy pipeline loads once; lemmas not required for fixed-phrase match
    results = []
    for clause in parsed_policy.clauses:
        clause_type = classify_clause_type(clause.section_title, clause.text)
        entities = extract_entities_from_text(clause.text)
        amounts = extract_amounts(clause.text)
        results.append(ExtractedClause(
            clause_id=clause.clause_id,
            section_number=clause.section_number,
            section_title=clause.section_title,
            clause_type=clause_type,
            text=clause.text,
            entities=entities,
            amounts=amounts,
        ))
    return results


if __name__ == "__main__":
    from parser import parse_policy

    policy = parse_policy("data/sample_motor_policy.pdf")
    for ec in extract(policy):
        print(ec.clause_id, ec.clause_type, ec.entities, ec.amounts)
