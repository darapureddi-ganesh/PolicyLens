"""Module 1 -- PDF Parser.

Reads a text-layer insurance policy PDF with PyMuPDF and splits it into
Section / Clause records. Two complementary strategies run in a single pass:

1. Structured numbering: lines matching "Section N: Title" / "N.M <text>"
   (used by the bundled motor policy template).
2. Freeform fallback: for real-world policy PDFs without that numbering,
   headings are detected heuristically (short, title-case/ALL-CAPS, or a
   numbered list prefix) and body text between headings is split into
   auto-numbered paragraph "clauses" on blank-line boundaries. Lines that
   repeat near-verbatim across most pages (running headers/footers/letterhead)
   are stripped before either strategy runs.
"""
from dataclasses import dataclass, field
from collections import Counter
import re

import fitz

SECTION_RE = re.compile(r"^Section\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*[.)]?\s+[A-Z]")


@dataclass
class Clause:
    clause_id: str          # e.g. "3.2" (structured) or "4.2" (auto, per section)
    section_number: str     # e.g. "3"
    section_title: str      # e.g. "Coverage"
    text: str               # full clause text
    numbering: str = "structured"  # "structured" | "auto"

    @property
    def citation(self) -> str:
        if self.numbering == "structured":
            return f"Section {self.section_number}, Clause {self.clause_id.split('.')[1]}"
        title = f" ({self.section_title})" if self.section_title else ""
        return f"Section {self.section_number}{title}, Para {self.clause_id.split('.')[1]}"


@dataclass
class ParsedPolicy:
    source_path: str
    clauses: list = field(default_factory=list)

    def get(self, clause_id: str):
        for c in self.clauses:
            if c.clause_id == clause_id:
                return c
        return None


_PUA_RE = re.compile(r"[-]")  # private-use-area glyphs (custom bullet fonts etc.)


def _sanitize(line: str) -> str:
    line = _PUA_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _page_lines(pdf_path: str) -> list:
    """Per-page list of stripped lines, '' kept as an explicit blank-line marker."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append([_sanitize(raw) for raw in text.split("\n")])
    doc.close()
    return pages


def _strip_boilerplate(pages: list) -> list:
    """Drop lines (running headers/footers/letterhead) that repeat near-verbatim
    across a large fraction of pages."""
    if len(pages) <= 1:
        return pages
    counts = Counter()
    for lines in pages:
        seen_this_page = set(l for l in lines if l)
        for l in seen_this_page:
            counts[l] += 1
    threshold = max(2, int(len(pages) * 0.35))
    boilerplate = {l for l, c in counts.items() if c >= threshold}
    return [["" if l in boilerplate else l for l in lines] for lines in pages]


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 90 or line.endswith((".", ",", ";")):
        return False
    words = line.split()
    if not words or len(words) > 12:
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    if upper_ratio > 0.8:
        return True
    if line.istitle():
        return True
    if NUMBERED_HEADING_RE.match(line) and len(words) <= 8:
        return True
    return False


def parse_policy(pdf_path: str) -> ParsedPolicy:
    """Parse a policy PDF into a ParsedPolicy of Section/Clause records."""
    pages = _strip_boilerplate(_page_lines(pdf_path))

    clauses = []
    current_section_num = 0
    current_section_title = ""
    auto_para_counter = 0

    structured_pending = None   # (clause_id, sec_num, sec_title, [parts])
    freeform_buffer = []         # accumulating lines for the current auto paragraph

    def flush_structured():
        nonlocal structured_pending
        if structured_pending is not None:
            clause_id, sec_num, sec_title, parts = structured_pending
            clauses.append(Clause(
                clause_id=clause_id, section_number=sec_num, section_title=sec_title,
                text=" ".join(parts).strip(), numbering="structured",
            ))
            structured_pending = None

    def flush_freeform():
        nonlocal freeform_buffer, auto_para_counter
        text = " ".join(freeform_buffer).strip()
        freeform_buffer = []
        if len(text) < 20:  # skip noise fragments too short to be a meaningful clause
            return
        auto_para_counter += 1
        clauses.append(Clause(
            clause_id=f"{current_section_num}.{auto_para_counter}",
            section_number=str(current_section_num),
            section_title=current_section_title,
            text=text, numbering="auto",
        ))

    def new_section(num, title):
        nonlocal current_section_num, current_section_title, auto_para_counter
        flush_structured()
        flush_freeform()
        current_section_num = num
        current_section_title = title
        auto_para_counter = 0

    for lines in pages:
        for line in lines:
            if line == "":
                flush_freeform()
                continue

            sec_match = SECTION_RE.match(line)
            if sec_match:
                new_section(sec_match.group(1), sec_match.group(2).strip())
                continue

            clause_match = CLAUSE_RE.match(line)
            if clause_match:
                flush_freeform()
                flush_structured()
                major, minor, rest = clause_match.groups()
                sec_num = current_section_num or major
                structured_pending = (f"{major}.{minor}", str(sec_num), current_section_title, [rest.strip()])
                continue

            if structured_pending is not None:
                # continuation of the current structured clause's wrapped text
                clause_id, sec_num, sec_title, parts = structured_pending
                parts.append(line)
                continue

            if _looks_like_heading(line):
                new_section(str(int(current_section_num) + 1) if str(current_section_num).isdigit() else line, line)
                continue

            freeform_buffer.append(line)

    flush_structured()
    flush_freeform()
    return ParsedPolicy(source_path=pdf_path, clauses=clauses)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_motor_policy.pdf"
    policy = parse_policy(path)
    print(f"{len(policy.clauses)} clauses found")
    for c in policy.clauses:
        print(f"[{c.numbering}]", c.citation, "->", c.text[:90])
