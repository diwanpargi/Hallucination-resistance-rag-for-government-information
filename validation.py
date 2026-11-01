 
import os
import re

import config
from generation import build_context, call_gemini


def _debug(*args):
    if getattr(config, "DEBUG", False):
        print(*args)


# Check 1: Is there enough context to answer?
def enough_context(hits, top_score):
    if not hits:
        return False, "no_relevant_chunks"

    if top_score < config.MIN_BEST_SCORE:
        return False, (
            f"weak_context: best score {top_score:.2f} "
            f"is below {config.MIN_BEST_SCORE}"
        )

    total_chars = sum(len((p.payload or {}).get("text", "")) for p in hits)
    if total_chars < config.MIN_CONTEXT_CHARS:
        return False, "too_little_text"

    return True, None


# Helper: normalize PDF paths and file names
def normalize_pdf_name(source):
    """Full path -> lowercase file name, with repeated spaces collapsed."""
    source = str(source or "").strip().replace("\\", "/")
    name = os.path.basename(source).strip().lower()
    return re.sub(r"\s+", " ", name)


# Helper: normalize page numbers
def _page_text(payload):
    """Return the page number as text, or None if unavailable."""
    page = payload.get("page_number")
    try:
        return str(int(page))
    except (TypeError, ValueError):
        return None


# Check 2: Extract citations from the generated answer
def find_citations(answer, real_pdfs=()):
    """
    Finds PDF names, page numbers and section IDs in the answer.

    PDF names come from "Source: <name>.pdf" lines only, plus any retrieved
    PDF whose name appears in the answer. This avoids capturing extra words or
    list numbers in front of a file name ("1. nsp_sop.pdf").
    """
    pdfs = set()

    for match in re.finditer(r"source\s*:\s*([^\n]+?\.pdf)\b", answer, re.I):
        pdfs.add(normalize_pdf_name(match.group(1)))

    flat_answer = re.sub(r"\s+", " ", answer).lower()
    for real in real_pdfs:
        if real and real in flat_answer:
            pdfs.add(real)

    pages = set()
    for match in re.finditer(
        r"pages?[ ]*[:#]?[ ]*(\d+(?:[ ]*(?:,|and|&)[ ]*\d+)*)", answer, re.I
    ):
        pages.update(re.findall(r"\d+", match.group(1)))

    sections = {
        x.lower()
        for x in re.findall(r"[\w\-.()]+\.pdf_+\d+", answer, re.I)
    }

    return pdfs, pages, sections


# Check 3: Validate PDF, page and section citations
def validate_citations(answer, hits):
    real_pdfs = {
        normalize_pdf_name(p.payload.get("source", ""))
        for p in hits
        if p.payload and p.payload.get("source")
    }

    cited_pdfs, cited_pages, cited_sections = find_citations(answer, real_pdfs)

    _debug("RETRIEVED PDFS:", sorted(real_pdfs))
    _debug("CITED PDFS:", sorted(cited_pdfs))

    if config.REQUIRE_CITATION and not cited_pdfs:
        return False, "no_citation: the answer does not name a source PDF"

    for pdf in cited_pdfs:
        if pdf not in real_pdfs:
            return False, f"unknown_source: {pdf}"

    real_pages = {
        _page_text(p.payload)
        for p in hits
        if (not cited_pdfs
            or normalize_pdf_name(p.payload.get("source", "")) in cited_pdfs)
    }
    real_pages.discard(None)

    for page in cited_pages:
        if page not in real_pages:
            return False, f"unknown_page: {page}"

    real_sections = {
        str(p.payload.get("chunk_id", "")).strip().lower()
        for p in hits
        if p.payload.get("chunk_id")
    }
    for section in cited_sections:
        if section not in real_sections:
            return False, f"unknown_section: {section}"

    return True, None


# Check 4: Verify claims and keep schemes separate
def check_claims(gemini, answer, hits):
    prompt = f"""You are a strict fact checker for a government scholarship assistant.

Documents (each DOCUMENT is a different scholarship scheme):
{build_context(hits)}

Answer to check:
{answer}


Split the answer into separate factual statements, such as eligibility,
amounts, dates, documents and conditions.

Ignore Source, Page and Section lines; they are checked separately.

For EACH factual statement, write one line in exactly this format:
SUPPORTED: <statement>
UNSUPPORTED: <statement>
MIXED: <statement>

SUPPORTED means the document clearly supports the statement.
Equivalent wording and reasonable paraphrasing are acceptable.

UNSUPPORTED means the statement is missing from or contradicts the documents.
MIXED means facts from different documents or schemes have been combined,
or a statement contains both supported and unsupported important claims.

Pay special attention to scholarship amounts, eligibility, renewal rules
and scheme-specific conditions.

Write only these lines."""

    reply = call_gemini(gemini, prompt)
    _debug("CLAIM CHECKER RESPONSE:\n", reply)

    unsupported, mixed, total = [], [], 0

    for line in reply.splitlines():
        match = re.match(
            r"^[\s\-*\d.)]*(UNSUPPORTED|SUPPORTED|MIXED)\s*:\s*(.*)",
            line,
            re.I,
        )
        if not match:
            continue

        total += 1
        label = match.group(1).upper()
        statement = match.group(2).strip()

        if label == "UNSUPPORTED":
            unsupported.append(statement)
        elif label == "MIXED":
            mixed.append(statement)

    if total == 0:
        return False, "claims_not_checked: the checker gave no readable result"
    if unsupported:
        return False, f"unsupported_claim: {unsupported[0]}"
    if mixed:
        return False, f"mixed_schemes: {mixed[0]}"

    return True, None


# Check 5: Run all enabled validations
def validate_answer(gemini, answer, hits):
    if config.CHECK_CITATIONS:
        ok, reason = validate_citations(answer, hits)
        if not ok:
            return False, reason

    if config.CHECK_CLAIMS:
        ok, reason = check_claims(gemini, answer, hits)
        if not ok:
            return False, reason

    return True, None