"""
src/llm/policy_grounding.py
────────────────────────────
Deterministically map a credit decision to the governing clauses of
CREDIT_POLICY.md, so credit memos can cite the exact policy behind each
decision — e.g. a DTI decline that quotes §3.3 Gate 2 — Hard Policy Rules.

WHY DETERMINISTIC (NOT RETRIEVAL)?
──────────────────────────────────
For a memo that must be defensible to an auditor, the same decision must
always cite the same clause. A fixed decision→section map gives exactly
that: reproducible, explainable, and independent of any model. This
module is self-contained — it parses the policy document and needs no
embeddings, vector store, or LLM.

USAGE
─────
    from src.llm.policy_grounding import ground_decision, format_policy_basis

    cites = ground_decision("DECLINE_POLICY", product_type=0)
    #  → [{"section": "§3.3 Gate 2 — Hard Policy Rules", "excerpt": "..."}]
    block = format_policy_basis(cites)   # ready to drop into a memo/prompt
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import config

POLICY_PATH = config.ROOT / "CREDIT_POLICY.md"

# Decision code → ordered list of policy section numbers to cite.
# These follow CREDIT_POLICY.md §3 "Hierarchical Decision Engine".
_DECISION_SECTIONS = {
    "DECLINE_FRAUD":  ["3.2"],          # Gate 1 — Fraud Screening
    "DECLINE_POLICY": ["3.3"],          # Gate 2 — Hard Policy Rules
    "DECLINE_CREDIT": ["3.4", "3.5"],   # Gate 3 — Credit Model / Gate 4 — Bands
    "REFER":          ["3.5", "3.6"],   # Gate 4 — Bands / Gate 5 — Human Authority
    "APPROVE":        ["3.5"],          # Gate 4 — Decision Bands
}

# Product-specific requirement section, added as a secondary citation.
_PRODUCT_SECTIONS = {0: "4.1", 1: "4.2"}

# Max citations to keep a memo tight.
_MAX_CITATIONS = 2

_HEADING_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\s+(.*)$")


def _parse_sections(text: str) -> dict:
    """Parse the policy into {section_number: {'title', 'body'}}."""
    sections: dict = {}
    cur_num: Optional[str] = None
    cur_title: str = ""
    buf: List[str] = []
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if cur_num is not None:
                sections[cur_num] = {"title": cur_title,
                                     "body": "\n".join(buf).strip()}
            cur_num, cur_title, buf = m.group(1), m.group(2).strip(), []
        elif cur_num is not None:
            buf.append(line)
    if cur_num is not None:
        sections[cur_num] = {"title": cur_title, "body": "\n".join(buf).strip()}
    return sections


def _excerpt(body: str, limit: int = 420) -> str:
    """A short, clean quote from a section body, cut on a clause boundary."""
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{2,}", "\n", body).strip()
    if len(body) <= limit:
        return body
    snippet = body[:limit]
    cut = max(snippet.rfind(". "), snippet.rfind("\n"))
    if cut > 120:
        snippet = snippet[:cut + 1]
    return snippet.strip().rstrip(".") + " …"


def ground_decision(decision: str,
                    product_type: int = 0,
                    policy_failures: Optional[List[str]] = None,
                    policy_path=None) -> List[dict]:
    """Return the policy clauses that govern a given decision.

    Args:
        decision: Decision code (APPROVE / REFER / DECLINE_* ).
        product_type: 0 = unsecured, 1 = HELOC (for the product-specific clause).
        policy_failures: Unused today; accepted so callers can pass the
            decision's failure list without breaking if the mapping is
            later refined per-rule.
        policy_path: Override the policy file location (defaults to
            CREDIT_POLICY.md at the project root).

    Returns:
        A list of up to two {"section", "excerpt"} dicts. Empty if the
        policy file is missing or the decision is unrecognised.
    """
    path = Path(policy_path or POLICY_PATH)
    if not path.exists():
        return []
    sections = _parse_sections(
        path.read_text(encoding="utf-8", errors="replace"))

    nums = list(_DECISION_SECTIONS.get(decision, []))
    # Add the product-specific requirement as a secondary citation.
    pnum = _PRODUCT_SECTIONS.get(int(product_type))
    if pnum and pnum not in nums:
        nums.append(pnum)

    out: List[dict] = []
    for num in nums:
        if num in sections and len(out) < _MAX_CITATIONS:
            sec = sections[num]
            out.append({"section": f"§{num} {sec['title']}",
                        "excerpt": _excerpt(sec["body"])})
    return out


def format_policy_basis(citations: List[dict]) -> str:
    """Render citations as a 'POLICY BASIS' block for a memo or prompt."""
    if not citations:
        return ""
    lines = ["POLICY BASIS (cite these clauses):"]
    for c in citations:
        lines.append(f"[{c['section']}]\n{c['excerpt']}")
    return "\n".join(lines)
