"""
src/llm/ollama_client.py
─────────────────────────
Ollama LLM integration for credit memo and adverse action generation.

WHY LOCAL LLM (OLLAMA)?
────────────────────────
PIPEDA prohibits transmitting borrower PII to third-party cloud APIs
without explicit consent and a data processing agreement.

By running Llama 3 locally via Ollama:
  - No borrower data ever leaves the lender's infrastructure
  - Full PIPEDA compliance by design
  - No per-token cloud API costs at scale
  - Consistent latency (no network dependency)
  - Auditable — the model weights are on-premise

SETUP (one-time):
  1. Install Ollama: https://ollama.ai
  2. Pull model: ollama pull llama3
  3. Start server: ollama serve  (runs on localhost:11434)

The client gracefully falls back to a template-based response
if Ollama is not running, so the platform works in demo mode
without requiring a local GPU.

OUTPUTS GENERATED:
  1. Credit Memo        — structured underwriting narrative
  2. Adverse Action Letter — ECOA/Regulation B compliant decline letter
  3. Risk Summary       — brief risk tier and key factors summary
"""

import httpx
import json
import logging
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
TIMEOUT_SEC  = 30


def _call_ollama(prompt: str,
                 system: str = "",
                 temperature: float = 0.3) -> str:
    """
    Call Ollama local LLM API.

    Args:
        prompt: User prompt.
        system: System instruction.
        temperature: 0.0=deterministic, 1.0=creative. 0.3 for credit.

    Returns:
        Model response string. Falls back to template if unavailable.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 600,
            "top_p": 0.9,
        }
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

    except httpx.ConnectError:
        log.warning("Ollama not running — using template fallback")
        return None
    except Exception as e:
        log.warning(f"Ollama call failed ({e}) — using template fallback")
        return None


def generate_credit_memo(applicant: dict,
                          pd_score: float,
                          credit_score: int,
                          risk_tier: str,
                          lgd: float,
                          ead: float,
                          expected_loss: float,
                          shap_factors: list,
                          decision: str,
                          product_type: int) -> str:
    """
    Generate a structured credit memo using Ollama LLM.

    Args:
        applicant: Dict of applicant features.
        pd_score: Probability of default (0-1).
        credit_score: PDO calibrated score (300-850).
        risk_tier: A/B/C/D/E.
        lgd: Loss Given Default.
        ead: Exposure at Default.
        expected_loss: EL in dollars.
        shap_factors: List of top SHAP factor dicts.
        decision: "APPROVE" or "DECLINE".
        product_type: 0=unsecured, 1=secured.

    Returns:
        Formatted credit memo string.
    """
    product_name = "Home Equity Line of Credit" if product_type == 1 \
                   else "Personal Instalment Loan"
    el_rate      = pd_score * lgd
    top_factors  = shap_factors[:3] if shap_factors else []

    factors_text = "\n".join([
        f"  - {f['feature'].replace('_',' ').title()}: "
        f"{f['value']} (impact: {'↑ risk' if f['shap_impact'] > 0 else '↓ risk'})"
        for f in top_factors
    ])

    system = (
        "You are a senior credit analyst at a Canadian financial institution. "
        "Write professional, concise credit memos in formal banking language. "
        "Be factual, reference specific data points, and keep to 250 words. "
        "Never fabricate data. Use only the information provided."
    )

    prompt = f"""
Write a credit memo for the following application:

PRODUCT: {product_name}
DECISION: {decision}
CREDIT SCORE: {credit_score} (Risk Tier {risk_tier})
PROBABILITY OF DEFAULT: {pd_score:.1%}
LOAN AMOUNT REQUESTED: ${applicant.get('loan_amount', 0):,.0f}
ANNUAL INCOME: ${applicant.get('annual_income', 0):,.0f}
DEBT-TO-INCOME RATIO: {applicant.get('dti', 0):.1f}%
EMPLOYMENT TENURE: {applicant.get('employment_length_years', 0):.0f} years
LTV RATIO: {(applicant.get('ltv_ratio') or 0):.0%} (secured only)

KEY RISK FACTORS (from SHAP model explanation):
{factors_text}

EXPECTED LOSS METRICS:
  Loss Given Default: {lgd:.1%}
  Exposure at Default: ${ead:,.0f}
  Expected Loss: ${expected_loss:,.0f} ({el_rate:.1%} of EAD)

Structure the memo as:
1. Application Summary (2 sentences)
2. Credit Assessment (3-4 sentences covering key strengths/weaknesses)
3. Risk Metrics (1-2 sentences on EL and model output)
4. Decision and Rationale (2-3 sentences)
"""

    response = _call_ollama(prompt, system, temperature=0.3)

    if response:
        return response
    else:
        return _credit_memo_template(
            applicant, pd_score, credit_score, risk_tier,
            lgd, ead, expected_loss, shap_factors, decision,
            product_name, el_rate
        )


def generate_adverse_action_letter(applicant_name: str,
                                    loan_amount: float,
                                    product_type: int,
                                    pd_score: float,
                                    adverse_reasons: list,
                                    lender_name: str = "Canadian Lender") -> str:
    """
    Generate ECOA/Regulation B compliant adverse action letter.

    Args:
        applicant_name: Borrower name (first name only for privacy).
        loan_amount: Requested loan amount.
        product_type: 0=unsecured, 1=secured.
        pd_score: Model PD (not disclosed to applicant).
        adverse_reasons: Top 3 SHAP-based reason codes (plain language).
        lender_name: Institution name.

    Returns:
        Formatted adverse action letter.
    """
    product_name = "Home Equity Line of Credit" if product_type == 1 \
                   else "Personal Instalment Loan"
    reasons_text = "\n".join([
        f"  {i+1}. {r}" for i, r in enumerate(adverse_reasons[:3])
    ])

    system = (
        "You are a compliance officer at a Canadian financial institution. "
        "Write a professional, empathetic adverse action letter that complies "
        "with ECOA and Canadian consumer protection requirements. "
        "Do NOT mention the probability of default or model scores. "
        "Keep to 200 words. Be respectful and provide clear reasons."
    )

    prompt = f"""
Write an adverse action letter for a declined credit application.

APPLICANT: {applicant_name}
PRODUCT: {product_name}
LOAN AMOUNT: ${loan_amount:,.0f}
LENDER: {lender_name}

PRIMARY REASONS FOR DECLINE (must be listed verbatim):
{reasons_text}

The letter must:
- Thank the applicant for their application
- Clearly state the credit decision
- List the specific reasons using the exact reasons above
- Note their right to request a free credit report
- Provide contact information for questions
- Be professional and respectful in tone
- NOT mention specific model scores or probabilities
"""

    response = _call_ollama(prompt, system, temperature=0.2)

    if response:
        return response
    else:
        return _adverse_action_template(
            applicant_name, loan_amount, product_name,
            adverse_reasons, lender_name
        )


def generate_risk_summary(pd_score: float,
                           credit_score: int,
                           risk_tier: str,
                           decision: str,
                           top_factor: str) -> str:
    """
    Generate a brief 2-sentence risk summary for the UI header.

    Args:
        pd_score: Probability of default.
        credit_score: PDO credit score.
        risk_tier: A-E tier label.
        decision: APPROVE or DECLINE.
        top_factor: Primary SHAP factor name.

    Returns:
        Two-sentence plain-language risk summary.
    """
    system = (
        "You are a credit analyst. Write exactly two clear, plain-English "
        "sentences summarising a credit decision. No jargon. "
        "First sentence: the decision and main reason. "
        "Second sentence: the primary risk driver."
    )

    prompt = f"""
Summarise this credit decision in exactly two sentences:
- Decision: {decision}
- Credit Score: {credit_score} (Tier {risk_tier})
- Probability of Default: {pd_score:.1%}
- Primary Risk Driver: {top_factor.replace('_', ' ')}
"""

    response = _call_ollama(prompt, system, temperature=0.2)

    if response and len(response) > 20:
        return response
    else:
        decision_text = "approved" if decision == "APPROVE" else "declined"
        return (
            f"This application has been {decision_text} based on a credit "
            f"score of {credit_score} (Risk Tier {risk_tier}) with an "
            f"estimated default probability of {pd_score:.1%}. "
            f"The primary driver of the risk assessment is "
            f"{top_factor.replace('_', ' ').lower()}, which has the largest "
            f"influence on the model's decision."
        )


# ── Template fallbacks (used when Ollama not running) ─────────────

def _credit_memo_template(applicant, pd_score, credit_score, risk_tier,
                           lgd, ead, expected_loss, shap_factors,
                           decision, product_name, el_rate) -> str:
    """Structured template credit memo — Ollama fallback."""
    is_heloc = "equity" in product_name.lower() or "heloc" in product_name.lower()

    # Filter out loan_term_months for HELOC — it is a product proxy signal,
    # not a meaningful credit quality factor for secured lending.
    filtered_factors = [
        f for f in shap_factors
        if not (is_heloc and "loan_term" in f.get("feature", "").lower())
    ] if is_heloc else shap_factors

    top = filtered_factors[0] if filtered_factors else (
        shap_factors[0] if shap_factors else {"feature": "credit_score"}
    )
    primary = top["feature"].replace("_", " ").title()

    strengths, weaknesses = [], []
    for f in filtered_factors[:4]:
        if f.get("shap_impact", 0) < 0:
            strengths.append(f["feature"].replace("_", " "))
        else:
            weaknesses.append(f["feature"].replace("_", " "))

    str_text = ", ".join(strengths[:2]) if strengths else "credit profile"
    wk_text  = ", ".join(weaknesses[:2]) if weaknesses else "risk factors"

    return f"""CREDIT MEMO — {product_name.upper()}

APPLICATION SUMMARY
The applicant has requested a {product_name} of \
${applicant.get('loan_amount', 0):,.0f} with an annual income of \
${applicant.get('annual_income', 0):,.0f} and \
{applicant.get('employment_length_years', 0):.0f} years of employment tenure.
The application has been reviewed using the institution's internal credit \
risk model.

CREDIT ASSESSMENT
The applicant's credit score of {credit_score} places them in Risk Tier \
{risk_tier} with an estimated probability of default of {pd_score:.1%}. \
Key strengths identified include {str_text}. \
Primary risk concerns relate to {wk_text}. \
The debt-to-income ratio of {applicant.get('dti', 0):.1f}% \
{"is within acceptable guidelines" if applicant.get('dti', 50) < 40 \
 else "exceeds preferred thresholds"}.

RISK METRICS
The model estimates a Loss Given Default of {lgd:.1%} and an Exposure at \
Default of ${ead:,.0f}, resulting in an Expected Loss of \
${expected_loss:,.0f} ({el_rate:.1%} of EAD). \
The primary SHAP driver is {primary}, consistent with industry benchmarks \
for this risk tier.

DECISION: {decision}
{"Application meets minimum credit criteria for approval at the requested amount." \
 if decision == "APPROVE" else \
 f"Application does not meet minimum credit criteria. Primary decline reason: {primary}."}

[Note: Generated using template fallback — start Ollama for LLM-generated memos]
"""


def _adverse_action_template(applicant_name, loan_amount, product_name,
                               adverse_reasons, lender_name) -> str:
    """Structured template adverse action letter — Ollama fallback."""
    from datetime import date
    today = date.today().strftime("%B %d, %Y")
    reasons_formatted = "\n".join(
        [f"    {i+1}. {r}" for i, r in enumerate(adverse_reasons[:3])]
    )

    return f"""{today}

Dear {applicant_name},

Thank you for your recent application for a {product_name} \
in the amount of ${loan_amount:,.0f} with {lender_name}.

After careful review of your application, we regret to inform you that \
we are unable to approve your request at this time. Our decision was \
based on the following principal reasons:

{reasons_formatted}

You have the right to obtain a free copy of your credit report from the \
credit reporting agency used in our decision-making process. You may \
request this report within 60 days of receiving this notice.

If you believe any information in your credit report is inaccurate, you \
have the right to dispute it directly with the credit reporting agency.

Should your financial circumstances change, we encourage you to reapply \
in the future. If you have any questions regarding this decision, \
please contact our Customer Service team.

Sincerely,

Credit Risk & Underwriting
{lender_name}

[Note: Generated using template fallback — start Ollama for LLM-generated letters]
"""


def check_ollama_status() -> dict:
    """Check if Ollama is running and the model is available."""
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = [m["name"] for m in
                          response.json().get("models", [])]
                llama3_available = any("llama3" in m for m in models)
                return {
                    "ollama_running":    True,
                    "llama3_available":  llama3_available,
                    "available_models":  models,
                    "status":            "ready" if llama3_available
                                         else "llama3 not pulled",
                }
    except Exception:
        pass

    return {
        "ollama_running":   False,
        "llama3_available": False,
        "available_models": [],
        "status":           "offline — using template fallback",
    }
