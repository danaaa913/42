"""Conservative privacy screening for a local government-service assistant.

The scanner intentionally reports categories and counts without copying detected
values into diagnostics. It is a safety net, not a legal data classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Tuple

from .models import SourceUnit


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

LABELED_IDENTIFIER = re.compile(
    r"(?:"
    r"(?:الرقم|رقمي)\s+(?:الوطني|الشخصي)|رقم\s+(?:الهوية|هويتي|جواز\s*السفر)|"
    r"national\s+(?:id|number)|identity\s+(?:id|number)|passport\s+(?:id|number|no\.?)"
    r")\s*(?:هو|is)?\s*[:：#-]?\s*([A-Z0-9][A-Z0-9-]{5,21})",
    re.I,
)
IBAN = re.compile(r"\bJO\d{2}[A-Z0-9]{26}\b", re.I)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
PHONE = re.compile(
    r"(?<!\d)(?:(?:\+?962|00962)\s*[- ]?(?:7(?:[ -]?\d){8}|[2356](?:[ -]?\d){7})|"
    r"0\s*[- ]?(?:7(?:[ -]?\d){8}|[2356](?:[ -]?\d){7}))(?!\d)"
)
SECRET = re.compile(
    r"(?<![\w])(?:password|passwd|api[_ -]?key|secret[_ -]?key|كلمة\s*(?:المرور|السر)|مفتاح\s*(?:سري|واجهة))"
    r"\s*[:=]\s*([^\s,;]{6,})",
    re.I,
)
AUTH_SECRET = re.compile(
    r"(?<![\w])(?:otp|one[- ]?time\s+password|pin|cvv|cvc|رمز\s*(?:التحقق|التأكيد|المرة\s*الواحدة)|"
    r"الرقم\s*السري|رمز\s*الأمان)\s*[:：#=-]?\s*([A-Z0-9]{3,12})",
    re.I,
)
LABELED_CARD = re.compile(
    r"(?:card\s*(?:number|no\.?)|credit\s*card|debit\s*card|رقم\s*(?:البطاقة|الكرت))"
    r"\s*[:：#=-]?\s*((?:\d[ -]?){12,19}\d)",
    re.I,
)
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
FILENAME_IDENTIFIER = re.compile(r"(?<!\d)\d{8,14}(?!\d)")
MEANINGFUL_TOKEN = re.compile(r"[A-Za-z\u0600-\u06ff]+")
NON_MEANINGFUL_PRIVACY_WORDS = {
    "رقمي", "الرقم", "رقم", "الوطني", "الشخصي", "هويتي", "الهوية", "جوازي",
    "جواز", "السفر", "هاتفي", "الهاتف", "جوالي", "الجوال", "بريدي", "البريد",
    "بطاقتي", "البطاقة", "حسابي", "الحساب", "my", "id", "identity", "number",
    "passport", "email", "phone", "mobile", "iban", "card", "account",
}


@dataclass(frozen=True)
class PrivacyDecision:
    """Safe routing result for one user question; never contains detected values."""

    action: Literal["allow", "redact", "block"]
    sanitized_question: str
    categories: Tuple[str, ...] = ()
    redacted_count: int = 0
    language: Literal["ar", "en"] = "ar"
    reason: str = ""


def _luhn(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def scan_personal_data(text: str) -> List[Dict[str, object]]:
    """Find likely identifiers without returning their raw values."""
    normalized = text.translate(ARABIC_DIGITS)
    findings: List[Dict[str, object]] = []
    patterns = (
        ("labeled_identity_number", "high", LABELED_IDENTIFIER),
        ("jordan_iban", "high", IBAN),
        ("credential_or_secret", "high", SECRET),
        ("authentication_secret", "high", AUTH_SECRET),
        ("email_address", "medium", EMAIL),
        ("jordan_phone_number", "medium", PHONE),
    )
    for code, severity, pattern in patterns:
        count = len(pattern.findall(normalized))
        if count:
            findings.append({"code": code, "severity": severity, "count": count})
    cards = [match.group(0) for match in CARD_CANDIDATE.finditer(normalized)]
    valid_cards = sum(_luhn(candidate.replace(" ", "").replace("-", "")) for candidate in cards)
    if valid_cards:
        findings.append({"code": "payment_card_number", "severity": "high", "count": valid_cards})
    return findings


def scan_units_privacy(units: Iterable[SourceUnit]) -> Dict[str, object]:
    findings: List[Dict[str, object]] = []
    for unit in units:
        for finding in scan_personal_data(unit.text):
            enriched = dict(finding)
            enriched["page"] = unit.page
            enriched["section"] = unit.section
            findings.append(enriched)
    high = sum(item["severity"] == "high" for item in findings)
    medium = sum(item["severity"] == "medium" for item in findings)
    return {
        "classification": "restricted" if high else ("personal_data_review" if medium else "public_or_unclassified"),
        "high_findings": high,
        "medium_findings": medium,
        "findings": findings,
    }


def filename_privacy_findings(filename: str) -> List[Dict[str, object]]:
    """Flag identifier-like filenames because provenance metadata is also displayed."""
    findings = scan_personal_data(filename)
    normalized = filename.translate(ARABIC_DIGITS)
    count = len(FILENAME_IDENTIFIER.findall(normalized))
    if count:
        findings.append({"code": "identifier_in_filename", "severity": "high", "count": count})
    return findings


def _language(text: str) -> Literal["ar", "en"]:
    arabic = sum("\u0600" <= char <= "\u06ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return "ar" if arabic >= latin else "en"


def _replace_spans(text: str, spans: List[Tuple[int, int]], placeholder: str) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + placeholder + text[end:]
    text = re.sub(r"(?:\s*\[بيانات شخصية محذوفة\]\s*){2,}", " [بيانات شخصية محذوفة] ", text)
    text = re.sub(r"(?:\s*\[PERSONAL DATA REDACTED\]\s*){2,}", " [PERSONAL DATA REDACTED] ", text)
    return re.sub(r"\s+", " ", text).strip(" ,،;؛:-")


def _has_meaningful_request(text: str) -> bool:
    value = re.sub(r"\[(?:بيانات شخصية محذوفة|PERSONAL DATA REDACTED)\]", " ", text)
    tokens = [token.casefold() for token in MEANINGFUL_TOKEN.findall(value)]
    return any(len(token) >= 3 and token not in NON_MEANINGFUL_PRIVACY_WORDS for token in tokens)


def decide_question_privacy(question: str) -> PrivacyDecision:
    """Allow useful context, redact direct identifiers, and block authentication secrets."""
    language = _language(question)
    normalized = question.translate(ARABIC_DIGITS)

    blocked_categories = []
    for code, pattern in (("credential_or_secret", SECRET), ("authentication_secret", AUTH_SECRET)):
        if pattern.search(normalized):
            blocked_categories.append(code)
    if blocked_categories:
        return PrivacyDecision(
            action="block",
            sanitized_question="",
            categories=tuple(sorted(set(blocked_categories))),
            language=language,
            reason="authentication_secret",
        )

    matches: List[Tuple[int, int, str]] = []
    for code, pattern in (
        ("labeled_identity_number", LABELED_IDENTIFIER),
        ("jordan_iban", IBAN),
        ("email_address", EMAIL),
        ("jordan_phone_number", PHONE),
        ("payment_card_number", LABELED_CARD),
    ):
        matches.extend((match.start(), match.end(), code) for match in pattern.finditer(normalized))
    for match in CARD_CANDIDATE.finditer(normalized):
        if _luhn(match.group(0).replace(" ", "").replace("-", "")):
            matches.append((match.start(), match.end(), "payment_card_number"))

    if not matches:
        return PrivacyDecision(action="allow", sanitized_question=question.strip(), language=language)

    # Merge overlapping detections so the report count cannot reveal formatting details.
    merged: List[Tuple[int, int]] = []
    for start, end, _ in sorted(matches):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    placeholder = "[بيانات شخصية محذوفة]" if language == "ar" else "[PERSONAL DATA REDACTED]"
    sanitized = _replace_spans(normalized, merged, placeholder)
    categories = tuple(sorted({code for _, _, code in matches}))
    if not _has_meaningful_request(sanitized):
        return PrivacyDecision(
            action="block",
            sanitized_question="",
            categories=categories,
            redacted_count=len(merged),
            language=language,
            reason="insufficient_after_redaction",
        )
    return PrivacyDecision(
        action="redact",
        sanitized_question=sanitized,
        categories=categories,
        redacted_count=len(merged),
        language=language,
        reason="direct_identifiers_removed",
    )
