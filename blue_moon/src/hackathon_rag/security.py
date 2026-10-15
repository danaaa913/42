"""Document provenance and conservative prompt-injection screening."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .models import SourceUnit


INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
BIDI_CONTROLS = re.compile(r"[\u202a-\u202e\u2066-\u2069]")

# These patterns intentionally target imperative override/exfiltration language rather
# than words such as "prompt" or "instruction" on their own. A document explaining
# prompt injection should not be quarantined merely for discussing the topic.
HIGH_RISK_PATTERNS = (
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+"
            r"(?:instructions?|prompts?|rules?)|"
            r"(?:تجاهل|انسَ|انسى|تخط(?:ى|ي)|الغ(?:ِ|ي))\s+(?:كل\s+)?(?:التعليمات|الأوامر|القواعد)\s+"
            r"(?:السابقة|الماضية|الأصلية)",
            re.I,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"(?:you are now|act as|switch to)\s+(?:a\s+)?(?:system|developer|admin|jailbreak)|"
            r"(?:أنت الآن|انت الان|تصرف كأنك|تحول إلى)\s+(?:النظام|مطور|مدير|مسؤول)",
            re.I,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"(?:reveal|print|show|repeat|leak|expose)\s+(?:the\s+)?(?:system|developer)\s+(?:prompt|instructions?)|"
            r"(?:اكشف|اطبع|اعرض|كرر|سرّب|سرب)\s+(?:تعليمات|أوامر|موجه|برومبت)\s+(?:النظام|المطور)",
            re.I,
        ),
    ),
    (
        "tool_command",
        re.compile(
            r"(?:call|invoke|execute|run)\s+(?:the\s+)?(?:tool|function|command)\b|"
            r"(?:نفذ|شغل|استدع)\S*\s+(?:الأداة|الدالة|الأمر)",
            re.I,
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(text: str, start: int, end: int, radius: int = 80) -> str:
    value = text[max(0, start - radius): min(len(text), end + radius)]
    value = re.sub(r"\s+", " ", value).strip()
    return value[:240]


def scan_text(text: str) -> List[Dict[str, object]]:
    """Return explainable findings without modifying the supplied text."""
    findings: List[Dict[str, object]] = []
    invisible_count = len(INVISIBLE_CHARS.findall(text))
    bidi_count = len(BIDI_CONTROLS.findall(text))
    if invisible_count:
        findings.append({
            "code": "invisible_unicode",
            "severity": "medium",
            "count": invisible_count,
            "sample": "Invisible Unicode characters were present before normalization.",
        })
    if bidi_count:
        findings.append({
            "code": "bidi_controls",
            "severity": "medium",
            "count": bidi_count,
            "sample": "Bidirectional text controls were present before normalization.",
        })
    for code, pattern in HIGH_RISK_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            first = matches[0]
            findings.append({
                "code": code,
                "severity": "high",
                "count": len(matches),
                "sample": _sample(text, first.start(), first.end()),
            })
    return findings


def scan_units(units: Iterable[SourceUnit]) -> Dict[str, object]:
    findings: List[Dict[str, object]] = []
    for unit in units:
        for finding in scan_text(unit.text):
            enriched = dict(finding)
            enriched["page"] = unit.page
            enriched["section"] = unit.section
            findings.append(enriched)
    high = sum(item["severity"] == "high" for item in findings)
    medium = sum(item["severity"] == "medium" for item in findings)
    return {
        "status": "quarantined" if high else ("suspicious" if medium else "clean"),
        "high_findings": high,
        "medium_findings": medium,
        "findings": findings,
    }
