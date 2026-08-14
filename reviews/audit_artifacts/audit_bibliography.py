#!/usr/bin/env python3
"""Validate every final-report bibliography identifier and citation key.

Crossref is used for DOI metadata; arXiv and ordinary URLs are checked at their
canonical endpoints.  This validates identity/availability, not whether every
sentence is a faithful interpretation of the cited full text.
"""

from __future__ import annotations

import csv
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


TEX = Path("/Users/lizequan/Desktop/FYP_revised/最终版report.tex")
OUT = Path(__file__).with_name("bibliography_identifier_audit.csv")
USER_AGENT = "Rhinoform-paper-code-audit/1.0 (mailto:repository-review@example.invalid)"


def plain_tex(value: str) -> str:
    value = value.replace("\\_", "_").replace("~", " ")
    value = re.sub(r"\\[A-Za-z]+\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def fetch_json(url: str) -> tuple[int | str, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20, context=ssl.create_default_context()) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {}
    except Exception as exc:  # network status is evidence and belongs in the CSV
        return type(exc).__name__, {}


def check_url(url: str) -> int | str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20, context=ssl.create_default_context()) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception as exc:
        return type(exc).__name__


def inspect(entry: dict[str, str]) -> dict[str, str | int]:
    kind = entry["identifier_type"]
    identifier = entry["identifier"]
    if kind == "doi":
        endpoint = "https://api.crossref.org/works/" + urllib.parse.quote(identifier, safe="")
        status, payload = fetch_json(endpoint)
        message = payload.get("message", {}) if payload else {}
        title = " ".join(message.get("title", []) or [])
        year_parts = (
            message.get("published-print", {}).get("date-parts")
            or message.get("published-online", {}).get("date-parts")
            or message.get("issued", {}).get("date-parts")
            or []
        )
        year = year_parts[0][0] if year_parts and year_parts[0] else ""
        entry.update(http_status=status, resolved_title=title, resolved_year=year)
    elif kind == "arxiv":
        status = check_url(f"https://arxiv.org/abs/{identifier}")
        entry.update(http_status=status, resolved_title="", resolved_year="")
    elif kind == "url":
        status = check_url(identifier)
        entry.update(http_status=status, resolved_title="", resolved_year="")
    else:
        entry.update(http_status="NO_NETWORK_IDENTIFIER", resolved_title="", resolved_year="")
    return entry


def main() -> None:
    source = TEX.read_text(encoding="utf-8")
    definitions = re.findall(r"\\bibitem\{([^}]+)\}", source)
    uses: list[str] = []
    for match in re.finditer(r"\\cite\{([^}]+)\}", source):
        uses.extend(part.strip() for part in match.group(1).split(","))
    if set(definitions) != set(uses):
        raise RuntimeError(
            f"citation mismatch undefined={sorted(set(uses)-set(definitions))} "
            f"unused={sorted(set(definitions)-set(uses))}"
        )

    entries: list[dict[str, str]] = []
    blocks = re.findall(
        r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
        source,
        re.S,
    )
    for key, body in blocks:
        quoted = re.search(r"``(.*?),?''", body, re.S)
        report_title = plain_tex(quoted.group(1)) if quoted else ""
        doi = re.search(r"doi:([^\s]+)", body, re.I)
        arxiv = re.search(r"arXiv:([0-9.]+)", body, re.I)
        url = re.search(r"\\url\{([^}]+)\}", body, re.S)
        if doi:
            kind = "doi"
            identifier = doi.group(1).rstrip(".").replace("\\_", "_")
        elif arxiv:
            kind = "arxiv"
            identifier = arxiv.group(1).rstrip(".")
        elif url:
            kind = "url"
            identifier = url.group(1).strip()
        else:
            kind = "none"
            identifier = ""
        entries.append(
            {
                "bibkey": key,
                "citation_use_count": str(uses.count(key)),
                "identifier_type": kind,
                "identifier": identifier,
                "report_title": report_title,
            }
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(inspect, entries))
    fields = [
        "bibkey",
        "citation_use_count",
        "identifier_type",
        "identifier",
        "http_status",
        "report_title",
        "resolved_title",
        "resolved_year",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(str(row["http_status"]) == "200" for row in rows)
    print(f"bibitems={len(rows)} cited_unique={len(set(uses))} network_200={ok}")


if __name__ == "__main__":
    main()
