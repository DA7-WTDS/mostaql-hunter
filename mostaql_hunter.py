import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
import httpx
from bs4 import BeautifulSoup

MOSTAQL_URL = "https://mostaql.com/projects/development?sort=latest"
SEEN_FILE = Path("seen.json")
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
MIN_BUDGET = 50
MAX_BIDS = 10
MIN_SCORE_THRESHOLD = 35
MAX_SEEN_HISTORY = 500

POSITIVE_KEYWORDS = {
    ".net": 35,
    "asp.net": 35,
    "c#": 35,
    "دوت نت": 35,
    "سي شارب": 35,
    "django": 30,
    "جانجو": 30,
    "python": 25,
    "بايثون": 25,
    "fastapi": 25,
    "backend": 25,
    "باك اند": 25,
    "api": 20,
    "واجهه برمجيه": 20,
    "واجهات برمجيه": 20,
    "لوحه تحكم": 15,
    "قاعده بيانات": 20,
    "قواعد بيانات": 20,
    "postgresql": 20,
    "sql server": 20,
    "redis": 20,
    "docker": 15,
    "node": 20,
    "mern": 20,
    "scraping": 20,
    "استخراج بيانات": 20,
    "سحب بيانات": 20,
    "اتمته": 20,
    "بوت": 15,
    "automation": 20,
    "playwright": 20,
    "selenium": 20,
    "yolo": 20,
    "opencv": 20,
    "openai": 20,
    "llm": 20,
    "ذكاء اصطناعي": 15,
}

NEGATIVE_KEYWORDS = {
    "wordpress": -40,
    "ووردبريس": -40,
    "وردبريس": -40,
    "elementor": -40,
    "اليمنتور": -40,
    "woocommerce": -35,
    "ووكومرس": -35,
    "سله": -30,
    "shopify": -35,
    "شوبيفاي": -35,
    "flutter": -30,
    "فلاتر": -30,
    "laravel": -25,
    "لارافيل": -25,
    "php": -25,
    "android": -25,
    "اندرويد": -25,
    "ios": -25,
    "ايفون": -25,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
}

TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
BUDGET_RE = re.compile(r"\$\s*(\d+(?:,\d+)*(?:\.\d+)?)")
BIDS_RE = re.compile(r"(\d+)\s*عروض?")


@dataclass
class Project:
    url: str
    title: str
    snippet: str
    meta_text: str
    full_description: str = ""
    budget_text: str = "غير محدد"
    min_budget: int | None = None
    bids_count: int = 0
    score: int = 0


def normalize_arabic(text: str) -> str:
    text = text.lower()
    text = TASHKEEL_RE.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def load_seen() -> list[str]:
    if not SEEN_FILE.exists():
        return []
    return json.loads(SEEN_FILE.read_text(encoding="utf-8"))


def save_seen(urls: list[str]) -> None:
    trimmed = urls[-MAX_SEEN_HISTORY:]
    SEEN_FILE.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_bids_count(meta_text: str) -> int:
    normalized = normalize_arabic(meta_text)
    if "اضف اول عرض" in normalized:
        return 0
    if "عرض واحد" in normalized:
        return 1
    if "عرضين" in normalized:
        return 2
    match = BIDS_RE.search(normalized)
    return int(match.group(1)) if match else 0


def score_text(text: str, bids_count: int) -> int:
    normalized = normalize_arabic(text)
    score = 0
    for kw, weight in POSITIVE_KEYWORDS.items():
        if kw in normalized:
            score += weight
    for kw, penalty in NEGATIVE_KEYWORDS.items():
        if kw in normalized:
            score += penalty
    if bids_count <= 2:
        score += 20
    elif bids_count <= 5:
        score += 10
    return score


async def fetch_project_list(client: httpx.AsyncClient) -> list[Project]:
    resp = await client.get(MOSTAQL_URL, headers=HEADERS, timeout=20.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    projects: list[Project] = []
    for row in soup.select("tr.project-row"):
        link_el = row.select_one("h2 a")
        if not link_el or not link_el.get("href"):
            continue

        title = link_el.get_text(strip=True)
        url = link_el["href"]
        snippet_el = row.select_one("p.project__brief")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        meta_text = row.get_text(" ", strip=True)
        bids_count = parse_bids_count(meta_text)

        project = Project(
            url=url,
            title=title,
            snippet=snippet,
            meta_text=meta_text,
            bids_count=bids_count,
            score=score_text(f"{title} {snippet}", bids_count),
        )
        projects.append(project)

    return projects


async def enrich_project_details(client: httpx.AsyncClient, project: Project) -> None:
    await asyncio.sleep(random.uniform(1.0, 2.5))
    resp = await client.get(project.url, headers=HEADERS, timeout=20.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    desc_el = soup.select_one("#project-brief-panel, .project-details, .text-wrapper-div")
    project.full_description = desc_el.get_text(" ", strip=True) if desc_el else project.snippet

    meta_panel = soup.select_one("#project-meta-panel, .project-meta, aside")
    meta_str = meta_panel.get_text(" ", strip=True) if meta_panel else soup.get_text(" ", strip=True)
    amounts = [int(float(m.replace(",", ""))) for m in BUDGET_RE.findall(meta_str)]

    if amounts:
        project.min_budget = min(amounts)
        if len(amounts) >= 2:
            project.budget_text = f"${amounts[0]} - ${amounts[1]}"
        else:
            project.budget_text = f"${amounts[0]}"

    project.score = score_text(f"{project.title} {project.full_description}", project.bids_count)


async def send_alert(client: httpx.AsyncClient, project: Project) -> None:
    summary = (project.full_description or project.snippet)[:220]
    message = (
        f"التقييم: {project.score} | الميزانية: {project.budget_text} | العروض: {project.bids_count}\n"
        f"{summary}..."
    )
    payload = {
        "topic": NTFY_TOPIC,
        "title": project.title,
        "message": message,
        "priority": 4,
        "click": project.url,
    }
    resp = await client.post("https://ntfy.sh/", json=payload, timeout=20.0)
    resp.raise_for_status()


async def main() -> None:
    seen_list = load_seen()
    seen_set = set(seen_list)
    updated = False

    async with httpx.AsyncClient(follow_redirects=True) as client:
        projects = await fetch_project_list(client)
        for project in projects:
            if project.url in seen_set:
                continue

            seen_set.add(project.url)
            seen_list.append(project.url)
            updated = True

            if project.bids_count > MAX_BIDS or project.score < 0:
                continue

            try:
                await enrich_project_details(client, project)
            except Exception as exc:
                print(f"Detail fetch failed for {project.url}: {exc}")
                continue

            if project.min_budget is not None and project.min_budget < MIN_BUDGET:
                continue

            if project.score >= MIN_SCORE_THRESHOLD:
                await send_alert(client, project)
                await asyncio.sleep(1.5)

    if updated:
        save_seen(seen_list)


if __name__ == "__main__":
    asyncio.run(main())