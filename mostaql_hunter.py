import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from playwright.async_api import Page, async_playwright

MOSTAQL_URL = "https://mostaql.com/projects/development?sort=latest"
SEEN_FILE = Path("seen.json")
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MIN_BUDGET = 50
MAX_BIDS = 10
MIN_LLM_SCORE = 70
MAX_SEEN_HISTORY = 500

TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
BUDGET_RE = re.compile(r"(?:\$\s*(\d+(?:,\d+)*(?:\.\d+)?)|(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:\$|دولار))")
BIDS_RE = re.compile(r"(\d+)\s*(?:عرض|عروض)")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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
    llm_reason: str = ""


def normalize_arabic(text: str) -> str:
    text = text.lower()
    text = TASHKEEL_RE.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def load_seen() -> list[dict | str]:
    if not SEEN_FILE.exists():
        return []
    return json.loads(SEEN_FILE.read_text(encoding="utf-8"))


def save_seen(entries: list[dict | str]) -> None:
    trimmed = entries[-MAX_SEEN_HISTORY:]
    SEEN_FILE.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")


def build_seen_entry(project: Project, status: str, reason: str) -> dict:
    return {
        "url": project.url,
        "title": project.title,
        "status": status,
        "reason": reason,
        "score": project.score,
        "bids": project.bids_count,
        "budget": project.budget_text,
    }


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


async def fetch_project_list(page: Page) -> list[Project]:
    await page.goto(MOSTAQL_URL, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_selector("tr.project-row", timeout=25000)
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    print(f"Page Title: {soup.title.get_text(strip=True) if soup.title else 'No Title'}")

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

        projects.append(
            Project(
                url=url,
                title=title,
                snippet=snippet,
                meta_text=meta_text,
                bids_count=bids_count,
            )
        )

    print(f"Projects found: {len(projects)}")
    return projects


async def enrich_project_details(page: Page, project: Project) -> None:
    await asyncio.sleep(random.uniform(1.0, 2.5))
    await page.goto(project.url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_selector("#project-brief-panel, .project-details, .text-wrapper-div", timeout=15000)
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    desc_el = soup.select_one("#project-brief-panel, .project-details, .text-wrapper-div")
    project.full_description = desc_el.get_text(" ", strip=True) if desc_el else project.snippet

    budget_source = ""
    budget_label = soup.find(string=re.compile(r"الميزانية"))
    if budget_label and budget_label.parent:
        budget_row = budget_label.parent.find_parent(["tr", "li", "div"])
        if budget_row:
            budget_source = budget_row.get_text(" ", strip=True)

    if not budget_source:
        meta_panel = soup.select_one("#project-meta-panel, .project-meta, aside")
        budget_source = meta_panel.get_text(" ", strip=True) if meta_panel else soup.get_text(" ", strip=True)

    raw_matches = BUDGET_RE.findall(budget_source)
    amounts = [int(float((m[0] or m[1]).replace(",", ""))) for m in raw_matches if (m[0] or m[1])]

    if amounts:
        project.min_budget = min(amounts)
        if len(amounts) >= 2:
            project.budget_text = f"${amounts[0]} - ${amounts[1]}"
        else:
            project.budget_text = f"${amounts[0]}"


async def score_with_llm(project: Project) -> tuple[int, str]:
    prompt = f"""You are a freelance job screener for a backend software engineer.
Evaluate this Arabic/English Mostaql job posting and return ONLY a JSON object with keys "score" (integer 0-100) and "reason" (one short sentence in English).

Developer Profile:
- Strong skills (score 75-100): C#/.NET, ASP.NET Core, Python, Django, FastAPI, Node.js/MERN, REST APIs, PostgreSQL, SQL Server, Redis, Docker, web scraping (Playwright/Selenium/BeautifulSoup), Python automation/bots, LLM/OpenAI API integrations, YOLO/OpenCV.
- Acceptable if clear scope (score 60-74): General backend fixes, database design, data extraction to Excel/Sheets.
- Reject (score 0-30): WordPress, Elementor, WooCommerce, Shopify, Salla, Zid, Flutter, native Android/iOS mobile apps, PHP/Laravel, pure graphic design, or training custom ML models from scratch.

Job Title: {project.title}
Budget: {project.budget_text}
Bids: {project.bids_count}
Description: {(project.full_description or project.snippet)[:1500]}
"""
    resp = await gemini_client.aio.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    data = json.loads(resp.text)
    return int(data.get("score", 0)), str(data.get("reason", "no reason provided"))


async def send_alert(page: Page, project: Project) -> None:
    message = (
        f"التقييم: {project.score}/100 | الميزانية: {project.budget_text} | العروض: {project.bids_count}\n"
        f"السبب: {project.llm_reason}\n"
        f"{(project.full_description or project.snippet)[:180]}..."
    )
    payload = {
        "topic": NTFY_TOPIC,
        "title": project.title,
        "message": message,
        "priority": 4,
        "click": project.url,
    }
    resp = await page.request.post("https://ntfy.sh/", data=payload, timeout=20000)
    if not resp.ok:
        raise RuntimeError(f"ntfy error {resp.status}: {await resp.text()}")


async def main() -> None:
    seen_list = load_seen()
    seen_set = {item["url"] if isinstance(item, dict) else item for item in seen_list}
    updated = False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            extra_http_headers={"Accept-Language": "ar,en-US;q=0.9,en;q=0.8"},
        )
        page = await context.new_page()

        projects = await fetch_project_list(page)
        for project in projects:
            if project.url in seen_set:
                continue

            seen_set.add(project.url)
            updated = True

            if project.bids_count > MAX_BIDS:
                seen_list.append(build_seen_entry(project, "rejected", f"too_many_bids ({project.bids_count} > {MAX_BIDS})"))
                continue

            try:
                await enrich_project_details(page, project)
            except Exception as exc:
                print(f"Detail fetch failed for {project.url}: {exc}")
                seen_list.append(build_seen_entry(project, "rejected", "detail_fetch_failed"))
                continue

            if project.min_budget is not None and project.min_budget < MIN_BUDGET:
                seen_list.append(build_seen_entry(project, "rejected", f"budget_too_low (${project.min_budget} < ${MIN_BUDGET})"))
                continue

            try:
                project.score, project.llm_reason = await score_with_llm(project)
            except Exception as exc:
                print(f"LLM scoring failed for {project.url}: {exc}")
                seen_list.append(build_seen_entry(project, "rejected", "llm_error"))
                continue

            if project.score < MIN_LLM_SCORE:
                seen_list.append(build_seen_entry(project, "rejected", f"llm_rejected: {project.llm_reason}"))
                continue

            await send_alert(page, project)
            seen_list.append(build_seen_entry(project, "sent", f"llm_matched: {project.llm_reason}"))
            await asyncio.sleep(1.5)

        await browser.close()

    if updated:
        save_seen(seen_list)


if __name__ == "__main__":
    asyncio.run(main())