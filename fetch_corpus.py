# -*- coding: utf-8 -*-
"""Скачать корпус Википедии по страницам из data/fresh.jsonl (ДЗ1).

Запуск: python fetch_corpus.py
Результат:
  data/corpus_2026.jsonl  — полные тексты страниц
  data/questions.jsonl    — вопросы, чья evidence есть в корпусе
"""
import json, re, sys, time
from pathlib import Path
import requests

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
QUESTIONS = DATA / "fresh.jsonl"
WIKI = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "agents-course-hw02/1.0 (https://postypashki.ru; educational)"}


def page_text(title):
    for attempt in range(5):
        try:
            r = requests.get(
                WIKI, timeout=40, headers=HEADERS,
                params={"action": "query", "prop": "extracts", "explaintext": 1,
                        "titles": title, "redirects": 1, "format": "json"},
            )
            if r.status_code == 200 and "application/json" in r.headers.get("content-type", ""):
                page = next(iter(r.json()["query"]["pages"].values()))
                return page.get("title", title), page.get("extract", "")
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return title, ""


def norm(t):
    return " ".join(re.sub(r"[^\w\s]", " ", str(t).lower()).split())


def main():
    tasks = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    titles = sorted({t["page"] for t in tasks if t.get("page")})

    # reuse already downloaded pages if present
    existing = {}
    corpus_path = DATA / "corpus_2026.jsonl"
    if corpus_path.exists():
        for row in [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]:
            existing[row["page"]] = row

    corpus = []
    for title in titles:
        if title in existing and existing[title].get("text"):
            corpus.append(existing[title])
            print(f"{title:42s} {len(existing[title]['text']):8d} символов (кэш)")
            continue
        real, text = page_text(title)
        if text:
            row = {"page": real, "url": "https://en.wikipedia.org/wiki/" + real.replace(" ", "_"), "text": text}
            corpus.append(row)
            existing[real] = row
        print(f"{real:42s} {len(text):8d} символов")
        time.sleep(1.2)

    with corpus_path.open("w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = {c["page"]: norm(c["text"]) for c in corpus}
    # also map by original title aliases
    for c in corpus:
        texts.setdefault(c["page"], norm(c["text"]))

    kept = []
    for t in tasks:
        page = t.get("page", "")
        body = texts.get(page, "")
        if not body:
            # try fuzzy: any corpus page containing evidence
            ev = norm(t["evidence"])[:120]
            hit = next((p for p, body2 in texts.items() if ev and ev in body2), None)
            if not hit:
                continue
            page = hit
        elif norm(t["evidence"])[:120] not in body:
            continue
        kept.append({
            "id": t["id"], "source": "fresh", "question": t["question"], "answer": t["answer"],
            "evidence": t["evidence"], "page": page, "url": t.get("url") or (
                "https://en.wikipedia.org/wiki/" + page.replace(" ", "_")
            ),
        })

    with (DATA / "questions.jsonl").open("w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    lines = sum(len([l for l in c["text"].splitlines() if len(l.strip()) >= 40]) for c in corpus)
    print(f"\nстраниц: {len(corpus)}, символов: {sum(len(c['text']) for c in corpus)}, содержательных строк: {lines}")
    print(f"вопросов с цитатой в корпусе: {len(kept)} из {len(tasks)}")


if __name__ == "__main__":
    main()
