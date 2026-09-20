# -*- coding: utf-8 -*-
"""Свежие вопросы для домашки: факты 2026 года из Википедии, на которых Sonnet без инструментов ошибается.

Запуск: python fetch_fresh.py [--limit 120]
Результат: data/fresh_2026.jsonl и сводка в консоли.

Как собирается:
1. Берём текст обзорных статей Википедии про 2026 год, режем на факты с датой (январь-август 2026).
2. Дешёвая модель делает из каждого факта вопрос с коротким ответом; ответ должен дословно быть в факте.
3. Сильная модель отвечает на вопрос без инструментов; оставляем только вопросы, где она ошиблась.
4. Агент на дешёвой модели с поиском по Википедии пробует ответить; результат пишется в поле agent_ok.
"""
import os, re, sys, json, time, random, argparse
from pathlib import Path
import requests
from pydantic import BaseModel, Field, ValidationError

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
for env in [HERE / ".env", HERE.parent.parent / "demo" / ".env"]:
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY=") and not os.getenv("OPENROUTER_API_KEY"):
                os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip()
assert os.getenv("OPENROUTER_API_KEY"), "нужен OPENROUTER_API_KEY"

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "agents-course-fresh-dataset"}
CHEAP, STRONG = "openai/gpt-4o-mini", "anthropic/claude-sonnet-4.6"
WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-seminar01/1.0 (https://postypashki.ru; educational project)"}
PAGES = ["2026", "2026 in science", "2026 in sports", "2026 in film", "2026 in spaceflight", "2026 in video games",
         "2026 in music", "2026 in television", "2026 in literature", "2026 Winter Olympics", "2026 FIFA World Cup",
         "2026 in aviation", "2026 in politics", "2026 in the United States", "2026 in the United Kingdom",
         "2026 in Russia", "2026 in Germany", "2026 in Japan", "2026 in India", "2026 in association football",
         "2026 in basketball", "2026 in tennis", "2026 in Formula One", "2026 in art", "2026 in architecture",
         "2026 in Canada", "2026 in Australia", "2026 in France", "2026 in Italy", "2026 in Spain", "2026 in Brazil",
         "2026 in China", "2026 in South Korea", "2026 in Mexico", "2026 in Ukraine", "2026 in Turkey",
         "2026 in American television", "2026 in British television", "2026 in chess", "2026 in cycling",
         "2026 in golf", "2026 in ice hockey", "2026 in athletics", "2026 in boxing", "2026 in mixed martial arts",
         "2026 in esports", "2026 in comics", "2026 in animation", "2026 in classical music", "2026 in paleontology",
         "2026 in archaeology", "2026 in rail transport", "2026 in American football", "2026 in baseball",
         "2026 in motorsport", "2026 in rugby union", "2026 in swimming", "2026 in figure skating", "2026 in cricket",
         "Portal:Current events/January 2026", "Portal:Current events/February 2026", "Portal:Current events/March 2026",
         "Portal:Current events/April 2026", "Portal:Current events/May 2026", "Portal:Current events/June 2026",
         "Portal:Current events/July 2026", "Portal:Current events/August 2026"]
MONTHS = "January|February|March|April|May|June|July|August"
COST = 0.0


def post(body, attempts=5):
    """POST к OpenRouter с повтором при сетевых ошибках и кодах 429/5xx."""
    global COST
    problem = "нет ответа"
    for attempt in range(attempts):
        try:
            r = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                data = r.json()
                COST += (data.get("usage") or {}).get("cost") or 0.0
                return data
            problem = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            problem = type(e).__name__
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(problem)


def chat(messages, model, temperature=0):
    data = post({"model": model, "messages": messages, "temperature": temperature, "usage": {"include": True}})
    return data["choices"][0]["message"]["content"] or ""


def wiki(params):
    for attempt in range(3):
        try:
            r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=20)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                return r.json()["query"]
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("Википедия недоступна")


def page_text(title):
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})["pages"]
    page = next(iter(pages.values()))
    return page.get("title", title), page.get("extract", "")


SKIP = re.compile(r"\(b\.\s*\d{4}|\(d\.\s*\d{4}|\bdies\b|\bdied\b|\bdeath\b|birthday|anniversary|years old|\baged\b|"
                  r"[A-Za-zÀ-ÿ]+,\s\d{2,3},\s|"
                  r"\b(will|scheduled|expected|planned|is set to|to be held|upcoming)\b", re.I)


def facts_from(text):
    out = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not (60 <= len(line) <= 420):
            continue
        if not re.search(rf"\b({MONTHS})\b\s*\d{{0,2}}", line):
            continue
        if SKIP.search(line):
            continue
        out.append(line)
    return out


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def json_from(text):
    m = re.search(r"\[.*\]|\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("нет JSON")
    cleaned = re.sub(r'(\\["\\/bfnrtu])|\\', lambda e: e.group(1) or "\\\\", m.group(0))
    return json.loads(cleaned, strict=False)


GEN_PROMPT = """Each fact below comes from a Wikipedia page about 2026 (the page title is given in parentheses). For each fact write one quiz question with a short unambiguous answer.
Rules: the answer is a single entity, at most five words: a person's name, a number, a title, a place or a date, nothing else;
the answer must appear verbatim in the fact; the question asks for a specific detail of a specific described event, never
"what happened" or "who is mentioned"; the question is self-contained: it names the country or field from the page title,
the month and year 2026 and enough of the event to identify it without seeing the fact; never put the answer into the question;
skip facts about future or scheduled events and facts without a clear short checkable detail.
Return only a JSON array of objects {"i": <fact index>, "question": "...", "answer": "..."}."""


def good_pair(question, answer, fact):
    a, q = normalize(answer), normalize(question)
    return (bool(a) and a in normalize(fact) and len(answer) <= 40 and len(answer.split()) <= 4
            and question.rstrip().endswith("?") and 30 <= len(question) <= 240 and a not in q)


def generate(facts):
    items = []
    for start in range(0, len(facts), 8):
        batch = facts[start:start + 8]
        listing = "\n".join(f"[{start + k}] ({f['page']}) {f['text']}" for k, f in enumerate(batch))
        try:
            for obj in json_from(chat([{"role": "system", "content": GEN_PROMPT}, {"role": "user", "content": listing}], CHEAP)):
                i, q, a = int(obj["i"]), str(obj["question"]).strip(), str(obj["answer"]).strip()
                if 0 <= i < len(facts) and good_pair(q, a, facts[i]["text"]):
                    items.append({**facts[i], "question": q, "answer": a})
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            print("  батч пропущен:", e, flush=True)
    return items


class Answer(BaseModel):
    reasoning: str = Field(description="brief reasoning")
    final: str = Field(description="short final answer only; if you do not know, write 'unknown'")


SCHEMA_PROMPT = ("Answer strictly with one JSON object matching this schema, no text around, no LaTeX:\n"
                 + json.dumps(Answer.model_json_schema()))


def strong_answer(question):
    for _ in range(2):
        try:
            return Answer.model_validate(json_from(chat([{"role": "system", "content": SCHEMA_PROMPT},
                                                          {"role": "user", "content": question}], STRONG))).final
        except (ValidationError, ValueError):
            pass
    return "unknown"


def web_search(query):
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["search"]
    if not hits:
        return "nothing found"
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1, "titles": hits[0]["title"]})["pages"]
    text = " ".join(next(iter(pages.values())).get("extract", "").split())[:1500]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")


def page_find(title, keywords):
    """Строки полного текста статьи, где встречается хотя бы половина ключевых слов."""
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if not text:
        return "no such page"
    words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = [l for l in lines if sum(w in l.lower() for w in words) >= max(1, (len(words) + 1) // 2)]
    return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"


TOOLS = {
    "web_search": (lambda a: web_search(a["query"]), {"type": "function", "function": {
        "name": "web_search", "description": "Searches English Wikipedia and returns the intro of the best article and the titles of two more",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "short query: a name, a title, a term"}},
                       "required": ["query"]}}}),
    "page_find": (lambda a: page_find(a["title"], a["keywords"]), {"type": "function", "function": {
        "name": "page_find", "description": "Looks inside the full text of a Wikipedia article (for example '2026 in Japan') and returns the lines containing the keywords",
        "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "exact article title"},
                                                        "keywords": {"type": "string", "description": "two to five keywords from the question"}},
                       "required": ["title", "keywords"]}}}),
}
AGENT_SYSTEM = ("You answer questions about events of 2026. Never answer from memory: use web_search to find the right article, "
                "then page_find to look inside long articles such as '2026 in <country or field>' for the specific fact. "
                "Do not repeat the same call. When done, write the last line as FINAL: <short answer>.")


def agent_answer(question, max_steps=7):
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
    seen = set()
    schemas = [t[1] for t in TOOLS.values()]
    for _ in range(max_steps):
        try:
            data = post({"model": CHEAP, "messages": messages, "tools": schemas, "temperature": 0, "usage": {"include": True}})
        except RuntimeError:
            return ""
        msg = data["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            m = re.search(r"FINAL:\s*(.+)", msg.get("content") or "")
            return m.group(1).strip() if m else (msg.get("content") or "")
        for c in calls:
            key = (c["function"]["name"], c["function"]["arguments"])
            if key in seen:
                return ""
            seen.add(key)
            try:
                result = TOOLS[key[0]][0](json.loads(key[1]))
            except Exception as e:
                result = f"tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)[:2500]})
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120, help="сколько фактов отдать на генерацию вопросов")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--append", action="store_true", help="дописать к существующему файлу без дублей")
    ap.add_argument("--out", default="data/fresh_2026.jsonl", help="куда писать результат")
    ap.add_argument("--target", type=int, default=130, help="остановиться, когда в файле столько вопросов")
    args = ap.parse_args()
    random.seed(args.seed)
    out = HERE / args.out
    existing = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()] if args.append and out.exists() else []
    used = {e["evidence"] for e in existing}

    facts = []
    for title in PAGES:
        try:
            real, text = page_text(title)
        except Exception as e:
            print(f"{title:40s} пропущена: {e}", flush=True)
            continue
        got = [f for f in facts_from(text) if f not in used]
        for f in got:
            facts.append({"page": real, "url": "https://en.wikipedia.org/wiki/" + real.replace(" ", "_"), "text": f})
        print(f"{real:40s} {len(got):4d} фактов", flush=True)
    random.shuffle(facts)
    facts = facts[:args.limit]
    print(f"кандидатов на генерацию: {len(facts)}", flush=True)

    items = generate(facts)
    print(f"вопросов, прошедших фильтры: {len(items)}", flush=True)

    kept = 0
    with out.open("a" if args.append else "w", encoding="utf-8") as f:
        for it in items:
            if len(existing) + kept >= args.target:
                break
            try:
                it["sonnet_answer"] = strong_answer(it["question"])
            except RuntimeError as e:
                print("  пропуск, сеть:", e, flush=True)
                continue
            if normalize(it["answer"]) in normalize(it["sonnet_answer"]):
                continue
            it["agent_answer"] = agent_answer(it["question"])
            it["agent_ok"] = normalize(it["answer"]) in normalize(it["agent_answer"])
            row = {"id": f"fresh-{len(existing) + kept}", "source": "fresh", "question": it["question"], "answer": it["answer"],
                   "evidence": it["text"], "page": it["page"], "url": it["url"], "sonnet_answer": it["sonnet_answer"],
                   "agent_ok": it["agent_ok"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            print(f"  {'agent+' if it['agent_ok'] else 'agent-'} | {it['question'][:90]} -> {it['answer']} | sonnet: {it['sonnet_answer'][:40]}", flush=True)
    print(f"Sonnet без инструментов ошибся на {kept} из {len(items)}; в {out} теперь {len(existing) + kept} вопросов; потрачено ${COST:.3f}", flush=True)


if __name__ == "__main__":
    main()
