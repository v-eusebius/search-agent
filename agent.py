# -*- coding: utf-8 -*-
"""ReAct-агент для свежих вопросов 2025–2026: Wikipedia, page_find, Tavily, замер и каскад.

Запуск:
  python agent.py --test-tools
  python agent.py --freshness          # Sonnet без инструментов (нужен OPENROUTER_API_KEY)
  python agent.py --eval               # полный замер (подвыборка по умолчанию)
  python agent.py --eval --limit 0     # весь датасет
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.stdout.reconfigure(encoding="utf-8")

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def require_openrouter():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("положите OPENROUTER_API_KEY в .env (см. .env.example)")
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://postypashki.ru",
        "X-Title": "agents-course-hw01",
    }
MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}
WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-hw01/1.0 (https://postypashki.ru; educational)"}
TRACES, IMG = ROOT / "traces", ROOT / "img"
TRACES.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)

TOOLS: dict = {}
SYSTEM = (
    "You answer questions about recent events (2025–2026). Never answer from memory: "
    "use web_search (and tavily_search if available) to find the right article, "
    "then page_find to look inside long articles such as '2026 in <country or field>' for the specific fact. "
    "Do not repeat the same call. When done, write the last line as FINAL: <short answer>."
)


# ---------- ledger / chat ----------
@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds=0.0):
        cost = (usage or {}).get("cost") or 0.0
        self.calls.append({
            "tag": tag, "model": model.split("/")[-1],
            "prompt": (usage or {}).get("prompt_tokens", 0),
            "completion": (usage or {}).get("completion_tokens", 0),
            "cost": cost, "seconds": round(seconds, 2),
        })
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    def table(self):
        if not self.calls:
            return pd.DataFrame()
        df = pd.DataFrame(self.calls)
        return df.groupby(["tag", "model"]).agg(
            calls=("cost", "size"), prompt=("prompt", "sum"), completion=("completion", "sum"),
            cost=("cost", "sum"), seconds=("seconds", "sum"),
        ).round(5)


ledger = Ledger()


def post_with_retry(body, attempts=5):
    problem = "нет ответа"
    headers = require_openrouter()
    for attempt in range(attempts):
        try:
            r = requests.post(CHAT_URL, json=body, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()
            problem = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            problem = type(e).__name__
        time.sleep(min(60, 2 * 2 ** attempt))
    raise RuntimeError(problem)


def chat(messages, model, tools=None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    started = time.perf_counter()
    data = post_with_retry(body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]


# ---------- utils ----------
NUMBERS = {
    w: str(n)
    for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}


def normalize(text):
    """Нижний регистр, без пунктуации и артиклей; слова-числа (twelve) становятся цифрами."""
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(NUMBERS.get(w, w) for w in text.split())


def json_from(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("в ответе нет JSON")
    return json.loads(m.group(0), strict=False)


def final_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def is_correct(task, answer):
    """Все слова эталона есть в ответе (порядок не важен); для дат допускаем перестановку день/месяц.

    Слова-числа приводятся к цифрам: эталон «Twelve» совпадает с ответом «12 people killed».
    """
    pred_words = set(normalize(answer).split())
    gold = normalize(task["answer"])
    if not gold:
        return False
    if set(gold.split()) <= pred_words:
        return True
    months = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }
    pred = " ".join(normalize(answer).split())
    # "15 march" <-> "march 15"
    m = re.match(r"^(\d{1,2})\s+([a-z]+)$", gold)
    if m and m.group(2) in months:
        alt = f"{m.group(2)} {int(m.group(1))}"
        if set(alt.split()) <= pred_words or alt in pred:
            return True
    m = re.match(r"^([a-z]+)\s+(\d{1,2})$", gold)
    if m and m.group(1) in months:
        alt = f"{int(m.group(2))} {m.group(1)}"
        if set(alt.split()) <= pred_words or alt in pred:
            return True
    return False


def register(fn, args_model, description):
    TOOLS[fn.__name__] = {
        "fn": fn, "args": args_model,
        "schema": {"type": "function", "function": {
            "name": fn.__name__, "description": description,
            "parameters": args_model.model_json_schema(),
        }},
    }


# ---------- tools ----------
def wiki(params, attempts=3):
    for attempt in range(attempts):
        r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=20)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()["query"]
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия ответила {r.status_code}")


class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")


class PageFindArgs(BaseModel):
    title: str = Field(description="точный заголовок статьи Википедии, например '2026 in Japan'")
    keywords: str = Field(description="два–пять ключевых слов из вопроса")


class TavilyArgs(BaseModel):
    query: str = Field(description="поисковый запрос по свежим новостям и веб-страницам")


class CalcArgs(BaseModel):
    expr: str = Field(description="арифметическое выражение, например 17*23+5")


class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")


def web_search(query: str) -> str:
    try:
        hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["search"]
    except RuntimeError as e:
        return f"ошибка поиска: {e}"
    if not hits:
        return "nothing found"
    try:
        pages = wiki({
            "action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
            "titles": hits[0]["title"],
        })["pages"]
    except RuntimeError as e:
        return f"ошибка чтения статьи: {e}"
    text = " ".join(next(iter(pages.values())).get("extract", "").split())[:1500]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")


def page_find(title: str, keywords: str) -> str:
    try:
        pages = wiki({
            "action": "query", "prop": "extracts", "explaintext": 1,
            "titles": title, "redirects": 1,
        })["pages"]
    except RuntimeError as e:
        return f"ошибка чтения страницы: {e}"
    text = next(iter(pages.values())).get("extract", "")
    if not text:
        return "no such page"
    words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    if not words:
        return "keywords not found on the page"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = [l for l in lines if sum(w in l.lower() for w in words) >= max(1, (len(words) + 1) // 2)]
    return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"


def tavily_search(query: str) -> str:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return "ошибка: нет TAVILY_API_KEY"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": 5, "include_answer": False},
            timeout=30,
        )
    except requests.RequestException as e:
        return f"ошибка tavily: {type(e).__name__}"
    if r.status_code != 200:
        return f"ошибка tavily: HTTP {r.status_code}"
    results = (r.json() or {}).get("results") or []
    if not results:
        return "nothing found"
    parts = []
    for item in results[:5]:
        title = item.get("title") or ""
        url = item.get("url") or ""
        content = " ".join((item.get("content") or "").split())[:500]
        parts.append(f"[{title}]({url}) {content}")
    return "\n".join(parts)


OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod,
}


def evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.operand))
    raise ValueError("допустимы только числа и арифметика")


def calculator(expr: str) -> str:
    try:
        val = evaluate(ast.parse(expr.replace(",", "."), mode="eval").body)
        return str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip("0")
    except Exception as e:
        return f"ошибка вычисления: {e}"


def python_exec(code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        return "Код превысил 5 сек"
    out = (r.stdout + r.stderr).strip()
    return out[:5000] if out else "Код исполнился но результат пустой"


def setup_tools():
    register(web_search, SearchArgs, "Searches English Wikipedia and returns the intro of the best article")
    register(page_find, PageFindArgs, "Looks inside the full text of a Wikipedia article and returns lines with the keywords")
    register(tavily_search, TavilyArgs, "Searches the live web via Tavily (news and recent pages)")
    register(calculator, CalcArgs, "Evaluates an arithmetic expression")
    register(python_exec, ExecArgs, "Runs Python code in a subprocess and returns printed output")


# ---------- agent loop ----------
@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0


def run_tool(call):
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:2500]}


def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated


def finish(model, messages, step):
    del messages[-1]
    messages.append({
        "role": "user",
        "content": "Инструменты больше недоступны. Ответь по тому, что уже известно, последней строкой FINAL: <ответ>.",
    })
    msg = chat(messages, model, tag="agent")
    messages.append(msg)
    return msg.get("content") or "", step + 1


def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    for step in range(1, max_steps + 1):
        schemas = [TOOLS[n]["schema"] for n in tool_names] if tool_names else None
        msg = chat(messages, model, tools=schemas, tag="agent")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or "", step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)
        messages += [run_tool(c) for c in calls]
    return "", max_steps


def agent(question, model, tool_names, max_steps=8):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)
    return Run(question, answer, steps, messages, ledger.total - before, time.perf_counter() - started)


# ---------- evaluation ----------
def load_tasks(path=None):
    path = Path(path or ROOT / "data" / "fresh.jsonl")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def summary(config, model, n, correct, cost, steps, seconds=0.0):
    return {
        "config": config, "model": model.split("/")[-1], "n": n,
        "accuracy": round(correct / n, 2) if n else 0,
        "cost_per_task": round(cost / n, 5) if n else 0,
        "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
        "avg_steps": round(steps, 2), "avg_seconds": round(seconds, 1),
    }


def run_tasks(tasks, model, tool_names, config, max_steps=8):
    folder = TRACES / config.replace("/", "_") / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"  [{config}|{model.split('/')[-1]}] {i}/{len(tasks)} {task['id']}", flush=True)
        try:
            if tool_names:
                run = agent(task["question"], model, tool_names, max_steps=max_steps)
            else:
                before, started = ledger.total, time.perf_counter()
                msg = chat(
                    [{"role": "user", "content": (
                        "Answer with one short phrase only. If you do not know, write unknown.\n"
                        f"Question: {task['question']}"
                    )}],
                    model, tag="solo",
                )
                ans = msg.get("content") or ""
                run = Run(task["question"], f"FINAL: {ans}", 1, [msg], ledger.total - before, time.perf_counter() - started)
        except Exception as e:
            run = Run(task["question"], f"ошибка: {e}", 0, [])
        ok = is_correct(task, final_answer(run.answer))
        rows.append({
            "config": config, "model": model.split("/")[-1], "id": task["id"], "source": task.get("source", "fresh"),
            "correct": ok, "steps": run.steps,
            "tool_calls": sum(m.get("role") == "tool" for m in run.messages),
            "cost": run.cost, "seconds": round(run.seconds, 1),
            "answer": final_answer(run.answer)[:80], "gold": task["answer"],
        })
        (folder / f"{task['id']}.json").write_text(
            json.dumps({
                "task": task, "answer": run.answer, "correct": ok, "cost": run.cost,
                "messages": run.messages,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return pd.DataFrame(rows)


def report(results):
    rows = [
        summary(config, model, len(df), int(df["correct"].sum()), df["cost"].sum(),
                df["steps"].mean(), df["seconds"].mean())
        for (config, model), df in results.groupby(["config", "model"])
    ]
    return pd.DataFrame(rows).sort_values("cost_per_task").reset_index(drop=True)


def money_chart(table, path=None):
    path = Path(path or IMG / "money_chart.png")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in table.iterrows():
        color = COLORS["red"] if "каскад" in r["config"] else COLORS["amber"] if "sonnet" in r["model"] else COLORS["violet"]
        x = max(r["cost_per_task"] * 100, 1e-4)
        ax.scatter(x, r["accuracy"] * 100, s=110, color=color)
        ax.annotate(
            f"{r['config']}\n{r['model']}", (x, r["accuracy"] * 100),
            fontsize=7, xytext=(6, 4), textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("цена задачи, центы (лог)")
    ax.set_ylabel("доля верных, %")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return fig


# ---------- cascade (agent first, strong if low confidence) ----------
class Confident(BaseModel):
    reasoning: str = Field(description="brief reasoning")
    final: str = Field(description="short final answer only")
    confidence: float = Field(description="honest confidence from 0 to 1")


SCHEMA_PROMPT = (
    "Answer strictly with one JSON object matching this schema, no text around:\n{schema}"
)


def tool_evidence(messages, limit=4000):
    """Тексты ответов инструментов: без них оценка уверенности не видит найденные строки."""
    parts = []
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "tool" and message.get("content"):
            parts.append(str(message["content"])[:800])
    return "\n".join(parts)[-limit:]


def rate_confidence(question, answer, model, evidence=""):
    user = f"Question: {question}\nProposed answer: {answer}\n"
    if evidence:
        user += f"Evidence the agent retrieved:\n{evidence}\n"
    user += (
        "Rate your confidence that the proposed answer is correct. "
        "If the retrieved evidence explicitly supports it, confidence should be high."
    )
    messages = [
        {"role": "system", "content": SCHEMA_PROMPT.format(schema=json.dumps(Confident.model_json_schema(), ensure_ascii=False))},
        {"role": "user", "content": user},
    ]
    for _ in range(2):
        msg = chat(messages, model, tag="confidence")
        try:
            return Confident.model_validate(json_from(msg.get("content") or ""))
        except (ValidationError, ValueError) as e:
            messages += [msg, {"role": "user", "content": f"Invalid JSON: {e}. Return strict JSON."}]
    return Confident(reasoning="fail", final=answer, confidence=0.0)


def cascade_run(tasks, tool_names, threshold=0.55, config="каскад (агент→sonnet)"):
    folder = TRACES / config.replace("/", "_") / "cascade"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"  [{config}] {i}/{len(tasks)} {task['id']}", flush=True)
        before = ledger.total
        started = time.perf_counter()
        run = agent(task["question"], MODELS["cheap"], tool_names, max_steps=8)
        ans = final_answer(run.answer)
        conf = rate_confidence(task["question"], ans, MODELS["cheap"], tool_evidence(run.messages))
        escalated = False
        if conf.confidence < threshold:
            escalated = True
            solo = chat(
                [{"role": "user", "content": (
                    "Answer with one short phrase only. If you do not know, write unknown.\n"
                    f"Question: {task['question']}"
                )}],
                MODELS["strong"], tag="cascade-strong",
            )
            ans = (solo.get("content") or "").strip()
        ok = is_correct(task, ans)
        cost = ledger.total - before
        seconds = time.perf_counter() - started
        rows.append({
            "config": config, "model": "mini+sonnet", "id": task["id"], "source": task.get("source", "fresh"),
            "correct": ok, "steps": run.steps + (1 if escalated else 0),
            "tool_calls": sum(m.get("role") == "tool" for m in run.messages),
            "cost": cost, "seconds": round(seconds, 1),
            "answer": ans[:80], "gold": task["answer"],
            "escalated": escalated, "confidence": conf.confidence,
        })
        (folder / f"{task['id']}.json").write_text(
            json.dumps({
                "task": task, "answer": ans, "correct": ok, "cost": cost,
                "escalated": escalated, "confidence": conf.confidence, "agent_messages": run.messages,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    return pd.DataFrame(rows)


# ---------- tests ----------
def test_tools():
    setup_tools()
    # web_search: ok / empty / error-ish
    assert web_search("Scott Derrickson").startswith("[Scott Derrickson]"), web_search("Scott Derrickson")[:80]
    assert web_search("qwxzv 1234567 nonsense xyzzy") == "nothing found"
    # force error path via bogus by temporarily breaking is not needed; empty is the empty case
    print("web_search ok")

    # page_find: ok / empty page / keywords missing
    hit = page_find("2026 in sports", "Masters McIlroy Augusta")
    assert "McIlroy" in hit or "Masters" in hit, hit[:200]
    assert page_find("ThisPageDefinitelyDoesNotExistXYZ123", "foo bar") == "no such page"
    assert "keywords not found" in page_find("2026 in sports", "zzzzqqq nonexistenttoken")
    print("page_find ok")

    # tavily: with key → ok/empty; without → error
    if os.getenv("TAVILY_API_KEY", "").strip():
        out = tavily_search("2026 Masters Tournament winner")
        assert out and not out.startswith("ошибка"), out[:120]
        empty = tavily_search("xyzzyqwxzvnonexistenttoken99999")
        assert empty == "nothing found" or "http" in empty.lower() or "[" in empty
        print("tavily_search ok (with key)")
    else:
        assert "TAVILY_API_KEY" in tavily_search("test")
        print("tavily_search ok (missing key path)")

    assert calculator("2+2") == "4"
    assert "ошибка" in calculator("__import__('os')")
    assert python_exec("print(2+2)") == "4"
    assert "ZeroDivisionError" in python_exec("1/0")
    assert "5 сек" in python_exec("import time; time.sleep(10)")
    print("calculator / python_exec ok")
    print("все тесты инструментов пройдены")


def check_freshness(tasks):
    """Sonnet без инструментов; критерий ≤20%."""
    right = []
    for i, t in enumerate(tasks, 1):
        print(f"  freshness {i}/{len(tasks)} {t['id']}", flush=True)
        msg = chat(
            [{"role": "user", "content": (
                "Answer with one short phrase only. If you do not know, write unknown.\n"
                f"Question: {t['question']}"
            )}],
            MODELS["strong"], tag="freshness",
        )
        ans = msg.get("content") or ""
        if is_correct(t, ans):
            right.append((t["id"], t["question"], ans))
    share = len(right) / len(tasks) if tasks else 0
    print(f"\nSonnet без инструментов: {len(right)} из {len(tasks)} верно, {share:.0%}. Потрачено ${ledger.total:.3f}")
    print("критерий свежести " + ("пройден" if share <= 0.2 else "НЕ пройден"))
    for rid, q, ans in right:
        print(f"  {rid}: {q[:90]} -> {ans[:40]}")
    out = ROOT / "freshness_report.txt"
    lines = [
        f"Sonnet без инструментов: {len(right)} из {len(tasks)} верно, {share:.0%}",
        f"Потрачено ${ledger.total:.3f}",
        "критерий свежести " + ("пройден" if share <= 0.2 else "НЕ пройден"),
    ]
    for rid, q, ans in right:
        lines.append(f"  {rid}: {q} -> {ans}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return share <= 0.2


def run_eval(limit=0, include_tavily=True, include_cascade=True):
    setup_tools()
    tasks = load_tasks()
    if limit and limit > 0:
        # стабильная подвыборка: первые 40 стартовых + все свои 22 + добивка
        own = [t for t in tasks if int(t["id"].split("-")[1]) >= 125]
        base = [t for t in tasks if int(t["id"].split("-")[1]) < 125]
        need = max(0, limit - len(own))
        tasks = base[:need] + own
        print(f"подвыборка: {len(tasks)} задач (limit={limit})")
    else:
        print(f"полный набор: {len(tasks)} задач")

    frames = []
    best_tools = ["web_search", "page_find"]
    best_tools_tavily = ["web_search", "page_find", "tavily_search"]

    # 1) strong solo
    frames.append(run_tasks(tasks, MODELS["strong"], [], "сильная, без инструментов"))
    # 2) cheap solo
    frames.append(run_tasks(tasks, MODELS["cheap"], [], "дешёвая, без инструментов"))
    # 3) cheap search only
    frames.append(run_tasks(tasks, MODELS["cheap"], ["web_search"], "дешёвая, только поиск"))
    # 4) cheap search + page_find
    frames.append(run_tasks(tasks, MODELS["cheap"], best_tools, "дешёвая, поиск+чтение"))
    mid_tools = best_tools_tavily if include_tavily and os.getenv("TAVILY_API_KEY", "").strip() else best_tools
    mid_name = "средняя, поиск+чтение+tavily" if mid_tools is best_tools_tavily else "средняя, поиск+чтение"
    frames.append(run_tasks(tasks, MODELS["mid"], mid_tools, mid_name))

    if include_tavily and os.getenv("TAVILY_API_KEY", "").strip():
        frames.append(run_tasks(tasks, MODELS["cheap"], best_tools_tavily, "дешёвая, поиск+чтение+tavily"))
    elif include_tavily:
        print("TAVILY_API_KEY нет — пропускаю строку с Tavily")

    if include_cascade:
        frames.append(cascade_run(tasks, mid_tools, threshold=0.55))

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(ROOT / "results_raw.csv", index=False)
    table = report(results)
    (ROOT / "results.md").write_text(table.to_string(index=False), encoding="utf-8")
    money_chart(table)
    print(table.to_string(index=False))
    print(f"итого потрачено ${ledger.total:.3f}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-tools", action="store_true")
    ap.add_argument("--freshness", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = весь датасет")
    ap.add_argument("--no-tavily", action="store_true")
    ap.add_argument("--no-cascade", action="store_true")
    args = ap.parse_args()

    if args.test_tools:
        test_tools()
        return
    if args.freshness:
        ok = check_freshness(load_tasks())
        sys.exit(0 if ok else 1)
    if args.eval:
        run_eval(limit=args.limit, include_tavily=not args.no_tavily, include_cascade=not args.no_cascade)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
