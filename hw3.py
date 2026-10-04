# -*- coding: utf-8 -*-
"""ДЗ3: счётчик HTTP, LangGraph, критик, голосование, план, потолок, HITL.

Запуск:
  python hw3.py --eval --limit 40
  python hw3.py --probe
  python hw3.py --demo-limit
  python hw3.py --demo-hitl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
warnings.filterwarnings("ignore")

import httpx
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
BASE_URL = "https://openrouter.ai/api/v1"
CHEAP = "openai/gpt-4o-mini"
STRONG = "anthropic/claude-sonnet-4.6"
COLORS = {
    "violet": "#5436A3",
    "amber": "#F09000",
    "teal": "#00838F",
    "red": "#C43C3C",
    "grey": "#787882",
}
WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "agents-course-hw03/1.0 (https://postypashki.ru; educational)",
}
IMG = ROOT / "img"
ART = ROOT / "artifacts_hw3"
IMG.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)
CACHE_PATH = ROOT / "data" / "wiki_cache_hw3.json"

CALLS: list[dict] = []
FIRST: dict = {}
TAG = {"now": "без метки"}
RESULTS: dict[str, list] = {}
ESCALATED: list[str] = []
WIKI_CACHE: dict = {}

AGENT_SYSTEM = (
    "You answer questions about events of 2026 using Wikipedia tools. Never answer from memory, "
    "do not repeat the same call. When done, reply with the short answer only, no explanation."
)
LAST_STEP = "No more tool calls. Answer now with what you found."
PROBE = "What is 2 + 2? Reply with the number only."
PLAN_SYSTEM = (
    "Write a short numbered plan (two to four steps) to answer the question with two Wikipedia tools: "
    "web_search finds an article, page_find looks for keywords inside an article. "
    "The dataset contains questions about 2026+; you may not know these facts. "
    "Plan only, no answer."
)
CRITIC_SYSTEM = (
    "You check a short answer against the evidence the agent retrieved from Wikipedia. "
    "Judge only by the evidence. If the evidence does not contain the answer, the answer is not supported."
)
NUMBERS = {
    w: str(n)
    for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())
}
MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
}


# ---------- HTTP counter ----------
class CompletionsClient(httpx.Client):
    """Injects OpenRouter usage.include so every framework reports cost."""

    def post(self, url, **kwargs):
        body = kwargs.get("json")
        if body is not None and "chat/completions" in str(url) and isinstance(body, dict):
            usage = body.get("usage")
            if not isinstance(usage, dict) or not usage.get("include"):
                body = {**body, "usage": {"include": True}}
                kwargs["json"] = body
        return super().post(url, **kwargs)


def on_response(response):
    try:
        response.read()
    except Exception:
        return
    if "chat/completions" not in str(response.request.url):
        return
    try:
        request = json.loads(response.request.content or b"{}")
        payload = response.json()
        usage = payload.get("usage") or {}
    except Exception:
        return
    FIRST.setdefault(TAG["now"], request)
    CALLS.append({
        "tag": TAG["now"],
        "model": str(request.get("model", "?")).split("/")[-1],
        "messages": len(request.get("messages") or []),
        "tools": len(request.get("tools") or []),
        "prompt": usage.get("prompt_tokens") or 0,
        "completion": usage.get("completion_tokens") or 0,
        "cost": usage.get("cost") or 0,
    })


def require_key():
    if not KEY:
        raise SystemExit("положите OPENROUTER_API_KEY в .env (см. .env.example)")
    return KEY


HTTP = CompletionsClient(event_hooks={"response": [on_response]}, timeout=180)
HEADERS = {
    "Authorization": f"Bearer {KEY}",
    "HTTP-Referer": "https://postypashki.ru",
    "X-Title": "agents-course-hw03",
}


def chat(messages, model=CHEAP, tools=None, temperature=0.0):
    body = {"model": model, "messages": messages, "temperature": temperature, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    last = "нет ответа"
    for attempt in range(5):
        try:
            r = HTTP.post(f"{BASE_URL}/chat/completions", json=body, headers=HEADERS)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504, 403):
                break
        except httpx.HTTPError as e:
            last = type(e).__name__
        time.sleep(min(60, 2 * 2 ** attempt))
    raise RuntimeError(last)


def spent() -> float:
    return sum(c["cost"] for c in CALLS)


# ---------- wiki tools ----------
def load_wiki_cache():
    global WIKI_CACHE
    if CACHE_PATH.exists():
        WIKI_CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_wiki_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(WIKI_CACHE, ensure_ascii=False), encoding="utf-8")


def wiki(params):
    key = json.dumps(params, sort_keys=True)
    error = None
    for attempt in range(4):
        if key not in WIKI_CACHE:
            try:
                r = httpx.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=25)
                r.raise_for_status()
                WIKI_CACHE[key] = r.json().get("query", {})
            except httpx.HTTPError as e:
                error = e
                time.sleep(2 * (attempt + 1))
                continue
        return WIKI_CACHE[key]
    raise RuntimeError(f"Википедия не отвечает: {error}")


def web_search(query: str) -> str:
    """Search English Wikipedia: returns the intro of the best article and the titles of two more articles."""
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3}).get("search", [])
    if not hits:
        return "nothing found"
    pages = wiki({
        "action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1,
        "titles": hits[0]["title"],
    }).get("pages", {})
    intro = " ".join(next(iter(pages.values()), {}).get("extract", "").split())[:1200]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {intro}" + (f" | other articles: {others}" if others else "")


def page_find(title: str, keywords: str) -> str:
    """Look inside the full text of a Wikipedia article and return the lines that contain the keywords."""
    pages = wiki({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "redirects": 1,
    }).get("pages", {})
    text = next(iter(pages.values()), {}).get("extract", "")
    if not text:
        return "no such page"
    kws = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    if not kws:
        return "keywords not found on the page"
    lines = [
        l.strip() for l in text.splitlines()
        if l.strip() and sum(w in l.lower() for w in kws) >= max(1, (len(kws) + 1) // 2)
    ]
    return "\n".join(l[:350] for l in lines[:5]) or "keywords not found on the page"


BARE_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": web_search.__doc__,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "short query: a name, a title, an event"}},
            "required": ["query"],
        },
    }},
    {"type": "function", "function": {
        "name": "page_find",
        "description": page_find.__doc__,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "exact article title, e.g. 2026 in Japan"},
                "keywords": {"type": "string", "description": "two to five keywords"},
            },
            "required": ["title", "keywords"],
        },
    }},
]
FUNCS = {"web_search": web_search, "page_find": page_find}


# ---------- scoring ----------
def load_tasks(path=None, limit=40):
    path = Path(path or ROOT / "data" / "fresh.jsonl")
    tasks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return tasks[:limit] if limit else tasks


def norm(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(NUMBERS.get(w, w) for w in text.split())


def date_alts(gold: str):
    alts = [gold]
    m = re.match(r"^(\d{1,2})\s+([a-z]+)$", gold)
    if m and m.group(2) in MONTHS:
        alts.append(f"{m.group(2)} {int(m.group(1))}")
    m = re.match(r"^([a-z]+)\s+(\d{1,2})$", gold)
    if m and m.group(1) in MONTHS:
        alts.append(f"{int(m.group(2))} {m.group(1)}")
    return alts


def is_correct(task, answer) -> bool:
    pred = set(norm(answer).split())
    gold = norm(task["answer"])
    if not gold:
        return False
    for alt in date_alts(gold):
        if set(alt.split()) <= pred:
            return True
    return False


def bag(answer: str) -> str:
    return " ".join(sorted(norm(answer).split()))


# ---------- bare loop ----------
def bare_agent(question, model=CHEAP, max_steps=6):
    messages, seen = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}], set()
    for step in range(max_steps):
        last = step == max_steps - 1
        if last:
            messages.append({"role": "user", "content": LAST_STEP})
        msg = chat(messages, model, None if last else BARE_TOOLS)
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v})
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content") or ""
        for c in calls:
            key = (c["function"]["name"], c["function"]["arguments"])
            if key in seen:
                result = "repeated call, answer now"
            else:
                try:
                    args = json.loads(key[1] or "{}")
                    result = FUNCS[key[0]](**args)
                except Exception as e:
                    result = f"tool error: {e}"
            seen.add(key)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)[:2500]})
    return messages[-1].get("content") or ""


# ---------- LangGraph ----------
LG_READY = False
LC_TOOLS_RAW = None
LC_TOOLS = None
LG_MIDDLEWARE: list = []


def _init_langgraph():
    global LG_READY, LC_TOOLS_RAW, LC_TOOLS, wrap_model_call, HumanMessage
    global ChatOpenAI, create_agent, tool, SearchArgs, FindArgs
    if LG_READY:
        return
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    from langchain.agents import create_agent as _create_agent
    from langchain_core.tools import tool as _tool
    from langchain.agents.middleware import wrap_model_call as _wrap
    from langchain_core.messages import HumanMessage as _HumanMessage

    ChatOpenAI = _ChatOpenAI
    create_agent = _create_agent
    tool = _tool
    wrap_model_call = _wrap
    HumanMessage = _HumanMessage

    class SearchArgs(BaseModel):
        query: str = Field(description="short query: a name, a title, an event")

    class FindArgs(BaseModel):
        title: str = Field(description="exact article title, e.g. 2026 in Japan")
        keywords: str = Field(description="two to five keywords")

    globals()["SearchArgs"] = SearchArgs
    globals()["FindArgs"] = FindArgs
    LC_TOOLS_RAW = [_tool(web_search), _tool(page_find)]
    LC_TOOLS = [_tool(web_search, args_schema=SearchArgs), _tool(page_find, args_schema=FindArgs)]
    LG_READY = True


def lc_model(model=CHEAP, temperature=0.0):
    _init_langgraph()
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=require_key(),
        http_client=HTTP,
        temperature=temperature,
        extra_body={"usage": {"include": True}},
    )


def last_step_without_tools(request, handler):
    if sum(getattr(m, "type", None) == "ai" for m in request.messages) >= 5:
        request = request.override(tools=[], messages=list(request.messages) + [HumanMessage(LAST_STEP)])
    return handler(request)


def patched_middleware():
    _init_langgraph()
    if not LG_MIDDLEWARE:
        LG_MIDDLEWARE.append(wrap_model_call(last_step_without_tools))
    return LG_MIDDLEWARE


def run_langgraph(question, model=CHEAP, temperature=0.0, extra="", raw=False):
    _init_langgraph()
    tools = LC_TOOLS_RAW if raw else LC_TOOLS
    middleware = [] if raw else patched_middleware()
    agent = create_agent(
        lc_model(model, temperature),
        tools=tools,
        system_prompt=AGENT_SYSTEM + extra,
        middleware=middleware,
    )
    out = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": 14},
    )
    return out["messages"]


def ask_langgraph(question, model=CHEAP, temperature=0.0, extra="", raw=False):
    messages = run_langgraph(question, model, temperature, extra, raw=raw)
    last = messages[-1]
    content = getattr(last, "content", None) or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def ask_langgraph_raw(question):
    return ask_langgraph(question, raw=True)


# ---------- smolagents (probe only) ----------
def ask_smol(question, kind="tools", model=CHEAP):
    from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool as smol_tool

    @smol_tool
    def smol_web_search(query: str) -> str:
        """Search English Wikipedia: returns the intro of the best article.

        Args:
            query: short query: a name, a title, an event
        """
        return web_search(query)

    @smol_tool
    def smol_page_find(title: str, keywords: str) -> str:
        """Look inside the full text of a Wikipedia article and return the lines that contain the keywords.

        Args:
            title: exact article title, e.g. 2026 in Japan
            keywords: two to five keywords
        """
        return page_find(title, keywords)

    model_obj = OpenAIServerModel(
        model_id=model,
        api_base=BASE_URL,
        api_key=require_key(),
        client_kwargs={"http_client": HTTP},
        temperature=0,
    )
    cls = ToolCallingAgent if kind == "tools" else CodeAgent
    agent = cls(
        tools=[smol_web_search, smol_page_find],
        model=model_obj,
        instructions=AGENT_SYSTEM,
        max_steps=10,
        verbosity_level=0,
    )
    return str(agent.run(question))


# ---------- critic graph ----------
class QA(TypedDict):
    question: str
    answer: str
    evidence: str
    feedback: str
    rounds: int
    checks: int


class Verdict(BaseModel):
    supported: bool = Field(description="true only if the evidence explicitly supports the answer")
    issue: str = Field(description="what is wrong or missing, empty if supported")


_checked = None


def answer_node(state):
    note = ""
    if state["feedback"]:
        note = (
            f"\n\nA reviewer rejected your previous answer '{state['answer']}': {state['feedback']} "
            "Search again and correct it."
        )
    messages = run_langgraph(state["question"] + note)
    found = "\n".join(
        str(getattr(m, "content", ""))[:800]
        for m in messages
        if getattr(m, "type", None) == "tool"
    )
    last = messages[-1]
    content = getattr(last, "content", None) or ""
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return {"answer": str(content), "evidence": found[-4000:], "rounds": state["rounds"] + 1}


def critic_node(state):
    judge = lc_model(CHEAP).with_structured_output(Verdict)
    verdict = judge.invoke([
        ("system", CRITIC_SYSTEM),
        ("user", (
            f"Question: {state['question']}\nAnswer: {state['answer']}\n"
            f"Evidence the agent found:\n{state['evidence'] or 'nothing'}"
        )),
    ])
    return {"feedback": "" if verdict.supported else verdict.issue, "checks": state["checks"] + 1}


def route(state):
    from langgraph.graph import END
    if state["feedback"] and state["rounds"] < 2:
        return "answer"
    return END


def compile_critic():
    global _checked
    if _checked is not None:
        return _checked
    from langgraph.graph import StateGraph, START, END
    graph = StateGraph(QA)
    graph.add_node("answer", answer_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", "critic")
    graph.add_conditional_edges("critic", route, ["answer", END])
    _checked = graph.compile()
    return _checked


def ask_checked(question):
    graph = compile_critic()
    out = graph.invoke({
        "question": question, "answer": "", "evidence": "",
        "feedback": "", "rounds": 0, "checks": 0,
    })
    return out["answer"]


def critic_mermaid() -> str:
    return compile_critic().get_graph().draw_mermaid()


# ---------- vote / cascade / plan ----------
def vote(question, k=3):
    with ThreadPoolExecutor(k) as ex:
        answers = list(ex.map(lambda _: ask_langgraph(question, CHEAP, 0.7), range(k)))
    keys = [bag(a) for a in answers]
    top, n = Counter(keys).most_common(1)[0]
    return {"answer": answers[keys.index(top)], "agreement": n / k, "votes": answers}


def cascade(question, votes=None):
    v = votes or vote(question, 3)
    if v["agreement"] == 1:
        return v["answer"], False, v
    ESCALATED.append(question)
    return ask_langgraph(question, STRONG), True, v


def plan_then_act(question):
    plan = lc_model(CHEAP).invoke([("system", PLAN_SYSTEM), ("user", question)]).content
    extra = "\nFollow this plan:\n" + str(plan)
    return ask_langgraph(question, CHEAP, extra=extra)


# ---------- eval harness ----------
def pmap(fn, items, workers=1):
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def run_config(tag, fn, sample=None, workers=1):
    sample = sample if sample is not None else tasks_default()
    TAG["now"] = tag
    print(f"\n=== {tag} n={len(sample)} ===", flush=True)

    def one(item):
        i, task = item
        print(f"  [{tag}] {i}/{len(sample)} {task['id']}", flush=True)
        try:
            return fn(task["question"])
        except Exception as e:
            return f"ошибка: {type(e).__name__}: {e}"

    answers = pmap(one, list(enumerate(sample, 1)), workers)
    RESULTS[tag] = [
        {"id": t["id"], "question": t["question"], "gold": t["answer"], "answer": a, "ok": is_correct(t, a)}
        for t, a in zip(sample, answers)
    ]
    save_artifacts()
    return report([tag])


def _calls_for(tag):
    calls = [c for c in CALLS if c["tag"] == tag]
    # каскад считает голоса + эскалацию: голоса пишутся под отдельной меткой
    if tag.startswith("каскад"):
        vote_tag = next((t for t in RESULTS if t.startswith("голосование")), None)
        if vote_tag:
            calls = [c for c in CALLS if c["tag"] == vote_tag] + calls
    return calls


def report(tags=None) -> pd.DataFrame:
    rows = []
    for tag in tags or list(RESULTS):
        res = RESULTS[tag]
        calls = _calls_for(tag)
        n, ok = len(res), sum(r["ok"] for r in res)
        cost = sum(c["cost"] for c in calls)
        tokens = sum(c["prompt"] + c["completion"] for c in calls)
        rows.append({
            "конфигурация": tag,
            "верно": f"{ok}/{n}",
            "доля": round(ok / n, 3) if n else 0,
            "вызовов на вопрос": round(len(calls) / n, 1) if n else 0,
            "токенов на вопрос": round(tokens / n) if n else 0,
            "центов на вопрос": round(cost / n * 100, 4) if n else 0,
            "центов за верный": round(cost / ok * 100, 4) if ok else None,
        })
    return pd.DataFrame(rows)


def wrong(tag, n=10) -> pd.DataFrame:
    rows = [
        {"id": r["id"], "эталон": r["gold"], "ответ": str(r["answer"])[:160], "вопрос": r.get("question", "")[:120]}
        for r in RESULTS.get(tag, []) if not r["ok"]
    ]
    return pd.DataFrame(rows[:n])


def color_of(tag):
    if "каскад" in tag:
        return COLORS["red"]
    if "Sonnet" in tag or "sonnet" in tag:
        return COLORS["amber"]
    if "smolagents" in tag:
        return COLORS["teal"]
    return COLORS["violet"]


def money_chart(tags, path=None, title="ДЗ3: цена вопроса против доли верных"):
    path = Path(path or IMG / "money_hw3.png")
    table = report(tags)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xscale("log")
    for _, r in table.iterrows():
        x = max(r["центов на вопрос"], 1e-4)
        ax.scatter(x, r["доля"] * 100, s=110, color=color_of(r["конфигурация"]), zorder=3)
    ax.margins(x=0.18, y=0.14)
    fig.canvas.draw()
    placed = []
    for _, r in table.sort_values("центов на вопрос").iterrows():
        x = max(r["центов на вопрос"], 1e-4)
        box = None
        for dx, dy in [(8, 4), (8, -12), (-8, 4), (-8, -12), (8, 16), (8, -24)]:
            label = ax.annotate(
                r["конфигурация"], (x, r["доля"] * 100),
                xytext=(dx, dy), textcoords="offset points", fontsize=8,
                ha="left" if dx > 0 else "right", color=color_of(r["конфигурация"]),
            )
            box = label.get_window_extent()
            if not any(box.overlaps(b) for b in placed) or dy == -24:
                break
            label.remove()
        if box is not None:
            placed.append(box)
    ax.set_xlabel("цена вопроса, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных, %")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return table


def tasks_default():
    return getattr(run_eval, "tasks", None) or load_tasks(limit=40)


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8")) or default
    except json.JSONDecodeError:
        return default


def save_artifacts():
    """Пишет артефакты, не затирая более полный прошлый прогон пустым процессом."""
    save_wiki_cache()
    ART.mkdir(exist_ok=True)

    old_calls = _read_json(ART / "calls.json", [])
    if CALLS and (not old_calls or len(CALLS) >= len(old_calls)):
        (ART / "calls.json").write_text(json.dumps(CALLS, ensure_ascii=False, indent=1), encoding="utf-8")
    elif CALLS and old_calls:
        old_tags = {c.get("tag") for c in old_calls}
        extra = [c for c in CALLS if c.get("tag") not in old_tags]
        if extra:
            (ART / "calls.json").write_text(json.dumps(old_calls + extra, ensure_ascii=False, indent=1), encoding="utf-8")

    compact_first = _read_json(ART / "first.json", {})
    for tag, req in FIRST.items():
        compact_first[tag] = {
            "model": req.get("model"),
            "n_messages": len(req.get("messages") or []),
            "n_tools": len(req.get("tools") or []),
            "system": _system_text(req)[:2500],
            "tool_params": _tool_params(req),
            "prompt_tokens_first": next((c["prompt"] for c in CALLS if c["tag"] == tag), None),
        }
    if compact_first:
        (ART / "first.json").write_text(json.dumps(compact_first, ensure_ascii=False, indent=1), encoding="utf-8")

    old_results = _read_json(ART / "results.json", {})
    if RESULTS:
        old_results.update(RESULTS)
        (ART / "results.json").write_text(json.dumps(old_results, ensure_ascii=False, indent=1), encoding="utf-8")

    table = report()
    if not table.empty:
        table.to_csv(ART / "table.csv", index=False)
        (ART / "table.md").write_text(table.to_string(index=False), encoding="utf-8")
    if ESCALATED:
        (ART / "escalated.json").write_text(json.dumps(ESCALATED, ensure_ascii=False, indent=1), encoding="utf-8")
    if RESULTS:
        wrongs = {tag: wrong(tag, 10).to_dict(orient="records") for tag in RESULTS}
        (ART / "wrong10.json").write_text(json.dumps(wrongs, ensure_ascii=False, indent=1), encoding="utf-8")


def _system_text(req) -> str:
    messages = req.get("messages") or []
    if not messages:
        return ""
    content = messages[0].get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _tool_params(req):
    tools = req.get("tools") or []
    out = {}
    for t in tools:
        fn = (t.get("function") or {})
        name = fn.get("name")
        if name:
            out[name] = fn.get("parameters")
    return out


def load_artifacts():
    global CALLS, FIRST, RESULTS, ESCALATED
    loaded_calls = _read_json(ART / "calls.json", [])
    if loaded_calls:
        CALLS[:] = loaded_calls
    loaded_results = _read_json(ART / "results.json", {})
    if loaded_results:
        RESULTS.update(loaded_results)
    esc = _read_json(ART / "escalated.json", [])
    if esc:
        ESCALATED[:] = esc
    table = report() if RESULTS else pd.DataFrame()
    if table.empty and (ART / "table.csv").exists():
        table = pd.read_csv(ART / "table.csv")
    return {
        "calls": CALLS,
        "results": RESULTS,
        "table": table,
        "first": _read_json(ART / "first.json", {}),
        "wrong": _read_json(ART / "wrong10.json", {}),
        "escalated": ESCALATED,
    }


# ---------- probe / demos ----------
def probe_empty():
    load_wiki_cache()
    require_key()
    for tag, fn in [
        ("пустой вопрос: голый цикл", bare_agent),
        ("пустой вопрос: LangGraph", ask_langgraph),
        ("пустой вопрос: smolagents, инструменты", lambda q: ask_smol(q, "tools")),
        ("пустой вопрос: smolagents, код", lambda q: ask_smol(q, "code")),
    ]:
        TAG["now"] = tag
        print(f"probe {tag}...", flush=True)
        try:
            print(" ", fn(PROBE)[:200], flush=True)
        except Exception as e:
            print(f"  ошибка: {type(e).__name__}: {e}", flush=True)
    rows = [c for c in CALLS if str(c["tag"]).startswith("пустой")]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    summary = table.groupby("tag", sort=False).agg(
        вызовов=("prompt", "size"),
        первый_запрос=("prompt", "first"),
        всего_токенов=("prompt", lambda s: int(s.sum() + table.loc[s.index, "completion"].sum()) if False else int(s.sum())),
        центов=("cost", lambda s: round(s.sum() * 100, 4)),
    )
    # fix total tokens = prompt+completion per group
    totals = table.groupby("tag", sort=False).apply(
        lambda g: int((g["prompt"] + g["completion"]).sum()), include_groups=False
    )
    summary["всего_токенов"] = totals
    (ART / "empty_probe.csv").write_text(summary.to_csv(), encoding="utf-8")
    save_artifacts()
    print(summary.to_string())
    return summary


def demo_limit(question=None):
    _init_langgraph()
    from langchain.agents.middleware import ModelCallLimitMiddleware
    question = question or (
        "Compare the months of the 2026 Winter Olympics and the 2026 FIFA World Cup, "
        "then list three events of each."
    )
    TAG["now"] = "бюджет"
    before = sum(1 for c in CALLS if c["tag"] == "бюджет")
    capped = create_agent(
        lc_model(CHEAP),
        tools=LC_TOOLS,
        system_prompt=AGENT_SYSTEM,
        middleware=[ModelCallLimitMiddleware(run_limit=2, exit_behavior="end")],
    )
    out = capped.invoke({"messages": [{"role": "user", "content": question}]})
    last = out["messages"][-1]
    content = getattr(last, "content", "") or ""
    n_calls = sum(1 for c in CALLS if c["tag"] == "бюджет") - before
    payload = {"question": question, "answer": str(content)[:800], "model_calls": n_calls}
    (ART / "limit_demo.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    save_artifacts()
    print("потолок:", payload)
    return payload


def demo_hitl(question=None):
    _init_langgraph()
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    @tool
    def send_summary(to: str, text: str) -> str:
        """Send a short summary to a colleague by email. Irreversible."""
        return f"sent to {to}: {text}"

    question = question or (
        "Find who won the 2026 FIFA World Cup final and email a one-line summary to boss@example.com"
    )
    TAG["now"] = "человек в контуре"
    guarded = create_agent(
        lc_model(CHEAP),
        tools=LC_TOOLS + [send_summary],
        system_prompt=AGENT_SYSTEM.replace(
            " reply with the short answer only, no explanation",
            " report what you did",
        ),
        checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_summary": True})],
    )
    thread = {"configurable": {"thread_id": "demo-1"}, "recursion_limit": 14}
    steps = []
    out = guarded.invoke({"messages": [{"role": "user", "content": question}]}, thread)
    interrupt = out.get("__interrupt__")
    first_req = None
    if interrupt:
        first_req = interrupt[0].value.get("action_requests")
        steps.append({"stage": "interrupt", "action_requests": _jsonish(first_req)})
        out = guarded.invoke(
            Command(resume={"decisions": [{"type": "reject", "message": "Добавь в письмо счёт финала"}]}),
            thread,
        )
        interrupt2 = out.get("__interrupt__")
        if interrupt2:
            steps.append({"stage": "reject_then_interrupt", "action_requests": _jsonish(interrupt2[0].value.get("action_requests"))})
            out = guarded.invoke(Command(resume={"decisions": [{"type": "approve"}]}), thread)
        steps.append({"stage": "approve", "answer": str(getattr(out["messages"][-1], "content", ""))[:500]})
    else:
        steps.append({"stage": "no_interrupt", "answer": str(getattr(out["messages"][-1], "content", ""))[:500]})
    payload = {"question": question, "steps": steps}
    (ART / "hitl_demo.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    save_artifacts()
    print("HITL:", json.dumps(payload, ensure_ascii=False)[:800])
    return payload


def _jsonish(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


# ---------- full eval ----------
def critic_delta(base_tag="LangGraph, mini", crit_tag="LangGraph + критик, mini"):
    base, crit = RESULTS.get(base_tag, []), RESULTS.get(crit_tag, [])
    by_id = {r["id"]: r for r in crit}
    fixed = broken = 0
    details = []
    for b in base:
        c = by_id.get(b["id"])
        if not c:
            continue
        if (not b["ok"]) and c["ok"]:
            fixed += 1
            details.append({"id": b["id"], "kind": "fixed", "before": b["answer"], "after": c["answer"]})
        if b["ok"] and (not c["ok"]):
            broken += 1
            details.append({"id": b["id"], "kind": "broken", "before": b["answer"], "after": c["answer"]})
    payload = {"fixed": fixed, "broken": broken, "details": details}
    (ART / "critic_delta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def run_vote_and_cascade(sample, workers=1):
    """One pass: 3 votes, then Sonnet only on disagreement. Fills both table rows."""
    vote_tag = "голосование 3, mini"
    casc_tag = "каскад: 3 голоса mini, иначе Sonnet"
    TAG["now"] = vote_tag
    print(f"\n=== {vote_tag} / {casc_tag} n={len(sample)} ===", flush=True)
    vote_rows, casc_rows = [], []
    ESCALATED.clear()
    for i, task in enumerate(sample, 1):
        print(f"  [vote+cascade] {i}/{len(sample)} {task['id']}", flush=True)
        TAG["now"] = vote_tag
        try:
            v = vote(task["question"], 3)
            vote_ans = v["answer"]
        except Exception as e:
            v = {"answer": f"ошибка: {type(e).__name__}", "agreement": 0, "votes": []}
            vote_ans = v["answer"]
        vote_rows.append({
            "id": task["id"], "question": task["question"], "gold": task["answer"],
            "answer": vote_ans, "ok": is_correct(task, vote_ans), "agreement": v.get("agreement"),
        })
        TAG["now"] = casc_tag
        try:
            if v.get("agreement") == 1:
                casc_ans, escalated = vote_ans, False
            else:
                ESCALATED.append(task["question"])
                casc_ans, escalated = ask_langgraph(task["question"], STRONG), True
        except Exception as e:
            casc_ans, escalated = f"ошибка: {type(e).__name__}", True
        casc_rows.append({
            "id": task["id"], "question": task["question"], "gold": task["answer"],
            "answer": casc_ans, "ok": is_correct(task, casc_ans), "escalated": escalated,
        })
        RESULTS[vote_tag] = vote_rows
        RESULTS[casc_tag] = casc_rows
        save_artifacts()
    (ART / "escalated.json").write_text(json.dumps(ESCALATED, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ушло наверх:", len(ESCALATED), "из", len(sample))
    return report([vote_tag, casc_tag])


def run_eval(limit=40, skip_raw=False, skip_sonnet=False, skip_vote=False, workers=1):
    require_key()
    load_wiki_cache()
    tasks = load_tasks(limit=limit)
    run_eval.tasks = tasks
    print(f"набор: {len(tasks)} вопросов, кэш wiki={len(WIKI_CACHE)}", flush=True)

    run_config("голый цикл, mini", bare_agent, tasks, workers=workers)
    if not skip_raw:
        run_config("LangGraph как есть, mini", ask_langgraph_raw, tasks, workers=workers)
    run_config("LangGraph, mini", ask_langgraph, tasks, workers=workers)
    run_config("LangGraph + критик, mini", ask_checked, tasks, workers=workers)
    if not skip_vote:
        run_vote_and_cascade(tasks, workers=workers)
    run_config("план + LangGraph, mini", plan_then_act, tasks, workers=workers)
    if not skip_sonnet:
        run_config("LangGraph, Sonnet", lambda q: ask_langgraph(q, STRONG), tasks, workers=workers)

    delta = critic_delta()
    print("критик исправил", delta["fixed"], "| испортил", delta["broken"])
    tags = [t for t in [
        "голый цикл, mini", "LangGraph как есть, mini", "LangGraph, mini",
        "LangGraph + критик, mini", "голосование 3, mini",
        "каскад: 3 голоса mini, иначе Sonnet", "план + LangGraph, mini", "LangGraph, Sonnet",
    ] if t in RESULTS]
    table = money_chart(tags)
    save_artifacts()
    print(table.to_string(index=False))
    print(f"итого ${spent():.3f}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--demo-limit", action="store_true")
    ap.add_argument("--demo-hitl", action="store_true")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-sonnet", action="store_true")
    ap.add_argument("--skip-vote", action="store_true")
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args()

    load_wiki_cache()
    if args.probe:
        probe_empty()
    if args.demo_limit:
        demo_limit()
    if args.demo_hitl:
        demo_hitl()
    if args.eval:
        run_eval(
            limit=args.limit, skip_raw=args.skip_raw, skip_sonnet=args.skip_sonnet,
            skip_vote=args.skip_vote, workers=args.workers,
        )
        if not args.probe:
            probe_empty()
        if not args.demo_limit:
            demo_limit()
        if not args.demo_hitl:
            demo_hitl()
    if args.chart:
        load_artifacts()
        tags = [t for t in RESULTS]
        money_chart(tags)
        print(report().to_string(index=False))
    if not any([args.eval, args.probe, args.demo_limit, args.demo_hitl, args.chart]):
        ap.print_help()


if __name__ == "__main__":
    main()
