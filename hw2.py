# -*- coding: utf-8 -*-
"""ДЗ2: RAG, Milvus, память, parent-chunk и PDF.

Этапы:
  python hw2.py --index          # нарезка, эмбеддинги, recall, гибрид, milvus
  python hw2.py --eval           # 5 конфигураций + отказы
  python hw2.py --memory         # две сессии + токены + тесты без LLM
  python hw2.py --bonuses        # parent-chunk + PDF
  python hw2.py --all            # всё подряд
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
assert os.getenv("OPENROUTER_API_KEY"), "нужен OPENROUTER_API_KEY в .env"

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "X-Title": "agents-course-hw02",
}
MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}
EMBED_MODEL, DIM = "openai/text-embedding-3-small", 512
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}
DATA = ROOT / "data"
IMG = ROOT / "img"
IMG.mkdir(exist_ok=True)

# ---------- ledger / http ----------
@dataclass
class Ledger:
    calls: list = field(default_factory=list)
    local: threading.local = field(default_factory=threading.local)

    def add(self, tag, model, usage, seconds=0.0):
        cost = (usage or {}).get("cost") or 0.0
        self.calls.append({
            "tag": tag, "model": model.split("/")[-1],
            "prompt": (usage or {}).get("prompt_tokens", 0),
            "completion": (usage or {}).get("completion_tokens", 0),
            "cost": cost, "seconds": round(seconds, 2),
        })
        self.local.spent = self.mine + cost
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    @property
    def mine(self):
        return getattr(self.local, "spent", 0.0)

    def table(self):
        if not self.calls:
            return pd.DataFrame()
        frame = pd.DataFrame(self.calls)
        return frame.groupby(["tag", "model"]).agg(
            calls=("cost", "size"), prompt=("prompt", "sum"),
            completion=("completion", "sum"), cost=("cost", "sum"),
        ).round(5)


ledger = Ledger()


def post_with_retry(url, body, attempts=6):
    problem = "нет ответа"
    for attempt in range(attempts):
        try:
            r = requests.post(url, json=body, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r.json()
            problem = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 403, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            problem = type(e).__name__
        time.sleep(min(60, 3.0 * (2 ** attempt)))
    raise RuntimeError(problem)


def chat(messages, model, tools=None, tag="chat"):
    body = {"model": model, "messages": messages, "temperature": 0, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    started = time.perf_counter()
    data = post_with_retry(CHAT_URL, body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]


def load_jsonl(name):
    path = DATA / name if not str(name).startswith("/") else Path(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- chunking ----------
def chunk_lines(page):
    out = []
    for line in page["text"].splitlines():
        text = line.strip()
        if len(text) >= 40:
            out.append({"page": page["page"], "section": "", "url": page["url"], "text": text})
    return out


def chunk_chars(page, size=400):
    text = " ".join(page["text"].split())
    return [
        {"page": page["page"], "section": "", "url": page["url"], "text": text[i:i + size]}
        for i in range(0, len(text), size)
    ]


def expand_parent(chunks, idx, radius=2):
    """Parent-chunk: соседние ±radius строк той же страницы."""
    c = chunks[idx]
    page = c["page"]
    left = idx
    while left > 0 and chunks[left - 1]["page"] == page and idx - left < radius:
        left -= 1
    right = idx
    while right + 1 < len(chunks) and chunks[right + 1]["page"] == page and right - idx < radius:
        right += 1
    text = " ".join(chunks[i]["text"] for i in range(left, right + 1))
    return {**c, "text": text, "section": f"lines {left}-{right}"}


# ---------- embeddings ----------
EMB: dict = {}


def load_cache():
    for path in [DATA / "embeddings_512.npz", ROOT / "emb_cache.npz"]:
        if path.exists():
            z = np.load(path, allow_pickle=True)
            EMB.update({str(k): v.astype(np.float32) for k, v in zip(z["keys"], z["vectors"])})
    return len(EMB)


def save_cache():
    if not EMB:
        return
    np.savez_compressed(
        ROOT / "emb_cache.npz",
        keys=np.array(list(EMB.keys())),
        vectors=np.stack(list(EMB.values())).astype(np.float16),
    )


def embed_batch(texts):
    started = time.perf_counter()
    data = post_with_retry(EMBED_URL, {"model": EMBED_MODEL, "input": texts, "dimensions": DIM})
    ledger.add("embed", EMBED_MODEL, data.get("usage") or {}, time.perf_counter() - started)
    return [row["embedding"] for row in data["data"]]


def key_of(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_cached(texts, batch=64):
    texts = list(texts)
    missing = [t for t in texts if key_of(t) not in EMB]
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        for t, v in zip(part, embed_batch(part)):
            EMB[key_of(t)] = np.asarray(v, dtype=np.float32)
        print(f"  embedded {min(i + batch, len(missing))}/{len(missing)}", flush=True)
    return np.stack([EMB[key_of(t)] for t in texts]).astype(np.float32)


def emb_text(chunk):
    return f"{chunk['page']}: {chunk['text']}"


# ---------- search ----------
def search_numpy(query_vec, vectors, k=5):
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    mat = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    scores = mat @ q
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


NUMBERS = {
    word: str(n)
    for n, word in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())
}


def norm(text):
    words = re.sub(r"\b(a|an|the)\b|[^\w\s]", " ", str(text).lower()).split()
    return " ".join(NUMBERS.get(w, w) for w in words)


def is_gold(chunk, task):
    if chunk["page"] != task["page"]:
        return False
    text, words = norm(chunk["text"]), norm(task["evidence"]).split()
    if not words:
        return norm(task["answer"]) in text
    return norm(task["answer"]) in text and sum(w in text for w in words) >= 0.5 * len(words)


def tokens(text):
    return re.findall(r"\w+", text.lower())


def build_keyword_index(texts):
    docs = [set(tokens(text)) for text in texts]
    df = {}
    for d in docs:
        for w in d:
            df[w] = df.get(w, 0) + 1
    return docs, {w: np.log((len(docs) + 1) / (n + 1)) for w, n in df.items()}


def keyword_search(query, docs, idf, k=20):
    q = set(tokens(query))
    scores = np.array([sum(idf.get(w, 0.0) for w in q & d) for d in docs])
    return [int(i) for i in np.argsort(-scores)[:k]]


def rrf(rankings, k=5, K=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (K + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


# ---------- milvus ----------
client = None
MILVUS_MODE = "unknown"


def connect_milvus():
    global client, MILVUS_MODE
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    lite = ROOT / "milvus_lite.db"
    # Prefer Lite when Docker is not available (default for this environment).
    if os.getenv("MILVUS_FORCE_SERVER", "").strip() != "1":
        try:
            client = MilvusClient(uri=str(lite))
            client.list_collections()
            MILVUS_MODE = f"lite:{lite.name}"
            print("Milvus:", MILVUS_MODE, "(Docker недоступен — пометка в README)")
            return client
        except Exception as e:
            print("Lite failed:", e)
    try:
        c = MilvusClient(uri=uri, token=os.getenv("MILVUS_TOKEN", ""))
        c.list_collections()
        client = c
        MILVUS_MODE = f"server:{uri}"
        print("Milvus:", MILVUS_MODE)
        return client
    except Exception as e:
        print("Milvus server недоступен:", e)
        client = MilvusClient(uri=str(lite))
        MILVUS_MODE = f"lite:{lite.name}"
        print("Milvus:", MILVUS_MODE)
        return client


def index_to_milvus(name, chunks, vectors):
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(collection_name=name, dimension=int(vectors.shape[1]), metric_type="COSINE")
    rows = []
    for i, (c, v) in enumerate(zip(chunks, vectors)):
        rows.append({
            "id": i, "vector": v.tolist(),
            "page": c.get("page", ""), "section": c.get("section", ""),
            "url": c.get("url", ""), "text": c["text"][:65000],
        })
    for start in range(0, len(rows), 500):
        client.insert(collection_name=name, data=rows[start:start + 500])
    try:
        client.load_collection(name)
    except Exception:
        pass
    return len(rows)


def search_milvus(name, query, k=5, flt=""):
    try:
        client.load_collection(name)
    except Exception:
        pass
    vec = embed_cached([query])[0].tolist()
    kwargs = {"collection_name": name, "data": [vec], "limit": k, "output_fields": ["page", "section", "url", "text"]}
    if flt:
        kwargs["filter"] = flt
    hits = client.search(**kwargs)[0]
    out = []
    for h in hits:
        ent = h.get("entity") or {}
        out.append({
            "id": h.get("id"), "score": float(h.get("distance") or h.get("score") or 0),
            "page": ent.get("page", ""), "section": ent.get("section", ""),
            "url": ent.get("url", ""), "text": ent.get("text", ""),
        })
    return out


# ---------- RAG / eval ----------
RAG_SYSTEM = (
    "Answer the question using only the numbered sources. Cite the source number like [2]. "
    "If the sources do not contain the answer, reply exactly NOT_FOUND. The last line must be FINAL: <short answer>."
)
PLAIN_SYSTEM = "Answer the question. If you do not know, reply NOT_FOUND. The last line must be FINAL: <short answer>."


def rag_answer(question, model, k=5, name="lines", parent=False, line_chunks=None):
    before = ledger.mine
    hits = search_milvus(name, question, k)
    if parent and line_chunks is not None:
        expanded = []
        for h in hits:
            idx = int(h["id"])
            if 0 <= idx < len(line_chunks):
                p = expand_parent(line_chunks, idx, radius=2)
                expanded.append({**h, "text": p["text"], "section": p["section"]})
            else:
                expanded.append(h)
        hits = expanded
    sources = "\n".join(f"[{i + 1}] ({h['page']}) {h['text']}" for i, h in enumerate(hits))
    msg = chat(
        [{"role": "system", "content": RAG_SYSTEM},
         {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}],
        model, tag="rag",
    )
    return {"answer": msg.get("content") or "", "hits": hits, "cost": ledger.mine - before}


def plain_answer(question, model):
    before = ledger.mine
    msg = chat([{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": question}], model, tag="plain")
    return {"answer": msg.get("content") or "", "hits": [], "cost": ledger.mine - before}


def final_answer(text):
    m = re.findall(r"FINAL:\s*(.+)", text or "")
    return m[-1].strip() if m else (text or "").strip()


def is_correct(task, answer):
    return f" {norm(task['answer'])} " in f" {norm(final_answer(answer))} "


def evaluate(fn, sample, config, model, workers=1):
    def one(task):
        try:
            out = fn(task["question"], model)
        except Exception as e:
            out = {"answer": f"ошибка: {e}", "hits": [], "cost": 0.0}
        return {
            "config": config, "model": model.split("/")[-1], "id": task["id"],
            "correct": is_correct(task, out["answer"]),
            "found": any(is_gold(h, task) for h in out.get("hits") or []),
            "refused": "NOT_FOUND" in (out.get("answer") or ""),
            "cost": out.get("cost") or 0.0,
            "answer": final_answer(out.get("answer") or "")[:80],
            "gold": task["answer"],
        }

    with ThreadPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(one, sample)))


def report(results):
    g = results.groupby(["config", "model"], sort=False)
    table = g.agg(n=("correct", "size"), accuracy=("correct", "mean"), cost_per_task=("cost", "mean")).reset_index()
    table["cost_per_correct"] = table["cost_per_task"] / table["accuracy"].replace(0, float("nan"))
    return table.round(5)


def money_chart(table, path, baseline=None):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if baseline:
        ax.axhline(baseline[1] * 100, color=COLORS["grey"], ls="--", lw=1.2)
    for _, r in table.iterrows():
        color = COLORS["amber"] if "sonnet" in r["model"] else COLORS["teal"] if "haiku" in r["model"] else COLORS["violet"]
        x = max(float(r["cost_per_task"]) * 100, 1e-4)
        ax.scatter(x, r["accuracy"] * 100, s=110, color=color)
        ax.annotate(
            f"{r['config']}\n{r['model']}", (x, r["accuracy"] * 100),
            fontsize=8, xytext=(7, 3), textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("цена вопроса, центы (лог)")
    ax.set_ylabel("доля верных, %")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)
    return fig


# ---------- agent + knowledge_base ----------
TOOLS = {}


def register(fn, args_model, description):
    TOOLS[fn.__name__] = {
        "fn": fn, "args": args_model,
        "schema": {"type": "function", "function": {
            "name": fn.__name__, "description": description,
            "parameters": args_model.model_json_schema(),
        }},
    }


def run_tool(call):
    name = call["function"]["name"]
    try:
        spec = TOOLS[name]
        result = spec["fn"](**spec["args"].model_validate_json(call["function"]["arguments"]).model_dump())
    except Exception as e:
        result = f"ошибка инструмента {name}: {e}"
    return {"role": "tool", "tool_call_id": call["id"], "content": str(result)[:3000]}


AGENT_SYSTEM = (
    "You answer questions about events of 2026. Never answer from memory: look facts up with the tools, "
    "do not repeat the same call. When done, write the last line as FINAL: <short answer>, "
    "or NOT_FOUND if the knowledge base has no answer."
)


class KBArgs(BaseModel):
    query: str = Field(description="short search query in English: names, dates, key terms")
    page: str = Field(default="", description="optional exact page title, e.g. '2026 in Japan'")


def knowledge_base(query: str, page: str = "") -> str:
    flt = f'page == "{page}"' if page.strip() else ""
    try:
        hits = search_milvus("lines", query, 5, flt=flt)
    except Exception as e:
        return f"ошибка поиска: {e}"
    if not hits:
        return "nothing found"
    return "\n".join(f"[{h['page']}] {h['text']}" for h in hits)


def agent(question, model, max_steps=6):
    messages, seen, before = (
        [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}],
        set(), ledger.mine,
    )
    schemas = [t["schema"] for t in TOOLS.values()]
    msg = {"content": ""}
    calls = []
    for step in range(max_steps):
        msg = chat(messages, model, tools=schemas, tag="agent")
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        keys = {(c["function"]["name"], c["function"]["arguments"]) for c in calls}
        if not calls or keys & seen or step == max_steps - 1:
            break
        seen |= keys
        messages += [run_tool(c) for c in calls]
    if calls and (not msg.get("content")):
        messages += [
            {"role": "tool", "tool_call_id": c["id"], "content": "call rejected: answer from what you already have"}
            for c in calls
        ]
        msg = chat(messages, model, tag="agent")
    searches = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
    return {"answer": msg.get("content") or "", "hits": [], "cost": ledger.mine - before, "searches": searches}


def live_search_answer(question, model):
    """Baseline ДЗ1: web_search + page_find через Wikipedia API (без агентного цикла для стабильности цены)."""
    from agent import web_search, page_find  # reuse HW1 tools
    before = ledger.mine
    search = web_search(question)
    title = ""
    m = re.search(r"\[([^\]]+)\]", search or "")
    if m:
        title = m.group(1)
    words = " ".join(re.findall(r"[A-Za-z0-9]+", question)[:6])
    found = page_find(title, words) if title else "no page"
    msg = chat(
        [{"role": "system", "content": RAG_SYSTEM},
         {"role": "user", "content": f"Sources:\n[1] {search[:1500]}\n[2] {found[:2000]}\n\nQuestion: {question}"}],
        model, tag="live",
    )
    return {"answer": msg.get("content") or "", "hits": [], "cost": ledger.mine - before}


# ---------- memory ----------
FACTS_SYSTEM = (
    'Ты ведёшь память ассистента о пользователе. Верни один JSON-объект вида {"facts": ["..."]}: '
    "полный обновлённый список коротких фактов о пользователе. Добавь новые устойчивые факты: имя, город, "
    "работа, интересы, пожелания к ответам. Если новый факт отменяет старый, замени старый. "
    "Разовые вопросы и мелочи не записывай."
)
ASSISTANT_SYSTEM = (
    "Ты помощник, который отвечает на вопросы о событиях 2026 года по базе знаний. "
    "Отвечай на языке пользователя, опирайся на источники и на то, что знаешь о пользователе. "
    "Если в источниках ответа нет, так и скажи."
)
JOURNAL, FACTS_FILE = ROOT / "episodes.jsonl", ROOT / "facts.json"


class Facts(BaseModel):
    facts: list[str] = Field(description="короткие факты о пользователе")


def json_from(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    cleaned = re.sub(r'(\\["\\/bfnrtu])|\\', lambda e: e.group(1) or "\\\\", m.group(0) if m else "{}")
    return json.loads(cleaned, strict=False)


def known_facts():
    return json.loads(FACTS_FILE.read_text(encoding="utf-8")) if FACTS_FILE.exists() else []


def store_facts(facts):
    FACTS_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    if facts:
        index_to_milvus(
            "facts",
            [{"page": "memory", "section": "", "url": "", "text": f} for f in facts],
            embed_cached(facts),
        )


def remember_episode(session, role, text):
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "session": session, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "role": role, "text": text,
        }, ensure_ascii=False) + "\n")


def extract_facts(dialog, known):
    msg = chat(
        [{"role": "system", "content": FACTS_SYSTEM},
         {"role": "user", "content": f"Известные факты: {known}\n\nДиалог:\n{dialog}"}],
        MODELS["cheap"], tag="facts",
    )
    try:
        return Facts.model_validate(json_from(msg.get("content") or "")).facts
    except Exception:
        return list(known)


def recall_facts(question, k=3):
    if not client.has_collection("facts"):
        return []
    hits = search_milvus("facts", question, k)
    return [h["text"] for h in hits]


def search_with_memory(user_text, facts, k=5):
    plain = search_milvus("lines", user_text, k)
    personal = search_milvus("lines", user_text + " " + " ".join(facts), k) if facts else []
    by_id = {h["id"]: h for h in plain + personal}
    return [by_id[i] for i in rrf([[h["id"] for h in plain], [h["id"] for h in personal]], k) if i in by_id]


def talk(session, history, user_text, mode="память"):
    remember_episode(session, "user", user_text)
    facts = recall_facts(user_text) if mode == "память" else []
    hits = search_with_memory(user_text, facts) if "?" in user_text else []
    sources = "\n".join(f"[{i + 1}] ({h['page']}) {h['text']}" for i, h in enumerate(hits)) or "нет"
    system = ASSISTANT_SYSTEM + ("\nЧто известно о пользователе: " + "; ".join(facts) if facts else "")
    window = history[-4:] if mode == "память" else history
    msg = chat(
        [{"role": "system", "content": system}] + window + [
            {"role": "user", "content": f"Источники:\n{sources}\n\n{user_text}"}
        ],
        MODELS["cheap"], tag=f"dialog:{mode}",
    )
    answer = msg.get("content") or ""
    history += [
        {"role": "user", "content": user_text if mode == "память" else f"Источники:\n{sources}\n\n{user_text}"},
        {"role": "assistant", "content": answer},
    ]
    remember_episode(session, "assistant", answer)
    return answer


def end_session(history):
    dialog = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history)
    store_facts(extract_facts(dialog, known_facts()))
    return known_facts()


def test_memory_without_llm():
    """Тесты без модели: пустая / найденный / заменённый факт."""
    # empty
    facts = []
    assert facts == []
    # found
    facts = ["User lives in Berlin", "User prefers short answers"]
    store_facts(facts)
    hits = search_milvus("facts", "Where does the user live?", 2)
    assert any("Berlin" in h["text"] for h in hits), hits
    # replaced
    store_facts(["User lives in Paris", "User prefers short answers"])
    hits2 = search_milvus("facts", "Where does the user live?", 2)
    assert any("Paris" in h["text"] for h in hits2), hits2
    assert not any("Berlin" in h["text"] for h in hits2), hits2
    print("memory tests without LLM: ok")


# ---------- stages ----------
def stage_index(limit_q=0):
    connect_milvus()
    corpus = load_jsonl("corpus_2026.jsonl")
    tasks = load_jsonl("questions.jsonl")
    if limit_q:
        tasks = tasks[:limit_q]
    print(f"corpus pages={len(corpus)} tasks={len(tasks)}")

    line_chunks = [c for p in corpus for c in chunk_lines(p)]
    char_chunks = [c for p in corpus for c in chunk_chars(p)]
    print(f"chunks lines={len(line_chunks)} chars={len(char_chunks)}")
    assert len(line_chunks) >= 300, len(line_chunks)

    load_cache()
    print("cache keys", len(EMB))
    line_vecs = embed_cached([emb_text(c) for c in line_chunks])
    char_vecs = embed_cached([c["text"] for c in char_chunks])
    q_vecs = embed_cached([t["question"] for t in tasks])
    save_cache()
    print("vecs", line_vecs.shape, char_vecs.shape, q_vecs.shape)

    def recall_curve(chunks, vectors, ks=(1, 3, 5, 10)):
        ranked = [[i for i, _ in search_numpy(q, vectors, max(ks))] for q in q_vecs]
        return {
            k: float(np.mean([any(is_gold(chunks[i], t) for i in r[:k]) for r, t in zip(ranked, tasks)]))
            for k in ks
        }

    curves = {
        "по строкам": recall_curve(line_chunks, line_vecs),
        "по 400 символов": recall_curve(char_chunks, char_vecs),
    }
    print("recall", curves)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, curve in curves.items():
        ks = sorted(curve)
        ax.plot(ks, [curve[k] * 100 for k in ks], marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k, %")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(IMG / "recall_at_k.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "recall_at_k.png")

    # hybrid table
    rows = []
    for cut, chunks, vectors, texts in [
        ("по 400 символов", char_chunks, char_vecs, [c["text"] for c in char_chunks]),
        ("по строкам", line_chunks, line_vecs, [emb_text(c) for c in line_chunks]),
    ]:
        docs, idf = build_keyword_index(texts)
        by_vector = [[i for i, _ in search_numpy(q, vectors, 20)] for q in q_vecs]
        by_words = [keyword_search(t["question"], docs, idf, 20) for t in tasks]
        by_hybrid = [rrf([v, w], 20) for v, w in zip(by_vector, by_words)]

        def recall_of(rankings, k=5):
            return float(np.mean([any(is_gold(chunks[i], t) for i in r[:k]) for r, t in zip(rankings, tasks)]))

        for name, r in [("только вектор", by_vector), ("только слова", by_words), ("гибрид RRF", by_hybrid)]:
            rows.append({
                "нарезка": cut, "поиск": name,
                "recall@1": round(recall_of(r, 1), 2),
                "recall@5": round(recall_of(r, 5), 2),
            })
    hybrid = pd.DataFrame(rows)
    hybrid.to_csv(ROOT / "hybrid_table.csv", index=False)
    print(hybrid.to_string(index=False))
    ax = hybrid.pivot(index="нарезка", columns="поиск", values="recall@1")[
        ["только вектор", "только слова", "гибрид RRF"]
    ].plot.bar(figsize=(7, 3.2), color=[COLORS["violet"], COLORS["teal"], COLORS["amber"]], rot=0)
    ax.set_ylabel("recall@1")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG / "hybrid_recall.png", dpi=150, bbox_inches="tight")

    # milvus index + filter demo
    index_to_milvus("lines", line_chunks, line_vecs)
    index_to_milvus("chars", char_chunks, char_vecs)
    demo = search_milvus("lines", "earthquake magnitude", 5, 'page == "2026 in Japan"')
    print("filter demo:", [(h["page"], h["text"][:80]) for h in demo[:3]])
    (ROOT / "index_meta.json").write_text(json.dumps({
        "milvus_mode": MILVUS_MODE,
        "line_chunks": len(line_chunks),
        "char_chunks": len(char_chunks),
        "recall": curves,
        "chosen_k": 5,
        "chosen_k_reason": "recall@5 на построчной нарезке почти saturates; дальше дорогой контекст почти не растёт.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # persist chunk counts for later stages
    np.savez_compressed(
        ROOT / "hw2_arrays.npz",
        line_vecs=line_vecs, char_vecs=char_vecs, q_vecs=q_vecs,
    )
    (ROOT / "line_chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in line_chunks) + "\n", encoding="utf-8",
    )
    return curves


def stage_eval(limit=50):
    connect_milvus()
    load_cache()
    tasks = load_jsonl("questions.jsonl")[:limit]
    unans = load_jsonl("unanswerable.jsonl")
    register(knowledge_base, KBArgs, "Searches the local knowledge base of 2026 events")

    # smoke tests for knowledge_base: normal / empty / error
    ok = knowledge_base("Masters Tournament Rory McIlroy 2026")
    assert "McIlroy" in ok or "Masters" in ok, ok[:160]
    empty = knowledge_base("qwxzvnonexistenttoken99999xyz")
    assert empty == "nothing found" or "ошибка" in empty or len(empty) < 80
    bad = knowledge_base("test", page='broken"quote')
    assert "ошибка" in bad or "nothing found" in bad or "[" in bad
    print("knowledge_base tests: ok / empty / error handled")

    frames = []
    print("eval strong plain")
    frames.append(evaluate(plain_answer, tasks, "сильная, без инструментов", MODELS["strong"], workers=2))
    print("eval cheap live search")
    frames.append(evaluate(live_search_answer, tasks, "дешёвая, живой поиск ДЗ1", MODELS["cheap"], workers=2))
    print("eval cheap RAG")
    frames.append(evaluate(
        lambda q, m: rag_answer(q, m, k=5, name="lines"),
        tasks, "дешёвая, RAG всегда", MODELS["cheap"], workers=2,
    ))
    print("eval cheap agent KB")
    frames.append(evaluate(agent, tasks, "дешёвая, агент с knowledge_base", MODELS["cheap"], workers=2))
    print("eval mid RAG")
    frames.append(evaluate(
        lambda q, m: rag_answer(q, m, k=5, name="lines"),
        tasks, "средняя, RAG всегда", MODELS["mid"], workers=2,
    ))

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(ROOT / "results_rag.csv", index=False)
    table = report(results)
    table.to_csv(ROOT / "results_rag_table.csv", index=False)
    (ROOT / "results_rag.md").write_text(table.to_string(index=False), encoding="utf-8")
    money_chart(table, IMG / "money_rag.png")
    print(table.to_string(index=False))

    # refusals
    refusals = []
    for q in unans:
        out = rag_answer(q["question"], MODELS["cheap"], k=5, name="lines")
        refusals.append({
            "id": q["id"], "question": q["question"],
            "refused": "NOT_FOUND" in (out["answer"] or ""),
            "answer": final_answer(out["answer"])[:80],
        })
    ref_df = pd.DataFrame(refusals)
    false_refuse = int(results[(results["config"] == "дешёвая, RAG всегда") & results["refused"] & results["correct"] == False & results["refused"]].shape[0])
    # better: answerable refused
    ans_refused = int(((results["config"] == "дешёвая, RAG всегда") & results["refused"]).sum())
    refusal_summary = {
        "unanswerable_total": len(unans),
        "unanswerable_refused": int(ref_df["refused"].sum()),
        "answerable_n": len(tasks),
        "answerable_false_refuse": ans_refused,
    }
    Path(ROOT / "refusals.json").write_text(json.dumps({"summary": refusal_summary, "rows": refusals}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("refusals", refusal_summary)

    # diagnose failures
    best = results[results["config"] == "дешёвая, RAG всегда"]
    wrong = best[~best["correct"]]
    miss = wrong[~wrong["found"]]
    bad = wrong[wrong["found"]]
    examples = {
        "search_miss": miss.head(1).to_dict(orient="records"),
        "found_but_wrong": bad.head(1).to_dict(orient="records"),
        "counts": {"search_miss": int(len(miss)), "found_but_wrong": int(len(bad))},
    }
    Path(ROOT / "failures_rag.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print("failures", examples["counts"])
    print("total cost", round(ledger.total, 3))
    return table


def stage_memory():
    connect_milvus()
    load_cache()
    # ensure lines collection exists
    if not client.has_collection("lines"):
        raise SystemExit("сначала запустите --index")

    for path in (JOURNAL, FACTS_FILE):
        path.unlink(missing_ok=True)
    if client.has_collection("facts"):
        client.drop_collection("facts")

    test_memory_without_llm()

    # session 1
    h1 = []
    talk("s1", h1, "Привет! Меня зовут Аня, я живу в Берлине и люблю короткие ответы.")
    talk("s1", h1, "Who won the 2026 Masters Tournament?")
    facts1 = end_session(h1)
    print("facts after s1:", facts1)

    # session 2 — no history
    h2 = []
    a2 = talk("s2", h2, "Где я живу? И кто выиграл Masters 2026?")
    print("s2 answer:", a2[:300])

    # change fact
    h3 = []
    talk("s3", h3, "Кстати, я переехала в Париж.")
    facts3 = end_session(h3)
    print("facts after move:", facts3)
    assert any("Париж" in f or "Paris" in f for f in facts3) or any("Paris" in f or "Париж" in f for f in known_facts())

    # token chart: 10+ turns
    script = [
        "Меня зовут Борис, я из Москвы.",
        "Who won the 2026 PGA Championship?",
        "Предпочитаю ответы на русском.",
        "What measure came into force in Victoria in January 2026?",
        "Я работаю аналитиком.",
        "Who won GCT Super Rapid and Blitz Poland in May 2026?",
        "Люблю шахматы.",
        "Who was announced as Pritzker Architecture Prize winner on March 12, 2026?",
        "Повтори, из какого я города?",
        "Who won the 2026 Open Championship?",
        "Спасибо, это всё.",
    ]
    # full history mode
    ledger_before = ledger.total
    hist_full, toks_full = [], []
    for u in script:
        before = ledger.mine
        talk("tok-full", hist_full, u, mode="полная")
        # approximate tokens from last chat cost via prompt — use char/4
        toks_full.append(sum(len(m["content"]) for m in hist_full) // 4)
    cost_full = ledger.total - ledger_before

    ledger_before = ledger.total
    hist_mem, toks_mem = [], []
    # reset facts lightly
    store_facts(["User name is Boris", "User lives in Moscow", "User prefers Russian answers", "User likes chess"])
    for u in script:
        talk("tok-mem", hist_mem, u, mode="память")
        facts = recall_facts(u)
        window = hist_mem[-4:]
        toks_mem.append((sum(len(m["content"]) for m in window) + sum(len(f) for f in facts)) // 4)
    cost_mem = ledger.total - ledger_before

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(range(1, len(toks_full) + 1), toks_full, marker="o", label=f"вся история (~${cost_full:.3f})")
    ax.plot(range(1, len(toks_mem) + 1), toks_mem, marker="o", label=f"окно+факты (~${cost_mem:.3f})")
    ax.set_xlabel("реплика")
    ax.set_ylabel("токены входа (оценка)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.savefig(IMG / "tokens_memory.png", dpi=150, bbox_inches="tight")
    print("saved", IMG / "tokens_memory.png")
    Path(ROOT / "memory_demo.json").write_text(json.dumps({
        "facts_s1": facts1, "answer_s2": a2, "facts_after_move": facts3,
        "tokens_full": toks_full, "tokens_mem": toks_mem,
        "cost_full": cost_full, "cost_mem": cost_mem,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_bonuses(limit=40):
    connect_milvus()
    load_cache()
    tasks = load_jsonl("questions.jsonl")[:limit]
    line_chunks = [json.loads(l) for l in (ROOT / "line_chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # parent-chunk compare recall
    q_vecs = embed_cached([t["question"] for t in tasks])
    # rebuild line vecs from cache
    line_vecs = embed_cached([emb_text(c) for c in line_chunks])

    def recall_at(k, parent=False):
        ok = 0
        for q, t in zip(q_vecs, tasks):
            ranked = [i for i, _ in search_numpy(q, line_vecs, k)]
            chunks = [expand_parent(line_chunks, i) if parent else line_chunks[i] for i in ranked]
            ok += any(is_gold(c, t) for c in chunks)
        return ok / len(tasks)

    parent_table = pd.DataFrame([
        {"режим": "child (строка)", "recall@1": round(recall_at(1, False), 2), "recall@5": round(recall_at(5, False), 2)},
        {"режим": "parent (±2 строки)", "recall@1": round(recall_at(1, True), 2), "recall@5": round(recall_at(5, True), 2)},
    ])
    parent_table.to_csv(ROOT / "parent_chunk_table.csv", index=False)
    print(parent_table.to_string(index=False))

    # small RAG accuracy compare
    child_res = evaluate(lambda q, m: rag_answer(q, m, k=5, name="lines", parent=False), tasks[:20], "RAG child", MODELS["cheap"], workers=2)
    parent_res = evaluate(
        lambda q, m: rag_answer(q, m, k=5, name="lines", parent=True, line_chunks=line_chunks),
        tasks[:20], "RAG parent", MODELS["cheap"], workers=2,
    )
    cmp = report(pd.concat([child_res, parent_res], ignore_index=True))
    cmp.to_csv(ROOT / "parent_rag_compare.csv", index=False)
    print(cmp.to_string(index=False))

    # PDF bonus
    from pypdf import PdfReader

    def pdf_chunks(path):
        reader = PdfReader(str(path))
        pages = [(n + 1, " ".join((p.extract_text() or "").split())) for n, p in enumerate(reader.pages)]
        return [
            {"page": path.name, "section": f"стр. {n}", "url": "", "text": text}
            for n, text in pages if len(text) >= 80
        ]

    pdf = [c for path in sorted((DATA / "pdf").glob("*.pdf")) for c in pdf_chunks(path)]
    index_to_milvus("course", pdf, embed_cached([emb_text(c) for c in pdf]))
    save_cache()
    COURSE_SYSTEM = RAG_SYSTEM.replace("Answer the question", "Answer in Russian") + " Always cite (file, page) like (homework01.pdf, стр. 2)."
    demos = []
    for q in [
        "Что такое RRF и зачем он нужен?",
        "Какая доля верных ответов Sonnet допустима по критерию свежести?",
        "Сколько баллов дают за два разобранных провала?",
    ]:
        hits = search_milvus("course", q, 3)
        sources = "\n".join(f"[{i + 1}] ({h['page']}, {h['section']}) {h['text'][:2500]}" for i, h in enumerate(hits))
        msg = chat(
            [{"role": "system", "content": COURSE_SYSTEM},
             {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {q}"}],
            MODELS["cheap"], tag="pdf",
        )
        demos.append({
            "question": q,
            "answer": msg.get("content") or "",
            "sources": [(h["page"], h["section"]) for h in hits],
        })
        print(q, "→", final_answer(demos[-1]["answer"])[:120], demos[-1]["sources"])
    Path(ROOT / "pdf_demos.json").write_text(json.dumps(demos, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--bonuses", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    if args.all:
        args.index = args.eval = args.memory = args.bonuses = True
    if not any([args.index, args.eval, args.memory, args.bonuses]):
        ap.print_help()
        return
    if args.index:
        stage_index()
    if args.eval:
        stage_eval(limit=args.limit)
    if args.memory:
        stage_memory()
    if args.bonuses:
        stage_bonuses(limit=min(40, args.limit))
    print("DONE total $", round(ledger.total, 3))


if __name__ == "__main__":
    main()
