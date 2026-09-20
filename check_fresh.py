# -*- coding: utf-8 -*-
"""Проверка датасета свежих вопросов перед сдачей домашки.

Запуск: python check_fresh.py data/fresh.jsonl [--sonnet]

Без ключа: проверяет формат записей и что ответ дословно есть в цитате из источника.
С флагом --sonnet: прогоняет сильную модель без инструментов и печатает долю верных.
Критерий свежести: не выше 20 процентов. Вопросы, на которые Sonnet ответил верно, печатаются списком, их надо переписать или убрать.
"""
import sys, json, argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

REQUIRED = ["id", "source", "question", "answer", "evidence", "url"]


def load(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"строка {n}: не JSON ({e})")
    return rows


def normalize(text):
    import re
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def check_format(rows):
    problems = 0
    ids = set()
    for r in rows:
        missing = [k for k in REQUIRED if not str(r.get(k, "")).strip()]
        if missing:
            problems += 1
            print(f"{r.get('id', '?')}: нет полей {missing}")
            continue
        if r["id"] in ids:
            problems += 1
            print(f"{r['id']}: повторяется id")
        ids.add(r["id"])
        if normalize(r["answer"]) not in normalize(r["evidence"]):
            problems += 1
            print(f"{r['id']}: ответ «{r['answer']}» не найден в цитате")
        if len(r["answer"]) > 80:
            problems += 1
            print(f"{r['id']}: ответ длиннее 80 символов, это не короткий ответ")
        if not r["url"].startswith("http"):
            problems += 1
            print(f"{r['id']}: url не похож на ссылку")
    print(f"записей: {len(rows)}, проблем: {problems}")
    return problems == 0


def check_sonnet(rows):
    import fetch_fresh as f
    right = []
    for r in rows:
        ans = f.strong_answer(r["question"])
        if normalize(r["answer"]) in normalize(ans):
            right.append((r["id"], r["question"], ans))
    share = len(right) / len(rows) if rows else 0
    print(f"\nSonnet без инструментов: {len(right)} из {len(rows)} верно, {share:.0%}. Потрачено ${f.COST:.3f}")
    print("критерий свежести " + ("пройден" if share <= 0.2 else "НЕ пройден: не выше 20 процентов"))
    for rid, q, ans in right:
        print(f"  {rid}: {q[:90]} -> {ans[:40]}")
    return share <= 0.2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sonnet", action="store_true", help="прогнать сильную модель без инструментов (нужен ключ)")
    args = ap.parse_args()
    rows = load(args.path)
    ok = check_format(rows)
    if args.sonnet and rows:
        ok = check_sonnet(rows) and ok
    sys.exit(0 if ok else 1)
