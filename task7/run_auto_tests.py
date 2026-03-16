import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../rag_tg_bot
sys.path.insert(0, str(PROJECT_ROOT))

from task7.questions import questions
from task7.bot import ask, build_chroma_index, SETTINGS  # <-- поменяй при необходимости


OUT_DIR = PROJECT_ROOT / "task7" / "test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
JSONL_PATH = OUT_DIR / f"run_{RUN_ID}.jsonl"
TXT_PATH = OUT_DIR / f"run_{RUN_ID}.txt"

SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.5"))  # пауза между запросами
USER_ID = int(os.getenv("TEST_USER_ID", "999999"))        # фиктивный user_id


# ----------------------------
# Логирование
# ----------------------------
logger = logging.getLogger("auto_tests")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler(TXT_PATH, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)


def safe_call_ask(q: str, collection, embed_model) -> dict:
    t0 = time.time()
    ok = True
    err = None
    answer = None

    try:
        answer = ask(USER_ID, q, collection, embed_model)
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"

    dt_ms = int((time.time() - t0) * 1000)

    return {
        "question": q,
        "ok": ok,
        "latency_ms": dt_ms,
        "answer": answer,
        "error": err,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    logger.info("=== AUTO TESTS START ===")
    logger.info(f"Questions loaded: {len(questions)}")
    logger.info(f"JSONL output: {JSONL_PATH}")
    logger.info(f"Text log:     {TXT_PATH}")

    # 1) Инициализация Chroma + embedder
    client, collection, embed_model = build_chroma_index(SETTINGS)

    # 2) Прогон вопросов
    results = []
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            q = str(q).strip()
            if not q:
                continue

            logger.info(f"[{i}/{len(questions)}] Q: {q}")

            rec = safe_call_ask(q, collection, embed_model)
            results.append(rec)

            # JSONL запись (по одной строке на кейс)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            if rec["ok"]:
                # В лог печатаем коротко, чтобы не раздувать
                ans_preview = (rec["answer"] or "")[:300].replace("\n", " ")
                logger.info(f" -> OK ({rec['latency_ms']} ms): {ans_preview}")
            else:
                logger.error(f" -> ERROR ({rec['latency_ms']} ms): {rec['error']}")

            time.sleep(SLEEP_BETWEEN)

    # 3) Итог
    ok_count = sum(1 for r in results if r["ok"])
    err_count = len(results) - ok_count
    avg_latency = int(sum(r["latency_ms"] for r in results) / max(1, len(results)))

    logger.info("=== AUTO TESTS FINISH ===")
    logger.info(f"OK: {ok_count}, ERR: {err_count}, AVG_LATENCY_MS: {avg_latency}")


if __name__ == "__main__":
    main()
