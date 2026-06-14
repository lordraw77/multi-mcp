#!/usr/bin/env python3
"""
Elenca i modelli disponibili sull'endpoint gratuito NVIDIA NIM
(https://integrate.api.nvidia.com/v1) e misura la velocità di risposta
(tempo al primo token e token/sec) per ciascuno, ordinando i risultati
dal più rapido al più lento.

Uso:
  python3 nvidia_free_models.py                  # benchmark dei primi 15 modelli chat
  python3 nvidia_free_models.py --list           # elenca solo i modelli disponibili
  python3 nvidia_free_models.py --limit 30       # benchmark di 30 modelli
  python3 nvidia_free_models.py --filter llama   # solo modelli con "llama" nell'id
  python3 nvidia_free_models.py --max-tokens 100 # genera più token per misure più stabili

I modelli che falliscono vengono censiti per non ritestarli ogni volta:
  - 404/400 (non disponibili per l'account o richiesta non valida) -> .cache/nvidia_404_models.txt
  - timeout                              -> .cache/nvidia_timeout_models.txt
  - altri errori (es. 422 parametri)     -> .cache/nvidia_other_errors.txt
Usa --retest-404 / --retest-timeout / --retest-errors per ritestarli.

Richiede MAIN_AGENT_NVIDIA_API_KEY (o NVIDIA_API_KEY) nel file .env / ambiente.
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError

from nvidia_ratelimit import wrap_if_nvidia

BASE_URL = "https://integrate.api.nvidia.com/v1"

# File in cui vengono accumulati i modelli che falliscono con 404 (non
# disponibili per questo account) o in timeout, per non ritestarli ogni volta.
_CACHE_DIR = Path(__file__).parent / ".cache"
NOT_FOUND_FILE = _CACHE_DIR / "nvidia_404_models.txt"
TIMEOUT_FILE = _CACHE_DIR / "nvidia_timeout_models.txt"
OTHER_ERROR_FILE = _CACHE_DIR / "nvidia_other_errors.txt"

# Sostringhe che identificano modelli non adatti a un benchmark di chat
# semplice (embedding, reranking, guardrail, modelli vision/audio, ecc.)
SKIP_SUBSTRINGS = [
    "embed", "rerank", "retriever", "guard", "nemoguard",
    "nemotron-safety", "parakeet", "canary", "ocr", "vila",
    "fastpitch", "radtts", "vfm",
]


def _read_model_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_model(path: Path, model_id: str) -> None:
    existing = _read_model_list(path)
    if model_id in existing:
        return
    _CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(model_id + "\n")


def _is_404(error: str) -> bool:
    return "404" in error or "400" in error


def _is_timeout(error: str) -> bool:
    lower = error.lower()
    return "timed out" in lower or "timeout" in lower


def get_api_key() -> str:
    key = os.getenv("MAIN_AGENT_NVIDIA_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise SystemExit(
            "Nessuna API key NVIDIA trovata. Imposta MAIN_AGENT_NVIDIA_API_KEY "
            "(o NVIDIA_API_KEY) nel file .env."
        )
    return key


def list_models(client: OpenAI) -> list[str]:
    return sorted(m.id for m in client.models.list().data)


def looks_like_chat_model(model_id: str) -> bool:
    lower = model_id.lower()
    return not any(s in lower for s in SKIP_SUBSTRINGS)


def benchmark_model(client: OpenAI, model_id: str, max_tokens: int, prompt: str, timeout: float) -> dict:
    """Esegue una chiamata in streaming e misura tempo al primo token e token/sec."""
    start = time.monotonic()
    first_token_time = None
    token_count = 0
    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            timeout=timeout,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                token_count += 1
    except APIError as e:
        return {"model": model_id, "ok": False, "error": str(e)}
    except Exception as e:  # rete, timeout, ecc.
        return {"model": model_id, "ok": False, "error": str(e)}

    end = time.monotonic()
    if first_token_time is None or token_count == 0:
        return {"model": model_id, "ok": False, "error": "nessun token ricevuto"}

    total_time = end - start
    ttft = first_token_time - start
    gen_time = end - first_token_time
    tokens_per_sec = token_count / gen_time if gen_time > 0 else float("inf")

    return {
        "model": model_id,
        "ok": True,
        "ttft": ttft,
        "total_time": total_time,
        "tokens": token_count,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="elenca solo i modelli disponibili, senza benchmark")
    parser.add_argument("--limit", type=int, default=100, help="numero massimo di modelli da testare (default: 15)")
    parser.add_argument("--filter", default="", help="testa solo i modelli il cui id contiene questa stringa")
    parser.add_argument("--max-tokens", type=int, default=50, help="token massimi generati per test (default: 50)")
    parser.add_argument(
        "--prompt", default="Conta da 1 a 20.",
        help="prompt usato per il benchmark (default: 'Conta da 1 a 20.')",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="timeout in secondi per ogni chiamata (default: 5)",
    )
    parser.add_argument(
        "--retest-404", action="store_true",
        help=f"ritesta anche i modelli già segnati come 404 in {NOT_FOUND_FILE}",
    )
    parser.add_argument(
        "--retest-timeout", action="store_true",
        help=f"ritesta anche i modelli già segnati come timeout in {TIMEOUT_FILE}",
    )
    parser.add_argument(
        "--retest-errors", action="store_true",
        help=f"ritesta anche i modelli già segnati con altri errori in {OTHER_ERROR_FILE}",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = get_api_key()
    raw_client = OpenAI(base_url=BASE_URL, api_key=api_key)
    client = wrap_if_nvidia("nvidia", raw_client, api_key)

    print("Recupero elenco modelli disponibili...")
    models = list_models(raw_client)
    print(f"Trovati {len(models)} modelli sull'endpoint gratuito NVIDIA NIM ({BASE_URL}).\n")

    if args.list:
        for m in models:
            print(m)
        return

    candidates = [m for m in models if looks_like_chat_model(m)]
    if args.filter:
        candidates = [m for m in candidates if args.filter.lower() in m.lower()]

    not_found = _read_model_list(NOT_FOUND_FILE)
    timed_out = _read_model_list(TIMEOUT_FILE)
    other_errors = _read_model_list(OTHER_ERROR_FILE)
    skipped = 0
    if not args.retest_404 and not_found:
        before = len(candidates)
        candidates = [m for m in candidates if m not in not_found]
        skipped += before - len(candidates)
    if not args.retest_timeout and timed_out:
        before = len(candidates)
        candidates = [m for m in candidates if m not in timed_out]
        skipped += before - len(candidates)
    if not args.retest_errors and other_errors:
        before = len(candidates)
        candidates = [m for m in candidates if m not in other_errors]
        skipped += before - len(candidates)
    if skipped:
        print(f"Saltati {skipped} modelli già noti come 404/timeout/errore "
              f"(usa --retest-404 / --retest-timeout / --retest-errors per ritestarli).\n")

    candidates = candidates[: args.limit]

    if not candidates:
        print("Nessun modello corrisponde ai criteri scelti.")
        return

    print(f"Benchmark di {len(candidates)} modelli (max_tokens={args.max_tokens}, prompt={args.prompt!r})...\n")

    results = []
    for i, model_id in enumerate(candidates, 1):
        if i > 1:
            time.sleep(1)
        print(f"[{i}/{len(candidates)}] {model_id} ...", end=" ", flush=True)
        result = benchmark_model(client, model_id, args.max_tokens, args.prompt, args.timeout)
        if result["ok"]:
            print(f"TTFT={result['ttft']:.2f}s  {result['tokens_per_sec']:.1f} tok/s  ({result['tokens']} token)")
        else:
            print(f"errore: {result['error']}")
            if _is_404(result["error"]):
                _append_model(NOT_FOUND_FILE, model_id)
            elif _is_timeout(result["error"]):
                _append_model(TIMEOUT_FILE, model_id)
            else:
                _append_model(OTHER_ERROR_FILE, model_id)
        results.append(result)

    ok_results = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    ok_results.sort(key=lambda r: r["tokens_per_sec"], reverse=True)

    print("\n=== Classifica per velocità (token/sec, dal più rapido) ===")
    for r in ok_results:
        print(f"{r['tokens_per_sec']:7.1f} tok/s | TTFT {r['ttft']:.2f}s | {r['model']}")

    if failed:
        print("\n=== Modelli non testabili / errore ===")
        for r in failed:
            print(f"{r['model']}: {r['error']}")


if __name__ == "__main__":
    main()
