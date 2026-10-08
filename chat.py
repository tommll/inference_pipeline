#!/usr/bin/env python3
"""
Interactive streaming chat against a local llama-server.

Standard library only, on purpose: the ~/.local Python environment on this
machine has a NumPy 2.x / system-SciPy conflict, so anything that imports
third-party packages is a coin flip. urllib is not.

Beyond streaming the reply, each turn reports the two numbers that actually
characterise CPU inference on this laptop, kept separate because they are
bound by different things:

  prefill  - time to first token, compute-bound, the slow part at 15 W
  decode   - tokens/sec after that, bandwidth-bound

Qwen3.5 is a hybrid reasoning model that thinks by default. At ~8 tok/s that
means a long silent preamble before any answer, so thinking is OFF here unless
you turn it on with /think.

Commands:  /reset  /system <text>  /think  /nothink  /stats  /quit
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

try:
    import readline  # noqa: F401  - line editing + history, if available
except ImportError:
    pass

HOST = os.environ.get("LLAMA_HOST", "http://127.0.0.1:8080")
DEFAULT_SYSTEM = "You are a helpful assistant. Be concise."

# Qwen3.5 non-thinking defaults; see the model card.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MAX_TOKENS = 512

DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
)
if not sys.stdout.isatty():
    DIM = BOLD = CYAN = GREEN = YELLOW = RED = RESET = ""


def server_info():
    """Return (ok, description). Confirms a model is actually loaded."""
    try:
        with urllib.request.urlopen(f"{HOST}/v1/models", timeout=5) as r:
            data = json.load(r)
        ids = [m.get("id", "?") for m in data.get("data", [])]
        return True, (ids[0] if ids else "unknown model")
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def stream_turn(messages, thinking=False):
    """POST one turn, print tokens as they arrive, return a timing dict."""
    payload = {
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    req = urllib.request.Request(
        f"{HOST}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    sent = time.perf_counter()
    first_tok = None
    pieces = []
    n_decoded = 0
    timings = {}
    usage = {}
    in_think = [False]

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue

            # llama.cpp attaches these; guard for both shapes.
            if isinstance(chunk.get("timings"), dict):
                timings = chunk["timings"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                # Thinking tokens arrive on a separate field; show them dimmed
                # so the wait is visible but never mistaken for the answer.
                think = delta.get("reasoning_content")
                if think:
                    if first_tok is None:
                        first_tok = time.perf_counter()
                        sys.stdout.write(DIM + "(thinking) ")
                        in_think[0] = True
                    n_decoded += 1
                    sys.stdout.write(think)
                    sys.stdout.flush()
                piece = delta.get("content")
                if not piece:
                    continue
                if in_think[0]:
                    sys.stdout.write(RESET + "\n")
                    in_think[0] = False
                if first_tok is None:
                    first_tok = time.perf_counter()
                n_decoded += 1
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()

    if in_think[0]:  # stream ended mid-thought; don't leave the terminal dimmed
        sys.stdout.write(RESET)
        sys.stdout.flush()

    done = time.perf_counter()
    text = "".join(pieces)

    # Prefer the server's own counters; fall back to wall clock.
    ttft = (first_tok - sent) if first_tok else (done - sent)
    decode_wall = (done - first_tok) if first_tok and done > first_tok else 0.0
    stats = {
        "text": text,
        "ttft": ttft,
        "decode_tps": (n_decoded / decode_wall) if decode_wall > 0 else 0.0,
        "n_decoded": usage.get("completion_tokens") or n_decoded,
        "n_prompt": usage.get("prompt_tokens"),
        "total": done - sent,
    }
    if timings:
        if timings.get("predicted_per_second"):
            stats["decode_tps"] = timings["predicted_per_second"]
        if timings.get("prompt_per_second"):
            stats["prefill_tps"] = timings["prompt_per_second"]
        if timings.get("prompt_n") is not None:
            stats["n_prompt"] = timings["prompt_n"]
    return stats


def fmt_stats(s):
    bits = [f"{s['decode_tps']:.1f} tok/s decode", f"{s['ttft']:.2f}s to first token"]
    if s.get("prefill_tps"):
        bits.append(f"{s['prefill_tps']:.0f} tok/s prefill")
    if s.get("n_prompt") is not None:
        bits.append(f"{s['n_prompt']} prompt")
    bits.append(f"{s['n_decoded']} generated")
    return DIM + "  [" + " · ".join(bits) + "]" + RESET


def main():
    ok, info = server_info()
    if not ok:
        print(f"{RED}Cannot reach llama-server at {HOST}{RESET}")
        print(f"  {info}")
        print(f"\nStart it first:  {BOLD}./serve.sh{RESET}")
        return 1

    name = os.path.basename(info)
    print(f"{BOLD}local chat{RESET} {DIM}·{RESET} {CYAN}{name}{RESET} {DIM}via {HOST}{RESET}")
    print(f"{DIM}/reset  /system <text>  /think  /nothink  /stats  /quit"
          f"   (Ctrl-C aborts a reply){RESET}\n")

    system = DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system}]
    history = []
    thinking = False

    while True:
        try:
            user = input(f"{GREEN}you ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            messages = [{"role": "system", "content": system}]
            print(f"{DIM}  context cleared{RESET}\n")
            continue
        if user.startswith("/system "):
            system = user[8:].strip()
            messages = [{"role": "system", "content": system}]
            print(f"{DIM}  system prompt set, context cleared{RESET}\n")
            continue
        if user in ("/think", "/nothink"):
            thinking = user == "/think"
            state = "on - expect a long silent preamble" if thinking else "off"
            print(f"{DIM}  thinking {state}{RESET}\n")
            continue
        if user == "/stats":
            if not history:
                print(f"{DIM}  no turns yet{RESET}\n")
            else:
                d = [h["decode_tps"] for h in history if h["decode_tps"]]
                print(f"{DIM}  {len(history)} turns · "
                      f"decode mean {sum(d)/len(d):.1f} tok/s · "
                      f"best {max(d):.1f} · worst {min(d):.1f}{RESET}")
                print(f"{DIM}  (a falling trend across turns is thermal throttling, "
                      f"not a bug){RESET}\n")
            continue

        messages.append({"role": "user", "content": user})
        print(f"{CYAN}llm ›{RESET} ", end="", flush=True)
        try:
            s = stream_turn(messages, thinking=thinking)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}  aborted{RESET}\n")
            messages.pop()
            continue
        except Exception as e:  # noqa: BLE001
            print(f"\n{RED}  request failed: {e}{RESET}\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": s["text"]})
        history.append(s)
        print("\n" + fmt_stats(s) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
