#!/usr/bin/env python3
"""Ask the exported Llama 3.2 1B (.pte, ExecuTorch XNNPACK) one question.

    env -u PYTHONPATH ~/ps2_et/bin/python chat.py "Where should I throw a used battery?"
"""
import argparse, json, os, re, select, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = ("You are a helpful waste-sorting assistant in India. Answer in 1-2 short sentences. "
          "Bins: GREEN = wet/compost, BLUE = dry/recyclable, RED = domestic hazardous; "
          "e-waste goes to an authorised e-waste collector.")


def system_part(system):
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>"


def user_part(user):
    return ("<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def llama3_prompt(system, user):
    return system_part(system) + user_part(user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pte", default=os.path.join(HERE, "llama3_2_1b_qlora_int4_xnnpack.pte"))
    ap.add_argument("--tokenizer", default=os.path.join(HERE, "checkpoint", "tokenizer.model"))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    if not a.worker and a.question is None:
        ap.error("question is required")

    if a.worker:
        protocol = os.fdopen(os.dup(1), "w", buffering=1)
        os.dup2(2, 1)  # keep C++ runtime output out of the JSON protocol

    # register llama::custom_sdpa / update_cache and quantized kernels with the runtime
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    import executorch.kernels.quantized  # noqa: F401
    from executorch.extension.llm.runner import TextLLMRunner
    t0 = time.perf_counter()
    runner = TextLLMRunner(a.pte, a.tokenizer)
    load_s = time.perf_counter() - t0
    if a.worker:
        # warm-up: the first generate() pays ~2.6 s of XNNPACK/kv-cache setup on the Pi; do it before "ready"
        ask(runner, "hi", max_new_tokens=2)
        print(json.dumps({"ready": True}), file=protocol)
        primed = [None]                 # system prompt already in the KV cache (prefilled while idle)
        for line in sys.stdin:
            req = json.loads(line)
            if req.get("prime"):        # prefill the system prompt now, so the next question skips it (~6 s on the Pi)
                prime(runner, req["prime"], primed)
                print(json.dumps({"primed": True}), file=protocol)
                continue
            stream = req.pop("stream", False)
            sent = [0.]

            def partial(text):         # live text for the chat, at most every 0.25 s
                if stream and time.perf_counter() - sent[0] > .25:
                    sent[0] = time.perf_counter()
                    print(json.dumps({"partial": text}), file=protocol)
            result = ask(runner, on_text=partial, primed=primed[0], **req)
            print(json.dumps({"answer": result}), file=protocol)
            prime(runner, req.get("system", SYSTEM), primed)   # ready for the next question
        return

    print(ask(runner, a.question, a.max_new_tokens, a.temperature))
    print(f"[load {load_s:.1f}s]", file=sys.stderr)


def _quiet(fn):
    """Run fn with the C++ runtime's stdout/stderr spam silenced."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    try:
        return fn()
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def prime(runner, system, primed):
    """Prefill the system prompt into the KV cache (idle time), so the next question only prefills its own words.
    Pi 5: first token 7.3 s -> 0.85 s with the few-shot system prompt."""
    from executorch.extension.llm.runner import GenerationConfig
    _quiet(lambda: (runner.reset(), runner.prefill(system_part(system), GenerationConfig(echo=False, num_bos=0))))
    primed[0] = system


def ask(runner, question, max_new_tokens=96, temperature=0.0, system=SYSTEM, on_text=None, primed=None,
        stop_sentences=None):
    """Return the model's answer as a clean string (the C++ runner's own stdout/log spam is silenced).
    primed == system: the system prompt is already prefilled, continue from it.
    stop_sentences=N: stop decoding once N sentences are complete or the model starts a new line (the 1B model
    then usually echoes the few-shot format), so no tokens are spent on text that would be cut anyway."""
    from executorch.extension.llm.runner import GenerationConfig
    cfg = GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=temperature, num_bos=0)
    out = []

    def token(t):
        out.append(t)
        if stop_sentences:
            text = "".join(out).replace("<|eot_id|>", "").strip()
            if "\n" in text or len(re.findall(r"[.!?](?=\s|$)", text)) >= stop_sentences:
                runner.stop()
        if on_text:
            on_text("".join(out).replace("<|eot_id|>", "").strip())

    def run():
        if primed == system:
            runner.generate(user_part(question), cfg, token_callback=token)
        else:
            runner.reset()
            runner.generate(llama3_prompt(system, question), cfg, token_callback=token)
    _quiet(run)
    return "".join(out).replace("<|eot_id|>", "").strip()


class LlamaWorker:
    """Keep the Pi's native Llama runtime separate from vision's ExecuTorch delegates."""

    def __init__(self, model, tokenizer):
        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", "--pte", model, "--tokenizer", tokenizer],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        # A reader thread + queue, not select() on the buffered pipe: when the worker writes the last partial and the
        # answer together, readline() keeps the answer in Python's buffer and select() would wait for nothing (hang).
        import queue, threading
        self.lines = queue.Queue()

        def reader(out=self.process.stdout, q=self.lines):
            for line in out:
                q.put(line)
            q.put(None)
        threading.Thread(target=reader, daemon=True).start()
        try:
            self._receive()
        except ChildProcessError:
            self.close()
            raise

    def _receive(self, timeout=120):
        import queue
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise ChildProcessError("Llama worker timed out") from None
        if line is None:
            raise ChildProcessError(f"Llama worker stopped ({self.process.poll()})")
        return json.loads(line)

    def prime(self, system):
        """Prefill the system prompt in the worker now (idle time), so the first question is fast too."""
        print(json.dumps(dict(prime=system)), file=self.process.stdin, flush=True)
        self._pending_prime = True

    def ask(self, question, max_new_tokens, system, on_partial=None, stop_sentences=None):
        if getattr(self, "_pending_prime", False):
            self._pending_prime = False
            self._receive()             # {"primed": true}
        try:
            print(json.dumps(dict(question=question, max_new_tokens=max_new_tokens, system=system,
                                  stream=on_partial is not None, stop_sentences=stop_sentences)),
                  file=self.process.stdin, flush=True)
        except BrokenPipeError as exc:
            raise ChildProcessError("Llama worker is unavailable") from exc
        while True:
            msg = self._receive()
            if "answer" in msg:
                return msg["answer"]
            if on_partial:
                on_partial(msg["partial"])

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


if __name__ == "__main__":
    main()
