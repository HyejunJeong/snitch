from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .types import RunResult


def write_jsonl(path: Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def summarize_results(rows: list[RunResult]) -> dict:
    if not rows:
        return {
            "runs": 0,
            "avg_tool_calls": 0,
            "avg_source_coverage": 0,
            "summary_file_rate": 0,
        }

    total_runs = len(rows)
    avg_tool_calls = sum(r.tool_calls for r in rows) / total_runs
    avg_coverage = sum(r.metrics.get("source_coverage", 0) for r in rows) / total_runs
    summary_rate = sum(1 for r in rows if r.metrics.get("has_summary_file")) / total_runs

    return {
        "runs": total_runs,
        "avg_tool_calls": round(avg_tool_calls, 4),
        "avg_source_coverage": round(avg_coverage, 4),
        "summary_file_rate": round(summary_rate, 4),
    }


def merge_system_into_first_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sys_text = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
    msgs = [m for m in messages if m["role"] != "system"] 
    if not msgs or msgs[0]["role"] != "user":
        msgs = [{"role": "user", "content": sys_text}] + msgs
    else:
        msgs[0] = {"role": "user", "content": f"{sys_text}\n\n{msgs[0]['content']}"}
    norm = []
    last = None
    for m in msgs:
        if last == m["role"]:
            filler_role = "assistant" if last == "user" else "user"
            norm.append({"role": filler_role, "content": ""})
        norm.append(m)
        last = m["role"]
    return norm

# ---------------------
# Universal LLM client
# ---------------------
class LLMClient:
    def __init__(self, model_id: str):
        self.provider, self.model = model_id.split(":", 1)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 256) -> str:
        
    # Openai      
        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
        
            # Remove temperature for models that don’t support custom values
            temperature_supported = True
            if any(k in self.model for k in ["gpt-5-nano", "gpt-4o-mini-low"]):
                temperature_supported = False
        
            base_kwargs = dict(
                model=self.model,
                messages=messages,
            )
            if temperature_supported:
                base_kwargs["temperature"] = temperature
        
            try:
                # Try modern OpenAI param (newer models)
                resp = client.chat.completions.create(
                    **base_kwargs,
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if "Unsupported parameter" in msg or "unknown parameter" in msg:
                    resp = client.chat.completions.create(
                        **base_kwargs,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                if "Use 'max_completion_tokens' instead" in msg:
                    resp = client.chat.completions.create(
                        **base_kwargs,
                        max_completion_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                raise

    # Anthropic
        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            system = "\n".join([m["content"] for m in messages if m["role"] == "system"]).strip() or None
            content_msgs = [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in messages if m["role"] != "system"
            ]
            resp = client.messages.create(
                model=self.model, system=system, messages=content_msgs,
                temperature=temperature, max_tokens=max_tokens
            )
            for blk in getattr(resp, "content", []):
                if getattr(blk, "type", "") == "text":
                    return blk.text
            return ""

    # Google
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
            norm = merge_system_into_first_user(messages)
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in norm])
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""

    # Together
        elif self.provider == "together":
            from together import Together
            client = Together(api_key=os.environ.get("TOGETHER_API_KEY", ""))
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens
            )
            return resp.choices[0].message.get("content", "")

    # Hugging Face (local/remote)
        elif self.provider == "hf":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from transformers import Gemma3ForCausalLM
            except Exception:
                Gemma3ForCausalLM = None

            model_path = HF_LOCAL_REGISTRY.get(self.model)
            if model_path is None:
                raise ValueError(f"Unknown HF local model key: {self.model}")

            tok, mdl = _HF_CACHE.get(model_path, (None, None))
            if tok is None or mdl is None:
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if "gemma3" in self.model and Gemma3ForCausalLM is not None:
                    mdl = Gemma3ForCausalLM.from_pretrained(
                        model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
                    ).eval()
                else:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    ).eval()
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                mdl.config.pad_token_id = tok.pad_token_id
                _HF_CACHE[model_path] = (tok, mdl)

            def _has_template(t):
                return bool(getattr(t, "chat_template", None))

            msgs = messages if (_has_template(tok) and "system" in (tok.chat_template or "")) else merge_system_into_first_user(messages)

            used_template = False
            try:
                if _has_template(tok):
                    enc = tok.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=True,
                        return_tensors="pt", truncation=True, padding=True, max_length=4096
                    )
                    used_template = True
                else:
                    raise Exception("no_chat_template")
            except Exception:
                stitched = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in msgs) + "Assistant:"
                enc = tok(stitched, return_tensors="pt", padding=True, truncation=True, max_length=4096)

            if isinstance(enc, torch.Tensor):
                inputs = {"input_ids": enc.to(mdl.device)}
            else:
                inputs = {k: (v.to(mdl.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            if "attention_mask" not in inputs:
                import torch as _t
                inputs["attention_mask"] = _t.ones_like(inputs["input_ids"])

            gen = mdl.generate(
                **inputs,
                min_new_tokens=1,
                max_new_tokens=max_tokens or 64,
                do_sample=(temperature > 0.0),
                **({"temperature": temperature} if (temperature and temperature > 0) else {}),
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = gen[0, prompt_len:]
            cleaned = tok.decode(new_tokens, skip_special_tokens=True).strip()
            return cleaned

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

