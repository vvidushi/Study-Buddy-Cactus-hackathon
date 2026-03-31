"""
coach.py — StudyBuddy: answer scoring via FunctionGemma tool-calling
==============================================================
FunctionGemma-270M is a function-calling model, not a generative one.
We only use it here for what it does well: structured tool output.

Study mode explanations are handled by cactus built-in RAG in app.py —
no prompting needed there.
"""

import sys
import json
import re
from pathlib import Path

CACTUS_REPO = Path(__file__).resolve().parent.parent / "cactus"
sys.path.insert(0, str(CACTUS_REPO / "python" / "src"))
from cactus import cactus_init, cactus_complete, cactus_destroy

LLM_PATH = str(CACTUS_REPO / "weights" / "functiongemma-270m-it")

# ── Tool definition ───────────────────────────────────────────────────────────
# FunctionGemma fills these fields — it does NOT generate free-form text.
FEEDBACK_TOOL = {
    "name": "give_feedback",
    "description": "Give structured feedback on a student's exam answer",
    "parameters": {
        "type": "object",
        "properties": {
            "score":        {"type": "integer", "description": "Score 1-10"},
            "suggestion_1": {"type": "string",  "description": "Improvement tip 1"},
            "suggestion_2": {"type": "string",  "description": "Improvement tip 2"},
            "suggestion_3": {"type": "string",  "description": "Improvement tip 3 (optional)"},
        },
        "required": ["score", "suggestion_1", "suggestion_2"],
    },
}

_DEFAULT_SUGGESTIONS = [
    "Structure your answer with an introduction, body, and conclusion.",
    "Use specific examples or evidence to support your points.",
    "Be more concise — aim for clarity over length.",
]


class StudyBuddy:

    def __init__(self, model_path: str = LLM_PATH):
        self.model_path = model_path

    def get_feedback(self, transcript: str,
                     question: str = "",
                     pdf_context: str = "") -> dict:
        """
        Score a student's answer using FunctionGemma tool-calling.
        Returns { "score": "7/10", "bullets": [...] }
        """
        transcript = transcript.strip()
        if not transcript:
            return {"score": "N/A", "bullets": ["No answer detected. Please try again."], "raw": ""}

        if len(transcript.split()) < 5:
            return {
                "score": "2/10",
                "bullets": [
                    "Your answer is too short — write at least 2-3 full sentences.",
                    "Explain your reasoning, not just a keyword.",
                    "Add examples to support your answer.",
                ],
                "raw": "",
            }

        # Brief context hint — FunctionGemma degrades with long input
        ctx = ""
        if question:
            ctx += f" Q: {question[:100]}."
        if pdf_context:
            ctx += f" Topic: {pdf_context[:150].replace(chr(10), ' ')}..."

        # The user message is just the student's answer + brief context.
        # FunctionGemma reads this and fills the FEEDBACK_TOOL fields.
        user_msg = f"Answer: \"{transcript[:400]}\"{ctx}"

        model = cactus_init(self.model_path)
        try:
            raw_str = cactus_complete(
                model,
                [{"role": "user", "content": user_msg}],
                tools=[{"type": "function", "function": FEEDBACK_TOOL}],
                force_tools=True,
                max_tokens=250,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )
        finally:
            cactus_destroy(model)

        return self._parse(raw_str, transcript, question)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, raw_str: str, transcript: str, question: str) -> dict:
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            return self._fallback(raw_str, transcript, question)

        calls = raw.get("function_calls", [])
        if calls:
            args   = calls[0].get("arguments", {})
            score  = self._valid_score(args.get("score"))
            tips   = [args.get(k, "").strip() for k in ("suggestion_1", "suggestion_2", "suggestion_3")]
            tips   = self._clean_tips(tips, transcript, question)
            if score and tips:
                return {"score": f"{score}/10", "bullets": tips, "raw": raw_str}

        return self._fallback(raw.get("response") or raw_str, transcript, question)

    def _valid_score(self, val) -> int | None:
        try:
            n = int(val)
            return n if 1 <= n <= 10 else None
        except (TypeError, ValueError):
            return None

    def _clean_tips(self, tips: list, transcript: str, question: str) -> list:
        bad = {transcript.lower().strip(), question.lower().strip()} - {""}
        out = []
        for t in tips:
            t = t.strip()
            if len(t) < 10:
                continue
            if any(t.lower().rstrip("?.") in ref or ref in t.lower() for ref in bad):
                continue
            out.append(t)
        return out or _DEFAULT_SUGGESTIONS

    def _fallback(self, text: str, transcript: str, question: str) -> dict:
        score = "?"
        m = re.search(r"(\d+)\s*/\s*10", str(text))
        if m:
            n = int(m.group(1))
            if 1 <= n <= 10:
                score = f"{n}/10"
        tips = re.findall(r"[-•*]\s+(.+)", str(text))[:3]
        tips = self._clean_tips(tips, transcript, question)
        return {"score": score, "bullets": tips, "raw": str(text)}
