"""LLM judge for the RL reward: is the requested name used as a person in the text?

Without this check the attacker can win by using the name as a place or a brand ("the nearest
Mulaton store"). The target detector is right not to tag such a mention, so it counts as a
miss and earns the full reward: 236 of 240 texts in pilot-opening-20261114-124322 did this.

The judge is a hosted chat model (OpenAI-style chat completions endpoint). A run stops when
the judge cannot be reached: letting texts through unchecked would bring the exploit back.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
PERSON, NOT_PERSON, UNPARSED = "person", "not_person", "unparsed"

QUESTION = (
    'Text:\n{text}\n\nIn this text, does "{name}" refer to a person '
    "(someone's first name, surname or full name)? Answer no if it is used as a "
    "place, building, organisation, brand, product, event or any other thing. "
    "Answer with one word: yes or no."
)


class JudgeUnavailableError(RuntimeError):
    """The judge could not be reached after every retry."""


def answer_text(payload: dict) -> str:
    """Final answer of a chat completion; reasoning models return a list of typed blocks."""
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    return " ".join(block.get("text", "") for block in content if block.get("type") == "text")


def parse_verdict(answer: str) -> str:
    word = answer.strip().lower().strip(".!\"'* ")
    if word.startswith("yes"):
        return PERSON
    if word.startswith("no"):
        return NOT_PERSON
    return UNPARSED


def http_transport(api_key: str, url: str, timeout: float) -> Callable[[dict], dict]:
    def post(body: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    return post


class PersonJudge:
    """Asks the model once per text, in parallel; `verdicts` keeps the input order."""

    def __init__(
        self,
        model: str,
        post: Callable[[dict], dict],
        max_workers: int = 32,
        attempts: int = 4,
        retry_seconds: float = 2.0,
        max_tokens: int = 700,
    ) -> None:
        self.model = model
        self.post = post
        self.max_workers = max_workers
        self.attempts = attempts
        self.retry_seconds = retry_seconds
        self.max_tokens = max_tokens

    def verdicts(self, names: list[str], texts: list[str]) -> list[str]:
        """PERSON, NOT_PERSON or UNPARSED per text; raises JudgeUnavailableError on outage."""
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._verdict, names, texts))

    def _verdict(self, name: str, text: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": QUESTION.format(text=text, name=name)}],
        }
        verdict = UNPARSED
        for attempt in range(self.attempts):
            try:
                verdict = parse_verdict(answer_text(self.post(body)))
            except (OSError, KeyError, IndexError, ValueError) as error:
                if attempt == self.attempts - 1:
                    raise JudgeUnavailableError(
                        f"person judge {self.model} failed {self.attempts} times: {error!r}"
                    ) from error
                time.sleep(self.retry_seconds * (attempt + 1))
                continue
            if verdict != UNPARSED:
                break
        return verdict
