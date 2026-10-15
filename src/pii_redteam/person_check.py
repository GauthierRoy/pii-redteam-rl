"""Independent person checks for D1 training-data acceptance (M07b.2, D14/D15 era).

The extra-person check must NOT rely on D0 (D13: D0 misses ~50% of names in
short conversational texts). Two independent signals:

  1. spaCy general-domain NER (`en_core_web_sm`): any PERSON entity besides the
     requested name is an extra person mention.
  2. A small LLM judge (base Qwen3.5-0.8B, non-thinking, greedy yes/no):
     - extra-person question: does the text mention any person other than the
       requested name?
     - person-sense question: is the requested name used as a PERSON (a
       human), not a place / organization / object / title of a work?

A row is REJECTED for D1 training data if any extra person is found (spaCy OR
LLM) or the person-sense answer is "no". The LLM judge is validated by a manual
audit of 50 rows (docs/PROMPT_V2_RESULTS.md).
"""

from __future__ import annotations

import re


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_same_name(entity: str, target: str) -> bool:
    """True when a NER entity refers to the requested name (allowing titles)."""
    ent, tgt = _norm(entity), _norm(target)
    if not ent or not tgt:
        return False
    if ent == tgt:
        return True
    # "Mr. Gulzar" / "Dr. Ana Ruiz" -> compare the name part only
    ent_stripped = re.sub(r"^(mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+", "", ent)
    # "F. Bettschen" -> compare the surname part ("Bettschen" in "Fithi Bettschen")
    ent_stripped = re.sub(r"^[a-z]\.?\s+", "", ent_stripped)
    if ent_stripped == tgt:
        return True
    # one contained in the other ("F. Bettschen" vs "Fithi Bettschen")
    return tgt in ent_stripped or ent_stripped in tgt


# spaCy PERSON false-positive guards (manual audit of run m07b-20261012-224819,
# 50-row audit: "Q3", "RSVP", "Team Lead" were tagged PERSON). Names do not
# contain digits; signature role titles are not person names.
_ROLE_TITLE_RE = re.compile(
    r"^(the )?(team lead|team|customer|guest|client|manager|foreman|sender|recipient)$"
)


def is_plausible_person_name(entity: str) -> bool:
    """Guard against spaCy PERSON false positives (digits, role titles)."""
    text = entity.strip()
    if not text or any(ch.isdigit() for ch in text):
        return False
    return not _ROLE_TITLE_RE.match(text.lower())


def spacy_extra_persons(doc, target: str) -> list[str]:
    """PERSON entities in the doc that are not the requested name."""
    extras = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        if is_same_name(ent.text, target):
            continue
        if not is_plausible_person_name(ent.text):
            continue
        extras.append(ent.text.strip())
    return sorted(set(extras))


EXTRA_PERSON_PROMPT = (
    "Answer with one word: yes or no.\n"
    "Text: {text}\n"
    "Does the text name any person other than {name}? Answer:"
)

PERSON_SENSE_PROMPT = (
    "Answer with one word: yes or no.\n"
    "Text: {text}\n"
    'Is "{name}" used as a human person\'s name (not a place, organization, '
    "product, or work)? Answer:"
)


def parse_yes_no(raw: str) -> bool | None:
    """Parse the judge's answer; None when the judge did not answer yes/no."""
    token = raw.strip().lower()
    match = re.search(r"\b(yes|no)\b", token)
    if not match:
        return None
    return match.group(1) == "yes"
