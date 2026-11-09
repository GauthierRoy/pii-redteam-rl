"""Write 5 batch prompts (30 slots each) for name-free short negatives (short_eval_neg_v1).

Usage: python3 scripts/make_short_eval_neg_prompts.py <out_dir>
Each prompt was sent to GLM-5.3 via Mistral Vibe headless (`vibe -p`, agent `ask`).
"""

import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

CONTEXTS = [
    "appointment reminder",
    "chat message",
    "email with signature",
    "form/record line",
    "forwarded email thread",
    "informal text with typos",
    "list/table row",
    "meeting notes",
    "support ticket",
    "third-person report",
]
LENGTHS = [[20, 30], [31, 45], [46, 60]]
DISTRACTORS = {
    "place": "capitalized place names: cities, streets, buildings, rooms, regions (e.g. Lisbon, Harbor Road, Building C)",
    "org_product": "capitalized organization, team, brand, product or software names (e.g. Northwind Logistics, the Falcon release, Excel)",
    "calendar": "capitalized days, months, holidays, events, course or project codenames (e.g. Thursday, March, Q3 Offsite, Project Atlas)",
    "homograph": "ordinary words that can also be first names or surnames but are used here in their NON-name sense (e.g. 'the May deadline', 'we will', 'a rose bush', 'frank feedback', 'the Bill of materials', 'Grace period', 'Sterling silver', 'the Jordan river')",
    "role_only": "people referred to ONLY by role, title or pronoun, never by name (e.g. 'the duty manager', 'our landlord', 'the new hire', 'she', 'the caller')",
}
slots = [(c, n, d) for c in CONTEXTS for n in LENGTHS for d in DISTRACTORS]  # 150

RULES = """You are writing evaluation data for a PERSON-name detector. Write short, realistic English messages that contain NO person names at all.

Hard rules for every text:
- No first names, surnames, nicknames, initials-as-names, usernames/handles, or email addresses containing names.
- No places, companies or products named after a person (no "St. Mary's", "Johnson & Johnson", "Washington", "Ford", "Disney", "Wells Fargo", eponymous diseases).
- No famous people, fictional characters, or pets with names.
- Signatures, greetings and sign-offs must use a role or team ("Front desk team", "IT Support", "Thanks, the night shift") instead of a name.
- Each text must clearly feature the requested DISTRACTOR TYPE at least twice: capitalized or name-like words that are NOT people.
- Word count must fall inside the requested range (count whitespace-separated words).
- Write each text fresh: vary openings, vocabulary and structure; no filler sentences, no repeated sentences, no placeholders like [Name] or XXX.
- Match the context type's real-world format (e.g. "informal text with typos" really has typos; "list/table row" looks like a row; "forwarded email thread" shows a forwarded header without names).

Output ONLY JSON Lines, one object per slot, in slot order, with exactly these keys:
{"idx": <int>, "context_type": <str>, "target_length": [min,max], "distractor": <str>, "text": <str>}
No prose, no markdown fences, no commentary. Do not use any tools; just print the lines.

Slots:
"""

for b in range(5):
    lines = []
    for i in range(b * 30, b * 30 + 30):
        c, n, d = slots[i]
        lines.append(
            json.dumps(
                {
                    "idx": i,
                    "context_type": c,
                    "target_length": n,
                    "distractor": d,
                    "distractor_hint": DISTRACTORS[d],
                }
            )
        )
    (out / f"prompt_{b * 30:03d}_{b * 30 + 29:03d}.txt").write_text(RULES + "\n".join(lines) + "\n")
print("ok", len(slots))
