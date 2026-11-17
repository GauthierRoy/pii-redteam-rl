"""M06 non-RL augmentation baselines, matched to the RL set `rl_aug_v1` (316 rows).

- Templates (M06.1): fixed sentence templates per context type, filled with the same 316 names
  and context types as `rl_aug_v1`, so the arms differ only in how the texts were written.
- Plain generation (M06.2): the base attacker's first sample per request, kept when it passes
  the same acceptance as the RL set (validity, person judge, no other person by the audit).
- Generate-and-filter (M06.3): from 4 samples per request, the accepted texts the target
  detector finds hardest (lowest mean probability on the name), at most one per name and at
  most `HARD_CONTEXT_CAP` per context type so one message kind cannot fill the set.

Every arm writes rows in the `rl_aug_v1` format: one PERSON entity and its character span.
"""

from __future__ import annotations

import random

ARM_ROWS = 316  # rows in rl_aug_v1
HARD_CONTEXT_CAP = 48  # about 1.5x an even share of 316 over 10 context types

TEMPLATES: dict[str, tuple[str, ...]] = {
    "support_ticket": (
        "Ticket #{n}: {name} reports that the login page keeps timing out. Please investigate.",
        "Hello {name}, thanks for contacting support. Your refund request #{n} has been approved.",
        "Customer {name} cannot download the invoice for order {n}. Priority: medium.",
        "Hi, the replacement part for ticket {n} was shipped today. Regards, {name}",
        "Update on case {n}: we called {name} and the issue is resolved.",
    ),
    "email_with_signature": (
        "Hi team, the slides for Thursday are in the shared folder.\n\nBest,\n{name}",
        "Thanks for the quick reply. I will send the signed copy tomorrow.\n\nKind regards,\n{name}",
        "Hello, please find the updated budget attached.\n\nCheers,\n{name}\nFinance",
        "Dear {name},\n\nYour booking {n} is confirmed for next week.\n\nBest regards,\nThe Events Team",
        "Quick note: the office is closed on Monday.\n\nThanks,\n{name}",
    ),
    "forwarded_email_thread": (
        "---------- Forwarded message ----------\nFrom: {name}\nSubject: Q{q} report\n\n"
        "See the numbers below.",
        "FYI, forwarding this from {name}.\n\n> Can we move the call to 3 pm?",
        "Fwd: Delivery {n}\n\n> {name} wrote: The package arrived damaged, please advise.",
        "Forwarding for visibility.\n\nOn Monday, {name} wrote:\n> The contract is ready to sign.",
        "-----Original Message-----\nFrom: {name}\nSent: Friday\nSubject: Invoice {n}\n\nPaid today.",
    ),
    "meeting_notes": (
        "Meeting notes, project kickoff. Attendees: {name}, design team. Action: draft plan by "
        "Friday.",
        "Weekly sync: {name} presented the Q{q} roadmap. Next meeting on Tuesday.",
        "Notes: budget approved. {name} to follow up with the vendor.",
        "Standup: {name} finished the migration and is now on testing.",
        "Action items: 1) {name} sends the minutes 2) review the open tickets.",
    ),
    "chat_message": (
        "hey {name}, are we still on for lunch?",
        "{name} just sent the photos, check the group chat",
        "ok {name}, see you at 7!",
        "did {name} reply to you yet?",
        "thanks {name}, that really helped :)",
    ),
    "appointment_reminder": (
        "Reminder: {name}, your appointment is on Tuesday at 10:30. Reply C to confirm.",
        "Hi {name}, this is a reminder of your dental check-up tomorrow at 2 pm.",
        "Appointment confirmed for {name} on the {d}th at 9:00. Please arrive 10 minutes early.",
        "Your consultation with our clinic is booked for Friday. Patient: {name}.",
        "Don't forget: {name}, your car service is scheduled for Monday morning.",
    ),
    "form_record_line": (
        "Name: {name} | ID: {n} | Status: active",
        "Applicant: {name}; Date: 2026-03-{d}; Ref: {n}",
        "Record {n} - holder {name} - renewal due",
        "Full name: {name}\nMembership: {n}\nPlan: basic",
        "Signed by: {name}  Date: {d}/04/2026",
    ),
    "list_or_table_row": (
        "{n} | {name} | Sales | Paris",
        "- {name}: 3 items, shipped",
        "Row {d}: {name}, room 2{d}, checked in",
        "{name}\t{n}\tpending",
        "Winner #{d}: {name} ({n} points)",
    ),
    "informal_text_with_typos": (
        "omg {name} forgot the keys agian lol",
        "cant make it tonite, tell {name} sory",
        "{name} sed the movie was rly good",
        "wher is {name}?? we r waiting",
        "thx for the ride, say hi to {name} pls",
    ),
    "third_person_report": (
        "{name} joined the company in 2019 and now leads the support team.",
        "According to the report, {name} completed the training on time.",
        "The inspector noted that {name} was present during the audit.",
        "Last week {name} submitted the final version of the proposal.",
        "The committee thanked {name} for organising the event.",
    ),
}


def fill_template(template: str, name: str, rng: random.Random) -> tuple[str, list[int]]:
    """The filled text and the name's character span; `name` occurs exactly once."""
    values = {
        "n": str(rng.randint(1000, 99999)),
        "q": str(rng.randint(1, 4)),
        "d": str(rng.randint(10, 28)),
    }
    before, after = template.split("{name}")
    head = before.format(**values)
    text = head + name + after.format(**values)
    if text.count(name) != 1:
        raise ValueError(f"name {name!r} does not occur once in {text!r}")
    return text, [len(head), len(head) + len(name)]


def template_rows(rl_rows: list[dict], seed: int) -> list[dict]:
    """One template text per `rl_aug_v1` row, on the same name and context type."""
    rng = random.Random(seed)
    rows = []
    for index, rl_row in enumerate(rl_rows):
        name = rl_row["request"]["entities"][0]["value"]
        context = rl_row["request"]["context_type"]
        text, span = fill_template(rng.choice(TEMPLATES[context]), name, rng)
        rows.append(
            {
                "id": f"template-aug-v1-{index:05d}",
                "split": "augmentation",
                "request": {
                    "language": "en",
                    "context_type": context,
                    "entities": [{"type": "PERSON", "value": name}],
                },
                "target": {"text": text, "span": span},
                "provenance": "template:baselines.TEMPLATES",
                "source_request_id": rl_row["source_request_id"],
            }
        )
    return rows


def plain_selection(accepted: list[dict], n: int = ARM_ROWS) -> list[dict]:
    """Ordinary generation: accepted first samples (sample_idx 0), in request order."""
    first = [row for row in accepted if row.get("sample_idx", 0) == 0]
    return sorted(first, key=lambda row: row["request_id"])[:n]


def hard_selection(
    accepted: list[dict], n: int = ARM_ROWS, context_cap: int = HARD_CONTEXT_CAP
) -> list[dict]:
    """Generate-and-filter: lowest target-detector probability first, one row per name, at most
    `context_cap` rows per context type."""
    by_name: dict[str, dict] = {}
    for row in accepted:
        best = by_name.get(row["name"])
        if best is None or row["p_target_mean"] < best["p_target_mean"]:
            by_name[row["name"]] = row
    chosen: list[dict] = []
    per_context: dict[str, int] = {}
    for row in sorted(by_name.values(), key=lambda r: (r["p_target_mean"], r["request_id"])):
        if per_context.get(row["context_type"], 0) >= context_cap:
            continue
        per_context[row["context_type"]] = per_context.get(row["context_type"], 0) + 1
        chosen.append(row)
        if len(chosen) == n:
            break
    return chosen


def aug_row(dataset: str, index: int, row: dict, provenance: str) -> dict:
    """An accepted generation row in the `rl_aug_v1` format."""
    start, end = row["name_span"]
    if row["text"][start:end] != row["name"]:
        raise ValueError(f"{row['request_id']}: span is not the name")
    return {
        "id": f"{dataset}-{index:05d}",
        "split": "augmentation",
        "request": {
            "language": "en",
            "context_type": row["context_type"],
            "entities": [{"type": "PERSON", "value": row["name"]}],
        },
        "target": {"text": row["text"], "span": [start, end]},
        "provenance": provenance,
        "source_request_id": row["request_id"],
        "target_detector_p_name": row["p_target_mean"],
    }
