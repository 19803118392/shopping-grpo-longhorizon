"""Small, explicit progress tracker for Environment v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from web_agent_site.engine.search_v2 import normalize_query


def canonical_action(action_name: str, action_argument: object) -> str:
    argument = normalize_query(action_argument) if action_name == "search" else str(action_argument or "").strip().casefold()
    return json.dumps(
        {"action": str(action_name).strip().casefold(), "argument": argument},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass
class ProgressTracker:
    max_steps: int = 35
    exact_repeat_limit: int = 2
    no_new_asin_limit: int = 4
    steps: int = 0
    last_signature: str | None = None
    consecutive_repeats: int = 0
    no_new_asin_steps: int = 0
    seen_asins: set[str] = field(default_factory=set)

    def record(self, action_name: str, action_argument: object, visible_asins=()):
        self.steps += 1
        signature = canonical_action(action_name, action_argument)
        if signature == self.last_signature:
            self.consecutive_repeats += 1
        else:
            self.consecutive_repeats = 0
        self.last_signature = signature

        visible = {str(asin) for asin in visible_asins}
        new_asins = visible - self.seen_asins
        self.seen_asins.update(visible)
        if new_asins:
            self.no_new_asin_steps = 0
        else:
            self.no_new_asin_steps += 1

        reason = None
        if self.consecutive_repeats >= self.exact_repeat_limit:
            reason = "repeat_loop"
        elif self.no_new_asin_steps >= self.no_new_asin_limit:
            reason = "repeat_loop"
        elif self.steps >= self.max_steps:
            reason = "max_steps"
        return {
            "termination_reason": reason,
            "step_count": self.steps,
            "action_signature": signature,
            "consecutive_repeats": self.consecutive_repeats,
            "new_asin_count": len(new_asins),
            "no_new_asin_steps": self.no_new_asin_steps,
            "seen_asin_count": len(self.seen_asins),
        }
