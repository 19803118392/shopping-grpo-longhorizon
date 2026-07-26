"""Small, explicit progress tracker for Environment v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from web_agent_site.engine.search_v2 import normalize_query


PAGINATION_ACTIONS = {"next >", "< prev"}


def canonical_action(action_name: str, action_argument: object) -> str:
    argument = (
        normalize_query(action_argument)
        if action_name == "search"
        else str(action_argument or "").strip().casefold()
    )
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
    opened_asins: set[str] = field(default_factory=set)

    def record(self, action_name: str, action_argument: object, visible_asins=()):
        self.steps += 1
        signature = canonical_action(action_name, action_argument)
        if signature == self.last_signature:
            self.consecutive_repeats += 1
        else:
            self.consecutive_repeats = 0
        self.last_signature = signature

        visible = {str(asin) for asin in visible_asins}
        normalized_name = str(action_name or "").strip().casefold()
        normalized_argument = str(action_argument or "").strip().casefold()
        discovery_attempted = (
            normalized_name == "search"
            or (
                normalized_name == "click"
                and normalized_argument in PAGINATION_ACTIONS
            )
        )
        candidate_opened = (
            normalized_name == "click"
            and normalized_argument.isdigit()
            and len(normalized_argument) == 12
        )

        new_asins = set()
        newly_opened_asins = set()
        if discovery_attempted:
            new_asins = visible - self.seen_asins
            self.seen_asins.update(visible)
            if new_asins:
                self.no_new_asin_steps = 0
            else:
                self.no_new_asin_steps += 1
        elif candidate_opened:
            opened = visible or {normalized_argument}
            newly_opened_asins = opened - self.opened_asins
            self.opened_asins.update(opened)
            if newly_opened_asins:
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
            "discovery_attempted": discovery_attempted,
            "candidate_opened": candidate_opened,
            "new_asin_count": len(new_asins),
            "newly_opened_asin_count": len(newly_opened_asins),
            "no_new_asin_steps": self.no_new_asin_steps,
            "seen_asin_count": len(self.seen_asins),
            "opened_asin_count": len(self.opened_asins),
        }
