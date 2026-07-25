"""veRL 0.8 ToolAgentLoop 的 ShopSimulator 轨迹生命周期适配。"""

from __future__ import annotations

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from shopping_grpo.context_window import ContextBudgetError, compact_token_trajectory
from shopping_grpo.verl_adapter.runtime import (
    current_runtime_state,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.verl_adapter.session import ShopSimulatorSession


class ShoppingToolAgentLoop(ToolAgentLoop):
    """Vanilla ToolAgentLoop with deterministic ShopSimulator termination and release."""

    def __init__(
        self,
        *args,
        base_url="http://127.0.0.1:5700",
        timeout=60,
        max_steps=35,
        reward_mode="native",
        context_window_tokens=24576,
        context_generation_reserve_tokens=512,
        context_safety_margin_tokens=512,
        context_input_budget_tokens=16384,
        context_preserve_recent_groups=1,
        env_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.reward_mode = str(reward_mode)
        self.context_window_tokens = int(context_window_tokens)
        self.context_generation_reserve_tokens = int(context_generation_reserve_tokens)
        self.context_safety_margin_tokens = int(context_safety_margin_tokens)
        self.context_input_budget_tokens = int(context_input_budget_tokens)
        self.context_preserve_recent_groups = int(context_preserve_recent_groups)
        maximum_context_input = (
            self.context_window_tokens
            - self.context_generation_reserve_tokens
            - self.context_safety_margin_tokens
        )
        if not 0 < self.context_input_budget_tokens <= maximum_context_input:
            raise ValueError(
                "context_input_budget_tokens must be positive and fit the model context window"
            )
        self.context_input_budget = self.context_input_budget_tokens
        if self.context_preserve_recent_groups < 1:
            raise ValueError("context_preserve_recent_groups must be positive")
        if self.reward_mode not in {"native", "constraint_aware"}:
            raise ValueError(f"unknown shopping reward mode: {self.reward_mode!r}")
        self.env_factory = env_factory

    async def _handle_generating_state(
        self,
        agent_data,
        sampling_params,
        ignore_termination=False,
    ):
        runtime_state = current_runtime_state.get()
        try:
            prompt_ids, response_mask, response_logprobs, stats = compact_token_trajectory(
                agent_data.prompt_ids,
                agent_data.response_mask,
                agent_data.response_logprobs,
                max_input_tokens=self.context_input_budget,
                preserve_recent_groups=self.context_preserve_recent_groups,
            )
        except ContextBudgetError as exc:
            if runtime_state is not None:
                runtime_state["terminate"] = True
                runtime_state["termination_reason"] = "context_budget_exhausted"
                runtime_state["error"] = f"context_budget_exhausted:{exc}"
                runtime_state["infrastructure_invalid"] = True
            return AgentState.TERMINATED
        if stats.removed_tokens:
            if agent_data.routed_experts is not None:
                if runtime_state is not None:
                    runtime_state["terminate"] = True
                    runtime_state["termination_reason"] = "context_compaction_unsupported_routed_experts"
                    runtime_state["error"] = runtime_state["termination_reason"]
                    runtime_state["infrastructure_invalid"] = True
                return AgentState.TERMINATED
            agent_data.prompt_ids = prompt_ids
            agent_data.response_mask = response_mask
            agent_data.response_logprobs = response_logprobs
            if runtime_state is not None:
                runtime_state["context_compactions"] += 1
                runtime_state["context_tokens_removed"] += stats.removed_tokens
        bounded_sampling_params = dict(sampling_params)
        if "max_tokens" in bounded_sampling_params:
            bounded_sampling_params["max_tokens"] = min(
                int(bounded_sampling_params["max_tokens"]),
                self.context_generation_reserve_tokens,
            )
        return await super()._handle_generating_state(
            agent_data,
            bounded_sampling_params,
            ignore_termination=ignore_termination,
        )

    async def _handle_processing_tools_state(self, agent_data):
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and len(agent_data.tool_calls) > 1:
            runtime_state["terminate"] = True
            runtime_state["termination_reason"] = "parallel_tool_calls"
            runtime_state["error"] = "parallel_tool_calls"
            return AgentState.TERMINATED
        next_state = await super()._handle_processing_tools_state(agent_data)
        runtime_state = current_runtime_state.get()
        if runtime_state is not None and runtime_state.get("terminate"):
            return AgentState.TERMINATED
        return next_state

    async def run(self, sampling_params, **kwargs):
        task_id = task_id_from_kwargs(kwargs)
        session = ShopSimulatorSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            if not state["done"] and not state["error"]:
                state["error"] = "assistant_finished_without_environment_done"
                state["termination_reason"] = state["error"]
                state["terminate"] = True
            breakdown = reward_breakdown(state)
            output.reward_score = terminal_reward(state, mode=self.reward_mode)
            output.extra_fields["shopping"] = {
                "task_id": task_id,
                "steps": len(state["steps"]),
                "done": bool(state["done"]),
                "termination_reason": state["termination_reason"],
                "error": state["error"],
                "infrastructure_invalid": bool(state["infrastructure_invalid"]),
                "action_attempts": int(state["action_attempt_count"]),
                "repeat_actions": int(state["repeat_action_count"]),
                "reward_mode": self.reward_mode,
                "reward": breakdown,
                "context_compactions": int(state["context_compactions"]),
                "context_tokens_removed": int(state["context_tokens_removed"]),
            }
            output.metrics["shopping_context/compactions"] = int(state["context_compactions"])
            output.metrics["shopping_context/tokens_removed"] = int(
                state["context_tokens_removed"]
            )
            return output
        finally:
            await session.close()
