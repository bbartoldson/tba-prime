"""Unit test for per-row checkpoint steps in get_parquet_table (within-group staleness mixing)."""

from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.rewards import CompletionReward, RequestRewards


class FakeCompletionOutput:
    """Minimal stand-in for vllm.CompletionOutput with the attributes get_parquet_table touches."""

    def __init__(self, index: int, token_ids: list[int], text: str):
        self.index = index
        self.token_ids = token_ids
        self.text = text
        self.logprobs = None


class FakeRequestOutput:
    """Minimal stand-in for vllm.RequestOutput with the attributes get_parquet_table touches."""

    def __init__(self, request_id: str, prompt_token_ids: list[int], outputs: list[FakeCompletionOutput]):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs


def _build_inputs(num_requests: int = 2, num_outputs: int = 4):
    request_outputs = []
    request_rewards = []
    for r in range(num_requests):
        request_id = f"req_{r}"
        outputs = [FakeCompletionOutput(index=i, token_ids=[1, 2], text=f"completion {r}-{i}") for i in range(num_outputs)]
        request_outputs.append(FakeRequestOutput(request_id=request_id, prompt_token_ids=[10, 11, 12], outputs=outputs))
        request_rewards.append(
            RequestRewards(
                request_id=request_id,
                rewards=[
                    CompletionReward(completion_id=i, reward=1.0, task_reward=1.0, length_penalty=0.0, advantage=0.0)
                    for i in range(num_outputs)
                ],
                task_type="null_reward",
            )
        )
    prompts = [f"prompt {r}" for r in range(num_requests)]
    proofs = [b"proof"] * (num_requests * num_outputs)
    target_lengths = [0] * num_requests
    problems = [{"problem_id": f"problem_{r}"} for r in range(num_requests)]
    seeds = list(range(num_outputs))
    return request_outputs, request_rewards, prompts, proofs, target_lengths, problems, seeds


def test_per_output_steps_stamps_rows():
    request_outputs, request_rewards, prompts, proofs, target_lengths, problems, seeds = _build_inputs()
    per_output_steps = [[9, 9, 3, 3], [9, 9, 3, 3]]

    table = get_parquet_table(
        request_outputs,
        request_rewards,
        prompts,
        proofs,
        step=42,
        target_lengths=target_lengths,
        problems=problems,
        enable_logprobs=False,
        seeds=seeds,
        temperature=1.0,
        per_output_steps=per_output_steps,
    )

    assert table.column("step").to_pylist() == [9, 9, 3, 3, 9, 9, 3, 3]


def test_scalar_step_without_per_output_steps():
    request_outputs, request_rewards, prompts, proofs, target_lengths, problems, seeds = _build_inputs()

    table = get_parquet_table(
        request_outputs,
        request_rewards,
        prompts,
        proofs,
        step=42,
        target_lengths=target_lengths,
        problems=problems,
        enable_logprobs=False,
        seeds=seeds,
        temperature=1.0,
    )

    assert table.column("step").to_pylist() == [42] * 8
