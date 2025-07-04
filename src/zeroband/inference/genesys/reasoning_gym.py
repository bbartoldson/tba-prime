import re

from reasoning_gym.factory import get_score_answer_fn


def _extract_post_string(completion: str) -> str | None:
    """Extract the model's predicted answer.

    Priority order:
    1. Anything enclosed by <answer> ... </answer> (preferred new format).
    2. Text following ``Final Answer:`` after an optional ``</think>`` tag
       (legacy format used previously).
    """
    # New preferred format: <answer> ... </answer>
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Legacy format: after </think> and 'Final Answer:'
    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None

    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None

    return final_response[1].strip()


def verify_reasoning_gym(completion: str, verification_info: dict) -> float:
    """Score ``completion`` for a Reasoning-Gym task.

    ``verification_info`` is expected to contain:
        dataset_name – the RG task name used to generate the sample
        entry        – the original sample dict (question/answer/metadata…)
    """

    dataset_name = verification_info["dataset_name"]
    entry = verification_info["entry"]

    answer = _extract_post_string(completion)
    if answer is None:
        return 0.0

    score_answer_fn = get_score_answer_fn(name=dataset_name)
    return score_answer_fn(answer=answer, entry=entry)
