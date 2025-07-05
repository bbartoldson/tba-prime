# (Jack): This is an umerged patch to fix a bug in vllm https://github.com/vllm-project/vllm/pull/19940
# This can be removed once the patch is merged and vllm is updated.
import zeroband.inference.monkeypatch_sampling_metadata  # noqa: F401

import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
import uuid
from typing import List, Tuple

# Import environment before any other imports
# ruff: noqa: I001
from zeroband.inference import envs

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import requests
import torch
from datasets import load_dataset, load_from_disk
from toploc.utils import sha256sum
import zeroband.vllm_08_shim # FIX?
from vllm import SamplingParams, TokensPrompt
from huggingface_hub import snapshot_download

from zeroband.utils.pydantic_config import parse_argv
from zeroband.eval.utils import run_benchmark
from zeroband.inference.config import Config as InferenceConfig
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.pipeline import all_reduce, patch_model_load, setup_comm, setup_hooks
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.inference.toploc import setup_toploc_cache
from zeroband.inference.toploc2 import Toploc2Sampler
from zeroband.utils.monitor import setup_monitor
from zeroband.inference.utils import (
    setup_model,
    filter_data_by_prompt_length,
    reload_checkpoint,
    reload_model_weights,
    compute_max_batch_size,
    get_inference_input_output_flops,
    generate_target_lengths,
    format_prompts,
)
from zeroband.training.mp import EnvWrapper
from zeroband.utils.utils import clean_exit
from zeroband.inference.logger import setup_logger


def build_privileged_prompt(
    prompt: str,
    answer: str,
    template: str,
) -> str:
    """
    Insert the ground-truth `answer` into a privileged prompt template.

    Default template (configurable via `config.data.privileged_template`):

        {prompt}

        # Ground-truth answer
        {answer}

        Please produce a detailed chain-of-thought that shows how one
        arrives at the answer step-by-step.

    """
    return template.format(prompt=prompt, answer=answer)


def expand_batch_for_dual_prompts(
    batch,
    privileged_template: str,
) -> Tuple[
    List[str],  # prompts
    List[dict],  # verification_infos
    List[str],  # task_types
    List[bool],  # is_privileged flags, aligned with prompts list
]:
    """
    For every item in `batch`, return [regular, privileged] variants.
    """
    out_prompts: List[str] = []
    out_verif: List[dict] = []
    out_task_types: List[str] = []
    out_priv_flags: List[bool] = []
    out_ground_truths: List[str] = []

    for item in batch:
        verif = json.loads(item["verification_info"])
        task_type = item["task_type"]

        # 1. regular
        out_prompts.append(item["prompt"])
        out_verif.append(verif)
        out_task_types.append(task_type)
        out_priv_flags.append(False)
        out_ground_truths.append(verif["ground_truth"])

        # 2. privileged
        priv_prompt = build_privileged_prompt(
            item["prompt"],
            verif["ground_truth"],  # dataset **must** expose ground-truth answer
            privileged_template,
        )
        out_prompts.append(priv_prompt)
        out_verif.append(verif)  # identical verification meta
        out_task_types.append(task_type)
        out_priv_flags.append(True)
        out_ground_truths.append(verif["ground_truth"])

    return out_prompts, out_verif, out_task_types, out_priv_flags, out_ground_truths


@clean_exit
def inference(config: InferenceConfig):
    # Initialize the logger
    dp_rank = int(os.environ.get("DP_RANK", 0))
    logger = setup_logger(config.log, parallel_config=config.parallel, dp_rank=dp_rank)
    logger.info("Starting inference")

    # Optionally, clean the rollout path
    if config.clean_rollout_path and config.rollout_path is not None:
        logger.info(f"Cleaning rollout path ({config.rollout_path})")
        shutil.rmtree(config.rollout_path, ignore_errors=True)

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initialize metrics
    monitor = setup_monitor(config.monitor, config.task_id, config)

    # Patch vLLM's model loading to load model shard
    patch_model_load(config=config.parallel.pp)

    # Initialize model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    tokenizer = llm.get_tokenizer()
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    if config.toploc.enable_toploc2:
        llm.llm_engine.model_executor.driver_worker.model_runner.sampler = Toploc2Sampler()
        logger.info("Using toploc2 sampler")

    # Initialize dataset
    logger.info(f"Initializing dataset (name={config.data.name}, split={config.data.split})")
    start_time = time.time()
    dataset = load_dataset(config.data.name, split=config.data.split) #load_from_disk(config.data.name)#load_dataset(config.data.name, split=config.data.split)

    if not config.rewards.compute_reward:
        logger.info("Reward computation is disabled, setting task_type to null_reward")
        dataset = dataset.map(lambda _: {"task_type": "null_reward"})

    logger.success(f"Initialized dataset with {len(dataset):,} problems in {time.time() - start_time:.2f}s")

    # Optionally shuffle dataset
    if config.group_id is not None:
        # We dont shuffle here because we shuffle reproducibly in the sampling loop.
        assert config.seed is None, "Seed is not supported when group ID is set"
        assert config.parallel.dp == 1, "DP is not supported when group ID is set"
        node_address_int = int(config.group_id, 16)
        logger.info(f"Seeding with {node_address_int} ({config.group_id})")
    else:
        # Seed the dataset with a random number
        seed = config.seed + int(os.environ.get("DP_RANK", 0)) if config.seed is not None else None
        generator = np.random.default_rng(seed)
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(generator=generator)
        node_address_int = None

    # Optionally, filter out prompts that are too long
    if config.data.max_prompt_len:
        logger.info(f"Filtering out prompts with more than {config.data.max_prompt_len} tokens")
        start_time = time.time()
        dataset = filter_data_by_prompt_length(dataset, config.data.max_prompt_len, tokenizer)
        logger.success(f"Filtered long prompts in {time.time() - start_time:.2f}s - {len(dataset)} samples remaining")

    # Optionally, filter dataset for samples within difficulty range
    if config.data.difficulty_filtering:
        logger.info(
            f"Filtering dataset for difficulty in [{config.data.difficulty_filtering.min_solve_rate}, {config.data.difficulty_filtering.max_solve_rate}]"
        )
        dataset = dataset.filter(
            lambda x: x[config.data.difficulty_filtering.solve_rate_field] >= config.data.difficulty_filtering.min_solve_rate
            and x[config.data.difficulty_filtering.solve_rate_field] <= config.data.difficulty_filtering.max_solve_rate
        )

    # Effective batch size 2x due to privileged and non privileged prompts
    prompts_per_query = 2
    effective_samples_per_problem = prompts_per_query * config.sampling.n
    
    # Initialize sampling parameters
    logger.info(f"Initializing sampling parameters ({config.sampling})")
    sampling_params = SamplingParams(**config.sampling.model_dump())

    # Setup pipeline parallel communication and hook
    node = setup_comm(config.parallel.pp)
    setup_hooks(llm, config.parallel.pp, node)

    # Compute the maximum batch size
    max_batch_size = config.max_batch_size
    if max_batch_size == "auto":
        # Automatically compute the maximum batch size
        logger.info("Auto-computing maximum batch size")
        local_max_batch_size = compute_max_batch_size(llm)
        max_batch_size = all_reduce(node, torch.tensor(local_max_batch_size), config=config.parallel.pp, op=torch.min).item()
        logger.info(f"Auto-computed maximum batch size: {max_batch_size}")
        max_batch_size = int(max_batch_size * config.scale_factor)
        logger.info(f"Scaled maximum batch size by {config.scale_factor} to {max_batch_size}")

    problems_per_batch = max_batch_size // effective_samples_per_problem
    batch_size = problems_per_batch * effective_samples_per_problem

    logger.info(f"Using maximum batch size: {max_batch_size}")

    # Throw an error if the batch cannot fit number of samples to generate per problem.
    # TODO(Mika): Circumvent this assertion by distribtuting parallel samples across multiple batches
    if config.sampling.n > max_batch_size:
        raise ValueError(f"Sampling.n ({config.sampling.n}) must be less than or equal to max_batch_size ({max_batch_size})")

    # Compute the true batch size
    problems_per_batch = max_batch_size // config.sampling.n
    batch_size = problems_per_batch * config.sampling.n
    logger.info(
        f"Problems per batch: {max_batch_size} // {config.sampling.n} = {problems_per_batch}, batch size: {problems_per_batch} * {config.sampling.n} = {batch_size} (missing: {max_batch_size % config.sampling.n})"
    )

    # Setup TOPLOC
    hidden_size = llm.llm_engine.model_executor.driver_worker.model_runner.model.config.hidden_size
    toploc_cache, _ = setup_toploc_cache(
        llm,
        pp_config=config.parallel.pp,
        disable=not config.toploc.enable_toploc1,
        max_seqs=batch_size,
        hidden_size=hidden_size,
        topk=config.toploc.topk,
    )

    ckpt_step = 0
    step = config.start_step
    if config.rl and config.rl.ckpt_start_path is not None:
        logger.info(f"Resuming from checkpoint {config.rl.ckpt_start_path}")
        path = Path(config.rl.ckpt_start_path)
        path_file = path / "model.safetensors"
        if not path_file.exists():
            raise FileNotFoundError(f"Checkpoint file {path_file} does not exist")
        ckpt_step = int(path.name.split("_")[-1])
        logger.info(f"Resuming from step {ckpt_step} at {path_file}")
        llm = reload_model_weights(llm, path_file)
        step = ckpt_step

    # Check if we should resume from step_path file
    if config.step_path is not None and config.step_path.exists():
        try:
            saved_step = int(config.step_path.read_text().strip())
            logger.info(f"Found existing step file at {config.step_path} with step {saved_step}")
            step = saved_step
            logger.info(f"Resuming from step {step} (loaded from {config.step_path})")
        except (ValueError, IOError) as e:
            logger.warning(f"Failed to read step from {config.step_path}: {e}")

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_samples = 0
    total_tokens = 0
    last_eval_step = -1

    dataset_offset = 0
    while True:
        if config.rl and config.rl.step_endpoint is not None:
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_step = requests.get(config.rl.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.rl.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_step != step:
                step = new_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        logger.info(f"Inference step {step} (Checkpoint step: {ckpt_step})")

        # Reload model weights from checkpoint if we are too far ahead of the checkpoint step
        if config.rl and step - ckpt_step > config.rl.async_level:
            logger.warning(
                f"Hit async level ({config.rl.async_level}) because inference step {step} is {step - ckpt_step} steps ahead of checkpoint step {ckpt_step}. Trying to reload model weights from {config.rl.ckpt_path}"
            )
            ckpt_step = step - config.rl.async_level
            llm = reload_checkpoint(llm, config.rl.ckpt_path, ckpt_step)

        # Optionally, run online evals at the specified interval
        if (
            config.rl
            and dp_rank == 0
            and config.eval
            and config.eval.online
            and ckpt_step % config.eval.online.interval == 0
            and ckpt_step > last_eval_step
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            for benchmark in config.eval.benchmarks:
                run_benchmark(llm, benchmark, config.model, config.sampling, ckpt_step, seed=config.seed, use_tqdm=config.use_tqdm)

        # Write the current step to a file, this is required for resuming tasks in production but can be ignored for local runs
        if config.step_path is not None:
            logger.info(f"Writing current inference step ({step}) to {config.step_path}")
            if not config.step_path.exists():
                config.step_path.parent.mkdir(parents=True, exist_ok=True)
            config.step_path.write_text(str(step))

        # Get batch
        if node_address_int is not None:
            # TODO: What if we have multiple sample per step?
            # Its impossible right now but we need to fix this if accept counter is used.

            # We reseed the generator here to make the sampling reproducible at each step.
            # This would work even if the node restarts and resumes from the current step.
            generator = np.random.default_rng(node_address_int * current_step_batch_counter + step)
            indices = generator.integers(0, len(dataset), problems_per_batch)
            sampling_params.seed = int(generator.integers(2**32))
        else:
            # Use modulo to cycle through the dataset instead of terminating
            indices = [(dataset_offset + j) % len(dataset) for j in range(problems_per_batch)]
            if seed is not None:
                sampling_params.seed = seed + step * 1_000_000  # 1M is needed to avoid collision from sampling.n

        logger.debug(f"Sampling batch with indices [{' '.join(map(str, indices[:3]))}...{' '.join(map(str, indices[-3:]))}]")
        problems = dataset.select(indices)

        # retrieve privileged template
        privileged_template = getattr(config.data, "privileged_template")

        prompts, verification_infos, task_types, priv_flags, ground_truths = expand_batch_for_dual_prompts(
            problems, privileged_template
        )

        #verification_infos = [json.loads(item["verification_info"]) for item in problems]
        #task_types = [item["task_type"] for item in problems]
        #prompts = [item["prompt"] for item in problems]

        target_lengths = generate_target_lengths(config.rewards.len_reward, len(prompts))
        for target_length, verification_info in zip(target_lengths, verification_infos):
            verification_info["target_length"] = target_length

        # Get tokenized prompts as BatchEncoding
        tokenized_prompts = format_prompts(
            prompts,
            target_lengths,
            config.rewards.len_reward,
            tokenizer=tokenizer,
            enable_thinking=config.model.enable_thinking,
            tokenize=True,
        )

        generate_start_time = time.time()
        if config.contexts:
            # Convert tokenized prompts to TokensPrompt objects (prompt_id -> TokensPrompt)
            # TODO(jack): These can probably be lists and then we dont need the list dict sort thing going on at the end?
            unordered_token_prompts: dict[int, TokensPrompt] = {
                i: TokensPrompt(prompt_token_ids=prompt_token_ids) for i, prompt_token_ids in enumerate(tokenized_prompts)
            }
            # Save chunked request outputs (prompt_id -> RequestOutput)
            unordered_output_token_ids: dict[int, list[int]] = {i: [] for i in unordered_token_prompts.keys()}
            unordered_proofs: dict[int, bytes] = {i: b"" for i in unordered_token_prompts.keys()}

            max_model_len = llm.llm_engine.model_config.max_model_len
            assert max(config.contexts) <= max_model_len, "The final context should be less than the max model length"
            assert sorted(config.contexts) == config.contexts, "Contexts must be sorted"
            num_finished_sequences = 0
            for context in config.contexts:
                # Auto-compute the max batch size for the current context
                logger.info(f"Running at context {context}")
                local_max_batch_size = compute_max_batch_size(llm, max_model_len=context)
                if node:
                    max_batch_size = int(
                        all_reduce(node, torch.tensor(local_max_batch_size), config=config.parallel.pp, op=torch.min).item()
                    )
                else:
                    max_batch_size = local_max_batch_size
                logger.info(f"Auto-computed maximum batch size for context {context}: {max_batch_size}")

                # Micro-batch the sequences to not trigger cache eviction
                prompt_ids = list(unordered_token_prompts.keys())
                micro_batch_prompt_ids = [prompt_ids[i : i + max_batch_size] for i in range(0, len(prompt_ids), max_batch_size)]
                micro_batches = [
                    {prompt_id: unordered_token_prompts[prompt_id] for prompt_id in micro_batch_prompt_ids}
                    for micro_batch_prompt_ids in micro_batch_prompt_ids
                ]
                finish_sequence = context == max(config.contexts)  # Last context
                for mb_i, mb_unordered_tokens_prompts in enumerate(micro_batches):
                    # 1. Figure out the max_tokens to set
                    input_token_count = sum(len(token_prompt["prompt_token_ids"]) for token_prompt in mb_unordered_tokens_prompts.values())
                    cacheable_area = max_batch_size * context
                    max_tokens = (cacheable_area - input_token_count) // len(mb_unordered_tokens_prompts)

                    # TopLoc1 proofs do chunks of 32 tokens
                    # So we must have a multiple of 32 tokens
                    max_tokens = int(max_tokens / 32) * 32
                    assert max_tokens > 0, "Context must be larger than the max input tokens"
                    sampling_params.max_tokens = max_tokens

                    # 2. Generate
                    logger.info(
                        f"Generating {len(mb_unordered_tokens_prompts)} samples with context {context} for micro batch {mb_i + 1}/{len(micro_batches)} ({max_tokens} tokens)"
                    )
                    logger.info(f"Token prompt IDs: {list(mb_unordered_tokens_prompts.keys())}")
                    start_time = time.time()
                    request_outputs = llm.generate(list(mb_unordered_tokens_prompts.values()), sampling_params, use_tqdm=config.use_tqdm)
                    generation_time = time.time() - start_time
                    logger.info(f"Generated {len(request_outputs)} samples in {generation_time:.2f}s")

                    # Force generation of proofs for the remaining sequences that haven't reached max_len.
                    toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

                    # Save the TOPLOC proofs for the current chunk
                    toploc_cache.wait_for_proofs()
                    chunked_proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
                    for prompt_id, chunked_proof in zip(mb_unordered_tokens_prompts.keys(), chunked_proofs):
                        unordered_proofs[prompt_id] += chunked_proof
                    toploc_cache.reset_cache()

                    # Only keep unfinished sequences for the next iteration
                    finished_prompt_ids = []
                    for prompt_id, request_output in zip(mb_unordered_tokens_prompts.keys(), request_outputs):
                        # Note: This assumes that sampling.n == 1, else it might break
                        assert len(request_output.outputs) == 1, "Sampling.n must be 1"
                        output = request_output.outputs[0]
                        unordered_output_token_ids[prompt_id].extend(output.token_ids)
                        if (
                            output.finish_reason == "stop"
                            or finish_sequence
                            or len(request_output.prompt_token_ids) + len(output.token_ids) >= max_model_len
                        ):
                            # We remember which sequences finished, so we can pop remove them from the prompt pool
                            finished_prompt_ids.append(prompt_id)
                        else:
                            # Overwrite the token prompt so that we prefill the next chunk
                            unordered_token_prompts[prompt_id] = TokensPrompt(
                                prompt_token_ids=[*request_output.prompt_token_ids, *output.token_ids]
                            )
                    for prompt_id in finished_prompt_ids:
                        unordered_token_prompts.pop(prompt_id)

                    num_finished_sequences += len(finished_prompt_ids)
                    logger.info(f"Finished {len(finished_prompt_ids)} sequences at micro batch {mb_i} at context {context}")
                logger.info(f"Finished {num_finished_sequences} sequences at context {context}")

            # Order the request outputs by prompt_id to pretend like we generated all samples in one go
            # request_outputs = list(dict(sorted(unordered_request_outputs.items(), key=lambda x: x[0])).values())
            from vllm.outputs import RequestOutput, CompletionOutput

            request_outputs = []
            for prompt_id in unordered_output_token_ids.keys():
                request_outputs.append(
                    RequestOutput(
                        request_id="shouldnt matter",
                        prompt=None,
                        prompt_logprobs=None,
                        finished=True,
                        prompt_token_ids=tokenized_prompts[prompt_id],
                        outputs=[
                            CompletionOutput(
                                index=0,
                                token_ids=unordered_output_token_ids[prompt_id],
                                text=tokenizer.decode(unordered_output_token_ids[prompt_id], skip_special_tokens=True),
                                cumulative_logprob=None,
                                logprobs=None,
                            )
                        ],
                    )
                )
            proofs = list(dict(sorted(unordered_proofs.items(), key=lambda x: x[0])).values())
            assert len(request_outputs) == batch_size, "Number of request outputs must match batch size"
            assert len(proofs) == batch_size, "Number of proofs must match batch size"
        else:
            token_prompts: list[TokensPrompt] = [TokensPrompt(prompt_token_ids=prompt_token_ids) for prompt_token_ids in tokenized_prompts]
            request_outputs = llm.generate(token_prompts, sampling_params, use_tqdm=config.use_tqdm)

            # This generates proofs for the remaining sequences that haven't reached max_len.
            # We call here to give time for the proofs to be generated non-blocking in the background.
            toploc_cache.maybe_generate_proofs_in_background(force_generate=True)

            # Compute proofs
            # Note (Jack): Currently, vllm guarantees that seq ids are in the same order as prompts passed to generate.
            # Generate always adds requests to the engine in the order of the prompts.
            # And returns them in the sequence they were added.
            toploc_cache.wait_for_proofs()
            proofs = [b"".join(proofs) for _, proofs in sorted(toploc_cache.proofs.items(), key=lambda x: x[0])]
            toploc_cache.reset_cache()

        generation_time = time.time() - generate_start_time

        # Compute progress metrics
        batch_problems = len(problems)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.success(f"Generated {batch_samples} samples for {batch_problems} problems in {generation_time:.2f}s")

        # Print example
        first_prompt = tokenizer.decode(request_outputs[0].prompt_token_ids)
        first_completion = tokenizer.decode(request_outputs[0].outputs[0].token_ids)
        logger.debug(f"Showing example (first completion):\n{first_prompt}{first_completion}")

        # Log progress metrics
        progress_metrics = {
            "progress/batch_problems": batch_problems,
            "progress/batch_samples": batch_samples,
            "progress/batch_tokens": batch_tokens,
            "progress/step": step,
            "step": ckpt_step,  # We use the train step to synchronize the wandb axis across runs
        }
        monitor.log(progress_metrics, wandb_prefix="infer")

        # Compute performance metrics
        batch_tokens_per_second = batch_tokens / generation_time
        batch_samples_per_minute = batch_samples / generation_time * 60
        batch_avg_seq_length = batch_tokens / batch_size
        logger.info(
            f"Batch throughput: {batch_tokens_per_second:.2f} tokens/sec, {batch_samples_per_minute:.2f} samples/min ({batch_tokens} tokens in {generation_time:.2f}s, avg seq len: {batch_avg_seq_length:.1f})"
        )

        # Log performance metrics
        perf_metrics = {
            "performance/batch_tokens_per_second": batch_tokens_per_second,
            "performance/batch_samples_per_minute": batch_samples_per_minute,
            "performance/batch_avg_seq_length": batch_avg_seq_length,
            "step": ckpt_step,
        }
        monitor.log(perf_metrics, wandb_prefix="infer")

        # Compute and log rewards and advantages
        logger.info("Computing rewards and advantages")
        request_rewards = compute_vllm_rewards(request_outputs, verification_infos, task_types, config.rewards)
        #batch_reward = sum(sum(r.reward for r in req.rewards) for req in request_rewards) / batch_samples
        # Split regular vs. privileged rewards
        reg_sum = priv_sum = 0.0
        reg_cnt = priv_cnt = 0
        for rr, is_priv in zip(request_rewards, priv_flags):
            for r in rr.rewards:           # each request can have n samples
                if is_priv:
                    priv_sum += r.reward
                    priv_cnt += 1
                else:
                    reg_sum += r.reward
                    reg_cnt += 1
        batch_reward          = reg_sum  / max(reg_cnt,  1)   # legacy metric (regular)
        priv_batch_reward     = priv_sum / max(priv_cnt, 1)

        logger.info(f"Average reward of the batch: {batch_reward:.2f}")
        rewards_metrics = {
            "rewards/batch_reward":       batch_reward,
            "rewards/priv_batch_reward":  priv_batch_reward,
            "step":                       ckpt_step,
        }
        monitor.log(rewards_metrics, wandb_prefix="infer")

        if sampling_params.seed is not None:
            sampling_seeds = [sampling_params.seed + i for i in range(sampling_params.n)] * problems_per_batch
        else:
            sampling_seeds = [None] * batch_samples

        # Get parquet table
        table = get_parquet_table(
            request_outputs,
            request_rewards,
            prompts,
            proofs,
            ckpt_step,
            target_lengths,
            problems,
            enable_logprobs=config.sampling.logprobs is not None,
            seeds=sampling_seeds,
            temperature=sampling_params.temperature,
            ground_truths=ground_truths,
        )
        row_priv_flags = []
        for flag, req in zip(priv_flags, request_outputs):
            row_priv_flags.extend([flag] * len(req.outputs))

        print(len(request_outputs), len(prompts), len(row_priv_flags))
        # Append EM-specific metadata so the trainer can filter quickly
        #table = table.append_column("is_privileged", pa.array(row_priv_flags))
        #table = table.append_column(
        #    "variant",
        #    pa.array([("privileged" if f else "regular") for f in row_priv_flags]),
        #)

        # Save outputs to parquet file
        step_path = Path(config.rollout_path) / f"step_{step}"
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"{uuid.uuid4()}.parquet"
        logger.info(f"Saving batch outputs to {save_path}")
        pq.write_table(table, save_path)

        # Log file metadata
        sha256 = sha256sum(save_path)
        flop_counts = [
            get_inference_input_output_flops(config.model.name, len(input_tokens), len(output_tokens))
            for input_tokens, output_tokens in zip(table.column("input_tokens").to_pylist(), table.column("output_tokens").to_pylist())
        ]

        work_submission = {
            "output/output_flops": sum(output_flops for _, output_flops in flop_counts) // config.parallel.pp.world_size,
            "output/input_flops": sum(input_flops for input_flops, _ in flop_counts) // config.parallel.pp.world_size,
            "output/save_path": save_path.as_posix(),
            "output/sha256": sha256,
            "output/step": step,
        }
        monitor.log(work_submission, exclude=["wandb"])

        step += 1

        if config.max_steps is not None and step > config.max_steps:
            logger.info(f"Reached max steps {config.max_steps}, stopping inference")
            break

        dataset_offset += problems_per_batch

    logger.success(f"Inference finished! Generated {total_samples} samples for {total_problems} problems")


def main(config: InferenceConfig) -> list[mp.Process]:
    processes = []

    if config.parallel.dp > 1:
        if config.parallel.tp == "auto":
            assert torch.cuda.device_count() % config.parallel.dp == 0, "Number of GPUs must be divisible by DP"
            config.parallel.tp = torch.cuda.device_count() // config.parallel.dp
        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.parallel.tp] for i in range(0, len(gpu_ids), config.parallel.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, env), args=(config,))
            processes.append(process)
    else:
        if config.parallel.tp == "auto":
            config.parallel.tp = torch.cuda.device_count()
        inference(config)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")

    config = parse_argv(InferenceConfig)

    if config.rl and config.rl.step_endpoint is not None:
        current_step = requests.get(config.rl.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        assert config.rl is not None, "RL config is required when SHARDCAST_SERVERS is set"
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rl.ckpt_path,
            config.rl.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.rl.async_level, 1),
        )
    else:
        shardcast_process = None

    try:
        main(config)

    finally:
        if shardcast_process is not None:
            import os
            import signal

            # SIGTERM is not working, so we use SIGKILL
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()
