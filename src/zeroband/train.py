import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import shardcast
import torch
import torch.distributed as dist
import torch.distributed.tensor
from jaxtyping import Float
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from torch._guards import log as torch_log
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from zeroband.training import envs
from zeroband.training.checkpoint import TrainingProgress, load_checkpoint_fsdp_state, save_checkpoint_fsdp_state, save_ckpt_for_rollout
from zeroband.training.config import Config as TrainingConfig
from zeroband.training.config import TBConfig
from zeroband.training.data import BatchOutput, DatasetOutput, get_dataloader, packed_batch
from zeroband.training.logger import setup_logger
from zeroband.training.loss import entropy_loss, grpo_loss, kl_penalty, selective_log_softmax
from zeroband.training.utils import (
    MetricsAverager,
    OffloadedTensor,
    PerfCounter,
    apply_ac_ckpt,
    copy_model_to_cpu,
    log_prompt_response_samples,
    offload_model_to_cpu,
    reshard_module,
    wake_up_model_from_cpu,
)
from zeroband.training.world_info import WorldInfo, get_world_info
from zeroband.utils.models import ModelType, get_model_and_tokenizer
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


def get_local_batch_size(batch_size: int, micro_bs: int, data_workers: int, world_info: WorldInfo) -> int:
    assert batch_size % world_info.world_size == 0
    batch_size = batch_size // world_info.world_size

    assert batch_size % micro_bs == 0, str(
        f"The micro batch size ({micro_bs}) must divide the number of samples on each GPU ({batch_size})"
    )

    assert batch_size % data_workers == 0, str(
        f"The batch size ({batch_size}) must be divisible by the number of data workers ({data_workers})."
    )

    return batch_size


def apply_fsdp(model: ModelType, reshard_after_forward: bool):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def get_device_placement(gpus_ids: list[int] | None, world_info: WorldInfo) -> int:
    """handle using a subset of GPUs. Should work like the CUDA_VISIBLE_DEVICES env var.
    The reason we use this is because in the rl launcher, torch is initialized before the env var is set, so we cannot use the CUDA_VISIBLE_DEVICES env var.
    """
    if gpus_ids is None:
        return world_info.local_rank


def get_logprobs(model: ModelType, input_ids: torch.Tensor, position_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    logits: Float[torch.Tensor, "batch seq vocab"] = model(input_ids=input_ids, position_ids=position_ids).logits.contiguous()

    input_ids_shifted = input_ids[:, 1:]
    logits_shifted = logits[:, :-1, :] / temperature
    logprobs = selective_log_softmax(logits_shifted, input_ids_shifted)
    del logits, logits_shifted
    return logprobs


@clean_exit
def train(config: TrainingConfig):
    if "ZERO_BAND_DEV" not in os.environ:
        torch_log.setLevel(logging.CRITICAL)

    world_info = get_world_info()
    logger = setup_logger(config.log, world_info)
    wandb_sample_history = None

    if config.ckpt.clean_rollout_path and config.ckpt.rollout_path is not None:
        logger.info(f"Cleaning rollout path {config.ckpt.rollout_path}")
        shutil.rmtree(config.ckpt.rollout_path, ignore_errors=True)

    logger.info(f"start training on {world_info.world_size} rank(s)")

    # Allow eager fallback during production so that training runs don't die if compile fails
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    torch.cuda.set_device(get_device_placement(config.gpus_ids, world_info))

    local_batch_size = get_local_batch_size(config.optim.batch_size, config.train.micro_bs, config.data.num_workers, world_info)

    if config.ckpt.rollout_path is not None and world_info.rank == 0:
        if envs.SHARDCAST_OUTPUT_DIR is not None:
            shardcast.initialize(envs.SHARDCAST_OUTPUT_DIR, max_distribution_folders=config.max_async_level)

    model, tokenizer = get_model_and_tokenizer(config.model.name, config.train.attn_impl)

    perf_counter = PerfCounter(window_size=min(10, 2 * config.optim.step_per_rollout), model=model, seq_len=config.data.seq_length)

    if config.train.liger_qwen:
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            model=model,
        )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    apply_fsdp(model, config.train.reshard_after_forward)

    if config.grpo.kl_coef is not None or isinstance(config.grpo.off_policy, TBConfig):
        model_reference, _ = get_model_and_tokenizer(config.model.name, config.train.attn_impl)
        apply_fsdp(model_reference, config.train.reshard_after_forward)

    if config.recompute_logprobs:
        model_for_logprob_only, _ = get_model_and_tokenizer(config.model.name, config.train.attn_impl)
        apply_fsdp(model_for_logprob_only, config.train.reshard_after_forward)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.optim.lr,
        weight_decay=config.optim.optim.weight_decay,
        betas=(config.optim.optim.betas1, config.optim.optim.betas2),
    )

    total_samples = config.start_total_samples if config.start_total_samples is not None else 0
    training_progress = TrainingProgress(total_tokens=0, step=config.start_step, total_samples=total_samples)

    # Setup the monitor
    monitor = setup_monitor(config.monitor, run_config=config)

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

        if config.grpo.kl_coef is not None:
            model_reference = torch.compile(model_reference) if not TYPE_CHECKING else model_reference

        if config.recompute_logprobs:
            model_for_logprob_only = torch.compile(model_for_logprob_only) if not TYPE_CHECKING else model_for_logprob_only

    tensor_offloaded_repository: dict[int, OffloadedTensor] = {}

    if config.grpo.kl_coef is not None:
        logger.info(f"memory before model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        tensor_offloaded_repository[0] = offload_model_to_cpu(model_reference)
        logger.info(f"memory after model reference offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if config.recompute_logprobs:
        logger.info(f"memory before model for logprob offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        tensor_offloaded_repository[0] = offload_model_to_cpu(model_for_logprob_only)
        # will be redundant if kl loss is use but probably fine with it
        logger.info(f"memory after model for logprob offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    if config.ckpt.resume:
        logger.info(f"loading checkpoint from {config.ckpt.resume}")
        load_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.resume)

    if training_progress.step % config.optim.step_per_rollout != 0:
        logger.warning(
            f"Resuming training from step {training_progress.step} seems invalid, as it should be multiple of train.step_per_rollout ({config.optim.step_per_rollout})"
            f"training will continue as if it was from step {training_progress.step - training_progress.step % config.optim.step_per_rollout}"
        )

    step_count_init = (
        config.start_rollout_step if config.start_rollout_step is not None else training_progress.step // config.optim.step_per_rollout
    )
    train_dataloader, prefetcher = get_dataloader(
        tokenizer=tokenizer,
        local_batch_size=local_batch_size,
        batch_size=config.optim.batch_size * config.optim.step_per_rollout,
        data_config=config.data,
        step_count_init=step_count_init,
    )
    train_dataloader_iterator = iter(train_dataloader)

    previous_ckpt_rollout = []

    # Number of samples per query for TB log Z acc
    if isinstance(config.grpo.off_policy, TBConfig):
        K = config.grpo.off_policy.n

    logger.info("Starting training loop")
    next_reference_reset_step = config.grpo.reference_reset_interval if config.grpo.reference_reset_interval else float("inf")

    # --- Configuration ---
    tb_beta = config.grpo.off_policy.beta if isinstance(config.grpo.off_policy, TBConfig) else None
    while True:
        if tb_beta is not None and config.grpo.off_policy.beta_decay_end > 0:
            beta_decay_end = config.grpo.off_policy.beta_decay_end
            final_beta = config.grpo.off_policy.beta / 100
            if config.grpo.off_policy.final_beta is not None:
                final_beta = config.grpo.off_policy.final_beta
            decay_schedule = torch.linspace(config.grpo.off_policy.beta, final_beta, beta_decay_end)
            tb_beta = decay_schedule[min(training_progress.step, beta_decay_end - 1)].item()

        time_start = time.time()

        total_time_data_loading = 0
        total_time_packing = 0
        if isinstance(config.grpo.off_policy, TBConfig):
            logZ_list = []
            kl_avg_list = []
            kl_abs_avg_list = []
            kl_noClamp_abs_avg_list = []
            kl_avg_var_list = []
            kl_max_list = []

        # here we want to pre-compute the logprobs with the model before update
        with torch.no_grad():
            if config.grpo.kl_coef is not None:
                wake_up_model_from_cpu(model_reference, tensor_offloaded_repository[0])
                # del tensor_offloaded_repository[0]

            if config.recompute_logprobs:
                og_infer_step = training_progress.step // config.optim.step_per_rollout - config.max_async_level
                infer_step = max(og_infer_step, 0)
                wake_up_model_from_cpu(model_for_logprob_only, tensor_offloaded_repository[infer_step])

                if og_infer_step == infer_step:
                    del tensor_offloaded_repository[infer_step]

            data: list[list[BatchOutput]] = []

            for rollout_step in range(config.optim.step_per_rollout):
                logger.debug(f"start rollout step {rollout_step} / {config.optim.step_per_rollout}")
                time_data_loading = time.time()

                batch_rollout: list[DatasetOutput] = next(train_dataloader_iterator)
                time_data_loading = time.time() - time_data_loading
                total_time_data_loading += time_data_loading

                time_0 = time.time()

                batch_packed = packed_batch(
                    batch_rollout, config.data.seq_length, tokenizer.pad_token_id, config.train.micro_bs, config.collate_mode
                )
                num_grad_acc_steps = len(batch_packed)

                if isinstance(config.grpo.off_policy, TBConfig):
                    num_steps_per_logZ = K // (config.train.micro_bs * world_info.world_size)
                    print(f"micro_bs is {config.train.micro_bs} -- i.e., # elements in each grad_acc_step per device")
                    print(f"steps per logZ is {num_steps_per_logZ} -- i.e., # grad_acc_step per query")
                    assert num_steps_per_logZ >= 1, f"{num_steps_per_logZ} is not >= 1, change hparams or logZ logic"
                    assert K % (config.train.micro_bs * world_info.world_size) == 0, (
                        f"K is not an integer multiple of micro_bs * world_size {config.train.micro_bs, world_info.world_size}, change hparams or logZ logic"
                    )
                    logZ = 0.0
                    kl_metric = 0.0
                    kl_var = 0.0
                    kl_max = 0.0
                    kl_abs_metric = 0
                    kl_noClamp_abs_metric = 0

                    problem_ids_in_window = []

                time_1 = time.time()
                total_time_packing += time_1 - time_0

                for grad_acc_step in range(num_grad_acc_steps):
                    batch = batch_packed[grad_acc_step]
                    assert len(batch["input_ids"]) == config.train.micro_bs, (
                        f"expected {config.train.micro_bs} elements in grad_acc_step, got {len(batch['input_ids'])}"
                    )

                    # Only compute logprobs if not using vllm logprobs or if the batch doesn't have them
                    if config.recompute_logprobs or isinstance(config.grpo.off_policy, TBConfig):
                        logger.debug(f"log prob grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")
                        input_ids = batch["input_ids"].to("cuda")
                        # Ensure input_ids are long type
                        if input_ids.dtype != torch.long:
                            logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                            input_ids = input_ids.long()

                        model_for_logprob = model_for_logprob_only if config.recompute_logprobs else model
                        per_token_logps = get_logprobs(model_for_logprob, input_ids, batch["position_ids"], batch["temperature"])

                        batch["logprobs"] = per_token_logps.to("cpu")

                        # For TBA we center the per-sample kl_est (computed against the
                        # live training policy inside tba_loss) using a kl_est mean that
                        # must come from the same train policy. batch["logprobs"] is the
                        # inference-snapshot recompute used by the IS ratio, so we
                        # populate a separate batch["train_logprobs"] from `model`.
                        if isinstance(config.grpo.off_policy, TBConfig):
                            if model_for_logprob is model:
                                batch["train_logprobs"] = batch["logprobs"]
                            else:
                                per_token_logps_train = get_logprobs(model, input_ids, batch["position_ids"], batch["temperature"])
                                batch["train_logprobs"] = per_token_logps_train.to("cpu")

                    if config.grpo.kl_coef is not None or isinstance(config.grpo.off_policy, TBConfig):
                        logger.debug(f"kl grad_acc_step {grad_acc_step} / {num_grad_acc_steps}, batch: {batch['input_ids'].shape}")
                        input_ids = batch["input_ids"].to("cuda")
                        # Ensure input_ids are long type
                        if input_ids.dtype != torch.long:
                            logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                            input_ids = input_ids.long()
                        per_token_logps_reference = get_logprobs(model_reference, input_ids, batch["position_ids"], batch["temperature"])
                        batch["ref_logprobs"] = per_token_logps_reference.to("cpu")

                    if isinstance(config.grpo.off_policy, TBConfig):
                        problem_ids_in_window.extend(batch["problem_ids"])

                        masked_per_token_logps = batch["loss_mask"][:, 1:] * batch["train_logprobs"]
                        masked_ref_logprobs = batch["loss_mask"][:, 1:] * batch["ref_logprobs"]
                        kl_est = masked_per_token_logps - masked_ref_logprobs
                        kl_noClamp_abs_metric = kl_noClamp_abs_metric + kl_est.detach().abs().sum(1).sum()
                        kl_est = torch.clamp(kl_est, min=-10, max=10)
                        kl_abs_metric = kl_abs_metric + kl_est.detach().abs().sum(1).sum()
                        kl_est = kl_est.sum(1)

                        kl_metric = kl_metric + kl_est.detach().sum()
                        kl_var = kl_var + (kl_est.detach() ** 2).sum()
                        kl_max = torch.maximum(torch.tensor(kl_max), torch.max(kl_est))

                        logZ = (
                            logZ + batch["rewards"].sum()
                            # + tb_beta * kl_est   # just average the rewards for now
                        )
                        if (grad_acc_step + 1) % num_steps_per_logZ == 0:
                            # All-gather logZ values from all GPUs
                            logZ = logZ.cuda()
                            all_logZ = [torch.zeros_like(logZ) for _ in range(world_info.world_size)]
                            dist.all_gather(all_logZ, logZ)

                            kl_metric = kl_metric.cuda()
                            all_kl_metric = [torch.zeros_like(kl_metric) for _ in range(world_info.world_size)]
                            dist.all_gather(all_kl_metric, kl_metric)
                            kl_avg_list.append(torch.tensor(all_kl_metric).sum().cpu() / K)
                            kl_metric = 0

                            kl_var = kl_var.cuda()
                            all_kl_metric = [torch.zeros_like(kl_var) for _ in range(world_info.world_size)]
                            dist.all_gather(all_kl_metric, kl_var)
                            kl_avg_var_list.append(torch.tensor(all_kl_metric).sum().cpu() / K - kl_avg_list[-1] ** 2)
                            kl_var = 0

                            kl_abs_metric = kl_abs_metric.cuda()
                            all_kl_metric = [torch.zeros_like(kl_abs_metric) for _ in range(world_info.world_size)]
                            dist.all_gather(all_kl_metric, kl_abs_metric)
                            kl_abs_avg_list.append(torch.tensor(all_kl_metric).sum().cpu() / K)
                            kl_abs_metric = 0
                            kl_noClamp_abs_metric = kl_noClamp_abs_metric.cuda()
                            all_kl_metric = [torch.zeros_like(kl_noClamp_abs_metric) for _ in range(world_info.world_size)]
                            dist.all_gather(all_kl_metric, kl_noClamp_abs_metric)
                            kl_noClamp_abs_avg_list.append(torch.tensor(all_kl_metric).sum().cpu() / K)
                            kl_noClamp_abs_metric = 0

                            kl_max = kl_max.cuda()
                            all_kl_max = [torch.zeros_like(kl_max) for _ in range(world_info.world_size)]
                            dist.all_gather(all_kl_max, kl_max)
                            kl_max_list.append(torch.max(torch.tensor(all_kl_max)).cpu())
                            kl_max = 0

                            # this is a check to make sure we have the same problem ID for all contributions to logZ
                            local_problem_id = torch.tensor([int(x.split("_")[-1]) for x in problem_ids_in_window]).cuda()
                            all_problem_ids = [torch.zeros_like(local_problem_id) for _ in range(world_info.world_size)]
                            dist.all_gather(all_problem_ids, local_problem_id)
                            all_problem_ids = sum([pids.cpu().tolist() for pids in all_problem_ids], [])  # concat list of lists
                            # Validate that we have exactly K samples from exactly one problem_id
                            counts = Counter(all_problem_ids)
                            unique_problems = list(counts.keys())
                            counts = list(counts.values())
                            assert len(unique_problems) == 1, (
                                f"GPU {world_info.rank}: Expected 1 unique id across GPUs, my problems: {local_problem_id}"
                            )
                            assert counts[0] == K, (
                                f"GPU {world_info.rank}: Expected {K} samples for problem_id {unique_problems}, got {counts}"
                            )

                            logZ_list.append(torch.tensor(all_logZ).sum().cpu() / K)
                            # print(f"GPU {world_info.rank}: Got logZ {all_logZ} with sum {logZ_list[-1]}",flush=True)
                            logZ = 0.0
                            problem_ids_in_window = []

                data.append(batch_packed)

            if config.grpo.kl_coef is not None:
                # if we don't manually reshard the the embed and lm head will conflict with the offloading because they will stay unshard until backward which we never call
                reshard_module(model_reference)
                tensor_offloaded_repository[0] = offload_model_to_cpu(model_reference)

            if config.recompute_logprobs:
                # here we sepcifically don't save the tensor offloaded, they are alreay consumed and we will never use it again.
                # this avoid having to make sure we don't keep too much tensor offloaded in cpu memory
                reshard_module(model_for_logprob_only)
                offload_model_to_cpu(model_for_logprob_only)

            avg_logZ_step = None
            if isinstance(config.grpo.off_policy, TBConfig):
                avg_logZ_step = torch.tensor(logZ_list).mean()
                avg_kl_step = torch.tensor(kl_avg_list).mean()
                avg_kl_var = torch.tensor(kl_avg_var_list).mean()
                avg_kl_abs_step = torch.tensor(kl_abs_avg_list).mean()
                avg_kl_abs_noClamp_step = torch.tensor(kl_noClamp_abs_avg_list).mean()
                max_kl_step = torch.tensor(kl_max_list).max()

            logprobs_aware_iterator = iter(data)

            total_time = time.time() - time_start
            total_time_logprob = total_time - total_time_data_loading - total_time_packing

            logger.debug(f"Time to data loading: {total_time_data_loading:.2f} seconds")
            logger.debug(f"Time to packing: {total_time_packing:.2f} seconds")
            logger.info(f"Time to compute logprobs: {total_time_logprob:.2f} seconds")
            logger.info(f"Total time data preprocessing: {total_time:.2f} seconds")

        logger.debug("start training rollout")

        # In the training loop
        for rollout_step in range(config.optim.step_per_rollout):
            logger.debug(f"training rollout step {rollout_step} / {config.optim.step_per_rollout}")
            metric_averager = MetricsAverager()
            loss_batch = torch.tensor(0.0, device="cuda")

            if config.train.memory_profile and world_info.rank == 0:
                torch.cuda.memory._record_memory_history()

            data_per_rollout = next(logprobs_aware_iterator)
            num_grad_acc_steps = len(data_per_rollout)

            # Collect samples for WandB logging - do this ONCE per step
            if config.monitor.wandb is not None and config.monitor.wandb.log_samples and world_info.rank == 0:
                # Use the first batch for logging (could be configurable if needed)
                batch = data_per_rollout[0]

                # Log the samples to WandB with history management
                try:
                    # Pass and update the sample history
                    wandb_sample_history = log_prompt_response_samples(tokenizer, batch, training_progress.step, wandb_sample_history)
                except Exception as e:
                    logger.warning(f"Error logging samples to WandB: {e}")

            # Now here's the complete grad_acc_step loop WITHOUT the WandB logging inside it:
            for grad_acc_step in range(num_grad_acc_steps):
                # logZ_batch = 0.0
                logZ_batch_reward_only = 0.0
                effective_tb_beta = None
                kl_term = 0
                if isinstance(config.grpo.off_policy, TBConfig):
                    effective_tb_beta = tb_beta
                    kl_term = kl_avg_list[grad_acc_step // num_steps_per_logZ]
                    # logZ_batch = (logZ_list[grad_acc_step // num_steps_per_logZ] - effective_tb_beta * kl_term).to("cuda")
                    logZ_batch_reward_only = (logZ_list[grad_acc_step // num_steps_per_logZ]).to("cuda")
                logger.debug(f"training grad_acc_step {grad_acc_step} / {num_grad_acc_steps}")
                batch = data_per_rollout[grad_acc_step]

                input_ids = batch["input_ids"].to("cuda")
                # Ensure input_ids are long type
                if input_ids.dtype != torch.long:
                    logger.warning(f"Converting input_ids from {input_ids.dtype} to torch.long")
                    input_ids = input_ids.long()
                if config.normalize_batch_to_token_count:
                    max_tokens = int(sum(batch["seq_lens"]))
                else:
                    max_tokens = input_ids.shape[0] * input_ids.shape[1]

                loss_mask = batch["loss_mask"]

                # Update general metrics
                for rewards in batch["rewards"]:
                    metric_averager.update("rewards/sample_reward", rewards)
                for seq_lens in batch["seq_lens"]:
                    metric_averager.update("lengths/seq_lens", seq_lens)
                for length_penalties in batch["length_penalties"]:
                    metric_averager.update("lengths/length_penalties", length_penalties)
                for target_lengths in batch["target_lengths"]:
                    metric_averager.update("lengths/target_lengths", target_lengths)

                # Task-specific metrics with proper grouping
                if "task_types" in batch:
                    # Group rewards by task type
                    task_type_rewards = defaultdict(list)
                    for i, task_type in enumerate(batch["task_types"]):
                        task_type_rewards[task_type].append(batch["task_rewards"][i].item())

                    # Update metrics with task-specific averages
                    for task_key, rewards in task_type_rewards.items():
                        if rewards:
                            avg_reward = sum(rewards) / len(rewards)
                            metric_averager.update(f"task_rewards/{task_key}", torch.tensor(avg_reward))

                    # Add aggregate task_reward metric
                    all_task_rewards = [reward.item() for reward in batch["task_rewards"]]
                    if all_task_rewards:
                        avg_task_reward = sum(all_task_rewards) / len(all_task_rewards)
                        metric_averager.update("rewards/task_reward", torch.tensor(avg_task_reward))

                # Forward
                logits: Float[torch.Tensor, "batch seq vocab"] = model(
                    input_ids=input_ids, position_ids=batch["position_ids"]
                ).logits.contiguous()

                # Gather args for grpo loss
                advantages = batch["advantages"].to("cuda")
                loss_mask = loss_mask.to("cuda")
                original_logprobs = batch["logprobs"].to("cuda")
                ref_logp = original_logprobs
                if isinstance(config.grpo.off_policy, TBConfig):
                    ref_logp = batch["ref_logprobs"].to("cuda")

                # Loss

                pg_loss, clip_ratio = grpo_loss(
                    logits,
                    input_ids,
                    batch["rewards"].to("cuda"),
                    advantages,
                    # logZ_batch,
                    logZ_batch_reward_only,
                    original_logprobs,
                    ref_logp,
                    loss_mask,
                    batch["temperature"],
                    max_tokens,
                    config.grpo.off_policy,
                    # tb_beta = tb_beta # adaptive_beta_test
                    tb_beta=effective_tb_beta,
                    mean_KL=kl_term,
                )

                with torch.no_grad() if config.grpo.entropy_loss_coeff == 0 else nullcontext():
                    entropy = entropy_loss(logits, loss_mask, batch["temperature"], max_tokens)

                loss = pg_loss - config.grpo.entropy_loss_coeff * entropy

                if config.grpo.kl_coef is not None:
                    kl = kl_penalty(original_logprobs, batch["ref_logprobs"].to("cuda"), loss_mask, max_tokens)
                    kl_scaled = kl * config.grpo.kl_coef
                    metric_averager.update("losses/kl", kl_scaled)
                    loss = loss + kl_scaled

                loss = loss / num_grad_acc_steps

                inputs_ids_shape = input_ids.shape

                # Now we can delete the batch data
                del batch, logits, input_ids, advantages, loss_mask, original_logprobs

                # Backward
                loss.backward()
                loss_batch += loss.detach().clone()

                metric_averager.update("losses/pg_loss", pg_loss.detach().clone())
                metric_averager.update("losses/entropy_loss", entropy.detach().clone())

                if clip_ratio is not None:
                    metric_averager.update("losses/clip_ratio", clip_ratio.detach().clone())

                del loss, pg_loss, entropy, clip_ratio

            metric_averager.sync()

            dist.all_reduce(loss_batch, op=dist.ReduceOp.AVG)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_norm_clip).full_tensor()  # type: ignore (is a dtensor)

            logger.debug(f"loss: {loss_batch.item()}, grad_norm: {grad_norm.item()}")

            # if isinstance(config.grpo.off_policy, TBConfig):
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            logger.debug("optimizer step")

            training_progress.step += 1

            # ProRL-style reference model reset
            if (
                config.grpo.reference_reset_interval is not None
                and
                # (config.grpo.kl_coef is not None or ...)
                isinstance(config.grpo.off_policy, TBConfig)
                and training_progress.step >= next_reference_reset_step
            ):
                # TB assumes the reference isn't offloaded, unlike GRPO/kl_coef
                # we could change the code later to work for both
                assert config.grpo.kl_coef is None, "resetting only works for TB right now"

                logger.info(f"Resetting reference model at step {training_progress.step}")

                # If reference model is offloaded, wake it up first
                # wake_up_model_from_cpu(model_reference, tensor_offloaded_repository[0])

                # Copy weights from training model to reference model
                copy_model_weights_fsdp(model, model_reference)

                # Re-offload reference model if it was offloaded
                # reshard_module(model_reference)
                # tensor_offloaded_repository[0] = offload_model_to_cpu(model_reference)

                # Update next reset step
                next_reference_reset_step += config.grpo.reference_reset_interval

                # Log the reset event
                if world_info.rank == 0:
                    monitor.log(
                        {
                            "step": training_progress.step,
                            "reference_reset": 1,  # Binary indicator for visualization
                        }
                    )

                if config.grpo.reference_reset_opt is not None:
                    optimizer = torch.optim.AdamW(
                        params=model.parameters(),
                        lr=config.optim.optim.lr,
                        weight_decay=config.optim.optim.weight_decay,
                        betas=(config.optim.optim.betas1, config.optim.optim.betas2),
                    )

            inner_lr = [group["lr"] for group in optimizer.param_groups][0]

            token_per_gpu = inputs_ids_shape[0] * inputs_ids_shape[1] * num_grad_acc_steps
            new_tokens = world_info.world_size * token_per_gpu
            perf_counter.count_tokens(new_tokens)
            training_progress.total_tokens += new_tokens
            training_progress.total_samples += config.optim.batch_size

            padding_proportion = (config.data.seq_length - metric_averager["lengths/seq_lens"].item() - 1) / config.data.seq_length

            metrics = {
                "step": training_progress.step,
                "losses/Loss": loss_batch.item(),
                "train/rollout_step": rollout_step,
                "train/inner_lr": inner_lr,
                "train/total_tokens": training_progress.total_tokens,
                "train/total_samples": training_progress.total_samples,
                "losses/grad_norm": grad_norm.item(),
                "lengths/padding_proportion": padding_proportion,
            }

            for key, value in metric_averager.items():
                metrics[key] = value.item()

            log = (
                f"step: {training_progress.step}, "
                f"rollout_step: {training_progress.step // config.optim.step_per_rollout}, "
                f"loss: {loss_batch.item():.4f}, "
                f"sample_reward: {metric_averager['rewards/sample_reward'].item():.4f}, "
            )

            del loss_batch, grad_norm

            tokens_per_second = perf_counter.get_tokens_per_second()
            if tokens_per_second is not None:
                tokens_per_second_per_gpu = tokens_per_second / world_info.world_size
                mfu = perf_counter.get_mfu()
                metrics.update(
                    {
                        "perf/tokens_per_second": tokens_per_second,
                        "perf/tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                        "perf/mfu": mfu,
                    }
                )

                log += f", tokens_per_second: {tokens_per_second:.2f}, tokens_per_second_per_gpu: {tokens_per_second_per_gpu:.2f}, mfu: {mfu:.2f}"

            if avg_logZ_step is not None:
                metrics["rewards/avg_logZ"] = avg_logZ_step.item()
                metrics["train/tb_beta"] = tb_beta
                metrics["rewards/avg_KL"] = avg_kl_step.item()
                metrics["rewards/avg_KL_var"] = avg_kl_var.item()
                metrics["rewards/max_KL"] = max_kl_step.item()
                metrics["rewards/avg_kl_abs_step"] = avg_kl_abs_step.item()
                metrics["rewards/avg_kl_abs_noClamp_step"] = avg_kl_abs_noClamp_step.item()
                log += f", avg_logZ: {avg_logZ_step.item():.4f}"

            if world_info.rank == 0:
                monitor.log(metrics)

            logger.info(log)

            time_rollout_ckpt = None
            time_shardcast = None
            time_rollout_delete = None

            # Lets do this first so that clients can start downloading as soon as possible
            if config.ckpt.rollout_path is not None and training_progress.step % config.optim.step_per_rollout == 0:
                logger.debug("saving rollout ckpt")
                rollout_step = training_progress.step // config.optim.step_per_rollout
                path = Path(config.ckpt.rollout_path) / f"step_{rollout_step}"
                previous_ckpt_rollout.append(path)
                t0 = time.time()
                safetensor_path = save_ckpt_for_rollout(model, tokenizer, path, async_save=config.ckpt.async_save)
                if training_progress.step in [200, 250, 300, 350, 400, 450, 500]:
                    final_path = str(path).replace("tba-prime", "tba-prime/final")
                    final_path = final_path.replace("_checkpoints", "")
                    _ = save_ckpt_for_rollout(model, tokenizer, Path(final_path), async_save=config.ckpt.async_save)
                time_rollout_ckpt = time.time() - t0

                time_shardcast = time.time()
                if world_info.rank == 0:
                    if envs.SHARDCAST_OUTPUT_DIR is not None:
                        logger.info(f"Broadcasting {safetensor_path}")
                        shardcast.broadcast(safetensor_path)  # TODO: Is this blocking?
                time_shardcast = time.time() - time_shardcast

                time_rollout_delete = time.time()
                if len(previous_ckpt_rollout) > config.max_async_level + 1:
                    path_to_delete = previous_ckpt_rollout.pop(0)
                    ckpt_step = int(str(path_to_delete).split("_")[-1])

                    should_keep = config.ckpt.interval_rollout is not None and ckpt_step % config.ckpt.interval_rollout == 0
                    if path_to_delete.exists() and not should_keep:
                        logger.info(f"Removing past rollout ckpt at {path_to_delete}")
                        shutil.rmtree(path_to_delete, ignore_errors=True)
                time_rollout_delete = time.time() - time_rollout_delete
            if config.train.memory_profile and (training_progress.step == 2) and world_info.rank == 0:
                logger.info("Dumping memory snapshot.")
                pickle_path: str = config.train.memory_profile
                if not pickle_path.endswith(".pickle"):
                    pickle_path += ".pickle"
                torch.cuda.memory._dump_snapshot(pickle_path)
                torch.cuda.memory._record_memory_history(enabled=False)

            if config.ckpt.interval is not None and training_progress.step % config.ckpt.interval == 0:
                logger.info(
                    f"Saving checkpoint at step {training_progress.step}, rollout_step {training_progress.step // config.optim.step_per_rollout}"
                )
                save_checkpoint_fsdp_state(model, [optimizer], training_progress, config.ckpt.path)

        if config.recompute_logprobs:
            reshard_module(model_for_logprob_only)
            tensor_offloaded_repository[training_progress.step // config.optim.step_per_rollout] = copy_model_to_cpu(model)

        time_rollout_step = time.time() - time_start
        logger.success(f"Finished training step {training_progress.step} in {time_rollout_step:.2f}s")
        if world_info.rank == 0:
            time_metrics = {
                "step": training_progress.step,
                "perf/time_rollout_step": time_rollout_step,
                "perf/time_logprob": total_time_logprob,
                "perf/time_data_loading": total_time_data_loading,
                "perf/time_packing": total_time_packing,
                "perf/time_data_preprocessing": total_time,
                "perf/time_rollout_delete": time_rollout_delete,
            }
            if time_rollout_ckpt is not None:
                time_metrics["perf/time_rollout_ckpt"] = time_rollout_ckpt
            if time_shardcast is not None:
                time_metrics["perf/time_shardcast"] = time_shardcast
            if time_rollout_delete is not None:
                time_metrics["perf/time_rollout_delete"] = time_rollout_delete

            monitor.log(time_metrics)

        if config.stop_after_steps is not None and training_progress.step >= config.stop_after_steps:
            break

    if prefetcher is not None:
        prefetcher.shutdown()

    logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.success("Training finished!")


def copy_model_weights_fsdp(source_model, target_model):
    """
    Copy weights from source model to target model, handling FSDP state.
    Both models should have the same architecture and FSDP wrapping.
    """
    with torch.no_grad():
        # Get state dicts
        source_state = source_model.state_dict()

        # Load into target model
        target_model.load_state_dict(source_state)


if __name__ == "__main__":
    config = parse_argv(TrainingConfig)
    train(config)
