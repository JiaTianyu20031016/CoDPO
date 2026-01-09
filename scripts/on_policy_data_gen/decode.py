from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
import math
import multiprocessing as mp
from contextlib import contextmanager
from pathlib import Path

from safetensors.torch import safe_open, save_file
import torch


def _is_truncated(finish_reason):
    reason = getattr(finish_reason, "name", None)
    if reason is None:
        reason = str(finish_reason)
    return "length" in reason.lower() or "max_tokens" in reason.lower()


def _is_value_head_key(key: str) -> bool:
    return key.startswith("v_head") or key.startswith("model.value_head")


@contextmanager
def strip_value_head_parameters(model_path: str):
    """Temporarily remove value_head parameters from safetensor shards.

    - No-op if the path is not a directory, index file is missing, or no value_head keys exist.
    - Backs up touched shards as *.bak, writes filtered shards to the original names.
    - Restores backups on exit and removes the backup artifacts.
    """

    backups = []
    try:
        model_dir = Path(model_path)
        if not model_dir.exists() or not model_dir.is_dir():
            yield
            return

        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            yield
            return

        try:
            index_data = json.loads(index_path.read_text())
        except Exception as exc:  # keep running even if index parsing fails
            print(f"[strip_value_head_parameters] Failed to read {index_path}: {exc}")
            yield
            return

        weight_map = index_data.get("weight_map", {})
        shards_to_edit = {shard for key, shard in weight_map.items() if _is_value_head_key(key)}
        if not shards_to_edit:
            yield
            return

        for shard_name in shards_to_edit:
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                continue

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                filtered_tensors = {k: f.get_tensor(k) for k in keys if not _is_value_head_key(k)}

            if len(filtered_tensors) == len(keys):
                continue

            backup_path = shard_path.with_suffix(shard_path.suffix + ".bak")
            os.replace(shard_path, backup_path)
            save_file(filtered_tensors, shard_path)
            backups.append((shard_path, backup_path))

        yield
    finally:
        for shard_path, backup_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, shard_path)


def _build_worker(gpu_id, prompts_slice, start_idx, args, return_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    tokenizer = llm.get_tokenizer()

    conversations = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts_slice
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=1 if args.alpaca_golden else args.num_samples,
    )

    outputs = llm.generate(conversations, sampling_params)
    generated_payloads = []
    truncated = 0
    total = 0
    for i, output in enumerate(outputs):
        generated_texts = [o.text for o in output.outputs]
        generated_payloads.append(
            {
                'prompt': prompts_slice[i],
                'format_prompt': output.prompt,
                'generated': generated_texts,
            }
        )
        for o in output.outputs:
            total += 1
            if _is_truncated(o.finish_reason):
                truncated += 1

    return_queue.put((start_idx, generated_payloads, truncated, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode with vllm')
    parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                        help='Directory containing the data')
    parser.add_argument('--data_split', type=str, default="train",
                        help='Data split to use')
    parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                        help='Path to the LLM model')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--max_num_seqs', type=int, default=128,
                        help='Maximum number of in-flight sequences per worker LLM')
    parser.add_argument('--max_num_batched_tokens', type=int, default=32768,
                        help='Maximum batched tokens per worker LLM to avoid OOM')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Number of responses to generate per prompt')
    parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                        help='output_dir')
    parser.add_argument('--alpaca_golden', action='store_true', help='Use alpaca_golden.json instructions and pair with golden outputs')
    args = parser.parse_args()

    print(args)

    # count visible GPUs
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_gpus is not None:
        gpu_ids = [g.strip() for g in visible_gpus.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    if not gpu_ids:
        gpu_ids = ["0"]
    num_gpus = len(gpu_ids)
    print(f"Number of visible GPUs: {num_gpus}")

    data_dir = args.data_dir
    
    if args.alpaca_golden:
        with open('/data2/jty/CoDPO/scripts/eval/alpaca_golden.json', 'r') as f:
            alpaca_data = json.load(f)
        prompts = [item['instruction'] for item in alpaca_data]
        gold_outputs = [item['output'] for item in alpaca_data]
        output_file = f'output_alpaca_{args.seed}.json'
    else:
        if not os.path.exists(data_dir):
            train_dataset = load_dataset(data_dir, split=args.data_split)
        else:
            train_dataset = load_from_disk(data_dir)[args.data_split]

        prompts = [item[0]['content'] for item in train_dataset['chosen']]
        gold_outputs = None
        output_file = f'output_{args.seed}.json'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    world_size = len(gpu_ids)
    per_chunk = max(1, math.ceil(len(prompts) / world_size)) if world_size > 0 else len(prompts)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []

    with strip_value_head_parameters(args.model):
        for rank, gpu_id in enumerate(gpu_ids):
            start_idx = rank * per_chunk
            end_idx = min(len(prompts), (rank + 1) * per_chunk)
            if start_idx >= len(prompts):
                continue
            prompt_slice = prompts[start_idx:end_idx]
            p = ctx.Process(target=_build_worker, args=(gpu_id, prompt_slice, start_idx, args, queue))
            p.start()
            procs.append(p)

        outputs_collected = [None] * len(procs)
        truncated_total = 0
        sample_total = 0

        while len(procs) > 0 and any(item is None for item in outputs_collected):
            start_idx, worker_data, truncated, total = queue.get()
            slot = start_idx // per_chunk
            outputs_collected[slot] = (start_idx, worker_data)
            truncated_total += truncated
            sample_total += total

        for p in procs:
            p.join()

    output_data = []
    for chunk in outputs_collected:
        if chunk is None:
            continue
        start_idx, worker_items = chunk
        for local_offset, item in enumerate(worker_items):
            global_idx = start_idx + local_offset
            if args.alpaca_golden:
                output_data.append(
                    {
                        'prompt': item['prompt'],
                        'format_prompt': item['format_prompt'],
                        'all_generated_responses': [item['generated'][0], gold_outputs[global_idx]],
                        'gt_chosen_idx': 0,
                        'gt_rejected_idx': 1,
                    }
                )
            else:
                output_data.append(
                    {
                        'prompt': item['prompt'],
                        'format_prompt': item['format_prompt'],
                        'all_generated_responses': item['generated'],
                    }
                )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        json.dump(output_data, f, indent=4)

    print(
        f'''
        ##########################################################################
        Outputs saved to {os.path.join(args.output_dir, output_file)}
        ##########################################################################
        '''
    )
    
    # report truncation rate
    truncated = truncated_total
    total = sample_total
    if total > 0:
        trunc_rate = truncated / total
        print(
            f'''
            #######################################################################
            Truncation rate: {trunc_rate:.4f} ({truncated}/{total})
            #######################################################################
            '''
        )