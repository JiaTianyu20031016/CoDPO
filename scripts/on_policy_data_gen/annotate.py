
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import json
import os
import argparse
import numpy as np
import datasets
from datasets import load_dataset, load_from_disk
import math
import multiprocessing as mp
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# local import; avoid circular deps when not needed
from trl.experimental.ppo.modeling_value_head import AutoModelForCausalLMWithValueHead


def build_inputs(prompt: str | list[str], candidate: str | list[str], tokenizer):
    # 支持 batch：可传单个字符串或字符串列表
    if isinstance(prompt, str):
        prompts = [prompt]
        candidates = [candidate]
    else:
        prompts = list(prompt)
        candidates = list(candidate)

    prompt_chats = [[{"role": "user", "content": p}] for p in prompts]
    full_chats = [
        [
            {"role": "user", "content": p},
            {"role": "assistant", "content": c},
        ]
        for p, c in zip(prompts, candidates)
    ]

    prompt_only = tokenizer.apply_chat_template(
        prompt_chats,
        return_tensors="pt",
        return_attention_mask=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
    )
    full = tokenizer.apply_chat_template(
        full_chats,
        return_tensors="pt",
        return_attention_mask=True,
        return_dict=True,
        padding=True,
    )

    # response_mask 标记 assistant 段 token 位置（考虑 left padding）
    response_mask = torch.zeros_like(full.attention_mask, dtype=torch.bool)
    full_lens = full.attention_mask.sum(dim=1)
    prompt_lens = prompt_only.attention_mask.sum(dim=1)
    seq_len = full.attention_mask.shape[1]
    left_pad = seq_len - full_lens
    starts = (left_pad + prompt_lens).to(torch.long)
    ends = (left_pad + full_lens).to(torch.long)
    for i in range(len(prompts)):
        response_mask[i, starts[i].item(): ends[i].item()] = True

    return full.input_ids, full.attention_mask, response_mask


def sequence_logprobs(model, input_ids, attention_mask, response_mask: torch.Tensor):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_logps = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:]
    if attention_mask is not None:
        mask = mask & (attention_mask[:, 1:] > 0)
    masked = token_logps.masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=1).clamp(min=1)
    logp_sum = masked.sum(dim=1)
    return logp_sum, lengths


def dpo_reward(
        policy_model, 
        ref_model, 
        input_ids, 
        attention_mask, 
        response_mask: torch.Tensor, 
        beta: float,
        avg: bool = True
    ):
    policy_logp, lengths = sequence_logprobs(policy_model, input_ids, attention_mask, response_mask)
    ref_logp, _ = sequence_logprobs(ref_model, input_ids, attention_mask, response_mask)
    if avg:
        return (policy_logp - ref_logp) / lengths * beta
    return (policy_logp - ref_logp) * beta


def value_reward(value_model, input_ids, attention_mask, response_mask: torch.Tensor):
    # value model must be of type AutoModelForCausalLMWithValueHead
    with torch.no_grad():
        base_out, values = value_model(input_ids=input_ids, attention_mask=attention_mask)
    if values.dim() == 3 and values.size(-1) == 1:
        values = values.squeeze(-1)

    mask = response_mask
    if attention_mask is not None:
        mask = mask & (attention_mask > 0)

    has_token = mask.any(dim=1)
    arange_idx = torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
    last_indices = (mask * arange_idx).max(dim=1).values
    batch_idx = torch.arange(values.size(0), device=values.device)
    last_values = values[batch_idx, last_indices]
    last_values = torch.where(has_token, last_values, torch.zeros_like(last_values))

    return last_values.detach()


def worker_process(start_idx, data_slice, args, gpu_id, queue):
    if isinstance(gpu_id, str) and gpu_id == "cpu":
        device = "cpu"
    else:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
    tokenizer.padding_side = "left"

    if args.reward_mode == "golden":
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model, device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        ref_model = None
    elif args.reward_mode == "dpo":
        reward_model = AutoModelForCausalLM.from_pretrained(
            args.reward_model, device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        ref_path = args.ref_model if args.ref_model is not None else args.reward_model
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    elif args.reward_mode in {"value", "vote"}:
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.reward_model, device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        if args.reward_mode == "vote":
            ref_path = args.ref_model if args.ref_model is not None else args.reward_model
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path, device_map=None, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(device)
        else:
            ref_model = None
    else:
        raise ValueError(f"Unsupported reward_mode: {args.reward_mode}")

    local_scores = [[0.0] * len(d["all_generated_responses"]) for d in data_slice]
    pairs = []
    for local_idx, data in enumerate(data_slice):
        responses = data["all_generated_responses"]
        for r_idx, resp in enumerate(responses):
            pairs.append((local_idx, r_idx, data["prompt"], resp))

    for start in tqdm(range(0, len(pairs), args.batch_size)):
        chunk = pairs[start:start + args.batch_size]
        batch_prompts = [c[2] for c in chunk]
        batch_responses = [c[3] for c in chunk]

        input_ids, attention_mask, response_mask = build_inputs(batch_prompts, batch_responses, tokenizer)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        response_mask = response_mask.to(device)

        if args.reward_mode == "golden":
            with torch.no_grad():
                output = reward_model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(output, "score"):
                    batch_scores = output.score.flatten().float().cpu().tolist()
                else:
                    batch_scores = output.logits.squeeze(-1).float().cpu().tolist()
        elif args.reward_mode == "dpo":
            reward = dpo_reward(reward_model, ref_model, input_ids, attention_mask, response_mask, args.beta)
            batch_scores = reward.float().cpu().tolist()
        elif args.reward_mode == "value":
            reward = value_reward(reward_model, input_ids, attention_mask, response_mask)
            batch_scores = reward.float().cpu().tolist()
        elif args.reward_mode == "vote":
            implicit_reward = dpo_reward(reward_model, ref_model, input_ids, attention_mask, response_mask, args.beta)
            value_r = value_reward(reward_model, input_ids, attention_mask, response_mask)
            # batch_scores = ((implicit_reward + value_r) / 2).float().cpu().tolist()
            batch_scores = implicit_reward.float().cpu().tolist()
        else:
            raise ValueError(f"Unsupported reward_mode: {args.reward_mode}")

        for (local_idx, r_idx, _, _), score in zip(chunk, batch_scores):
            local_scores[local_idx][r_idx] = score

    queue.put((start_idx, local_scores))


if __name__ == "__main__":

    # arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file", type=str, default="datasets/gemma2_ultrafeedback/all_outputs.json", help="Path to the output generation file")
    parser.add_argument("--reward_model", type=str, default="/data2/jty/models/ArmoRM", help="Path to reward model")
    parser.add_argument("--ref_model", type=str, default=None, help="Path to reference model for DPO implicit reward; defaults to reward_model when omitted")
    parser.add_argument("--reward_mode", type=str, default="golden", choices=["golden", "dpo", "value", "vote"], help="Reward calculation mode")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta scaling used for DPO implicit reward")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name when generation_file is a saved dataset (e.g., train)")
    parser.add_argument("--save_dataset", action="store_true", help="Whether to save the binarized data as a Hugging Face dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process (for debugging)")
    args = parser.parse_args()
    print(args)

    # read input data
    generation_file = args.generation_file
    if args.split and not generation_file.endswith('.json'):
        if not os.path.exists(generation_file):
            dataset= load_dataset(generation_file)
        else:
            dataset = load_from_disk(generation_file)

        if args.split not in dataset:
            raise ValueError(f"Split {args.split} not found in dataset at {generation_file}")
        split_ds = dataset[args.split]
        output_data = []
        for row in split_ds:
            chosen_turns = row["chosen"]
            rejected_turns = row["rejected"]
            prompt = chosen_turns[0]["content"]
            chosen_resp = chosen_turns[1]["content"]
            rejected_resp = rejected_turns[1]["content"]
            output_data.append({
                "prompt": prompt,
                "all_generated_responses": [chosen_resp, rejected_resp],
                "gt_chosen_idx": 0,
                "gt_rejected_idx": 1,
            })
    else:
        with open(generation_file, 'r') as f:
            output_data = json.load(f)
    
    if args.limit is not None and args.limit > 0:
        output_data = output_data[:args.limit]

    # Multiprocessing data-parallel scoring across available GPUs
    if torch.cuda.is_available():
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_gpus is not None:
            env_list = [x for x in visible_gpus.split(",") if x.strip()]
            gpu_ids = list(range(len(env_list)))  # remap to local indices 0..n-1
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = ["cpu"]

    if len(gpu_ids) == 0:
        gpu_ids = ["cpu"]

    world_size = len(gpu_ids)
    per_chunk = math.ceil(len(output_data) / world_size)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []

    for rank, gpu_id in enumerate(gpu_ids):
        start_idx = rank * per_chunk
        end_idx = min(len(output_data), (rank + 1) * per_chunk)
        if start_idx >= len(output_data):
            continue
        data_slice = output_data[start_idx:end_idx]
        p = ctx.Process(target=worker_process, args=(start_idx, data_slice, args, gpu_id, queue))
        p.start()
        procs.append(p)

    results_received = 0

    while results_received < len(procs):
        start_idx, local_scores = queue.get()
        for offset, scores in enumerate(local_scores):
            output_data[start_idx + offset]["all_rm_scores"] = scores
        results_received += 1

    for p in procs:
        p.join()

    # dump raw RM scores
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        rm_file_name = os.path.basename(args.generation_file).split('.json')[0] + "_rm.json"
        with open(os.path.join(args.output_dir, rm_file_name), 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Annotated outputs saved to {os.path.join(args.output_dir, rm_file_name)}")

    ######### binarize data #########
    # win = highest scoring reponse; lose = lowest scoring response
    correct = 0
    total = 0
    for data in output_data:
        chosen_idx = np.argmax(data["all_rm_scores"])
        rejected_idx = np.argmin(data["all_rm_scores"])

        if "gt_chosen_idx" in data:
            total += 1
            if data["all_rm_scores"][chosen_idx] == data["all_rm_scores"][data["gt_chosen_idx"]]:
                correct += 1

        chosen = []
        chosen.append({
            "role": "user",
            "content": data["prompt"]
        })
        chosen.append({
            "role": "assistant",
            "content": data["all_generated_responses"][chosen_idx]
        })
        rejected = []
        rejected.append({
            "role": "user",
            "content": data["prompt"]
        })
        rejected.append({
            "role": "assistant",
            "content": data["all_generated_responses"][rejected_idx]
        })
        data.update({
            "chosen": chosen,
            "rejected": rejected,
        })

    if total > 0:
        accuracy = correct / total
        print(
            f'''
            ##########################################################################
            Annotation accuracy vs. dataset labels: {accuracy:.4f} ({correct}/{total})
            ###########################################################################
            '''
            )

    # dump binarized data
    if args.output_dir is not None:
        output_file = os.path.basename(args.generation_file).split('.json')[0] + "_bin.json"
        with open(os.path.join(args.output_dir, output_file), 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Binarized outputs saved to {os.path.join(args.output_dir, output_file)}")
        
        # Convert the data to Hugging Face datasets format
        if args.save_dataset:
            full_dataset = datasets.Dataset.from_list(output_data)

            # Always create both splits; use at least one example for test when possible
            test_size = max(1, int(len(full_dataset) * 0.01)) if len(full_dataset) > 1 else 1
            dataset_dict = full_dataset.train_test_split(test_size=test_size, shuffle=False)

            save_path = os.path.join(args.output_dir)
            dataset_dict.save_to_disk(save_path)
            print(f"Binarized dataset (train/test) saved to {save_path}")