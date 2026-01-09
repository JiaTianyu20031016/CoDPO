import argparse
import json
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description="Compute average RM scores and win rate for generated vs GPT-4 responses.")
	parser.add_argument("--input", type=Path, required=True, help="Path to *_rm.json produced by annotate.py")
	parser.add_argument(
		"--output",
		type=Path,
		help="Where to write summary JSON. Default: <input> with _summary suffix.",
	)
	args = parser.parse_args()

	with args.input.open() as f:
		records = json.load(f)

	model_scores = []
	gpt4_scores = []
	wins = 0

	for rec in records:
		scores = rec.get("all_rm_scores")
		if not scores or len(scores) < 2:
			continue
		model_scores.append(scores[0])
		gpt4_scores.append(scores[1])
		if scores[0] >= scores[1]:
			wins += 1

	if not model_scores or not gpt4_scores:
		raise ValueError("No valid all_rm_scores entries with at least two scores were found.")

	total = len(model_scores)
	model_avg = sum(model_scores) / total
	gpt4_avg = sum(gpt4_scores) / total
	win_rate = wins / total

	summary = {
		"evaluated_model_average": model_avg,
		"gpt4_average": gpt4_avg,
		"win_rate": win_rate,
		"count": total,
	}

	output_path = args.output or args.input.with_name(args.input.stem + "_summary.json")
	with output_path.open("w") as f:
		json.dump(summary, f, indent=2)

	print(f"evaluated model average score: {model_avg:.6f}")
	print(f"GPT4 average score: {gpt4_avg:.6f}")
	print(f"win rate: {win_rate:.6f}")
	print(f"summary written to: {output_path}")


if __name__ == "__main__":
	main()
