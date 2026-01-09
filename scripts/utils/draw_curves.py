import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


def load_log_history(path: Path) -> List[Dict]:
	"""Load trainer_state-style JSON and return its log_history entries."""

	with path.open() as f:
		data = json.load(f)

	if "log_history" not in data:
		raise ValueError(f"Expected 'log_history' in {path}")

	return data["log_history"]


def infer_metrics(log_history: Sequence[Dict], explicit: Iterable[str] | None) -> List[str]:
	"""Pick metrics to plot.

	If metrics are provided explicitly, use them. Otherwise, prefer a small
	default set common to DPO runs and fall back to keys containing accuracy,
	margin, or loss.
	"""

	if explicit:
		return list(dict.fromkeys(explicit))

	default_candidates = ["rewards/accuracies", "rewards/margins", "loss"]
	present = {m for m in default_candidates if any(m in rec or f"eval_{m}" in rec for rec in log_history)}
	if present:
		return sorted(present)

	keywords = ("accuracy", "accuracies", "margin", "margins", "loss")
	inferred = set()
	for rec in log_history:
		for key in rec:
			base = key[5:] if key.startswith("eval_") else key
			if any(k in base for k in keywords):
				inferred.add(base)
	if not inferred:
		raise ValueError("Could not infer metrics to plot; please pass --metrics explicitly.")
	return sorted(inferred)


def collect_series(log_history: Sequence[Dict], metric: str) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
	"""Return (train, eval) series for a metric keyed by step."""

	train_series: List[Tuple[int, float]] = []
	eval_series: List[Tuple[int, float]] = []
	eval_key = f"eval_{metric}"

	for rec in log_history:
		step = rec.get("step")
		if step is None:
			continue
		if metric in rec:
			train_series.append((step, rec[metric]))
		if eval_key in rec:
			eval_series.append((step, rec[eval_key]))

	train_series.sort(key=lambda x: x[0])
	eval_series.sort(key=lambda x: x[0])
	return train_series, eval_series


def plot_metric(metric: str, train_series: List[Tuple[int, float]], eval_series: List[Tuple[int, float]], output_dir: Path) -> Path:
	"""Plot one metric and save the figure."""

	if not train_series and not eval_series:
		raise ValueError(f"Metric '{metric}' not found in log history.")

	output_dir.mkdir(parents=True, exist_ok=True)
	save_path = output_dir / f"{metric.replace('/', '_')}.png"

	plt.figure(figsize=(8, 4))
	if train_series:
		steps, values = zip(*train_series)
		plt.plot(steps, values, label="train", linewidth=2)
	if eval_series:
		steps_eval, values_eval = zip(*eval_series)
		plt.plot(steps_eval, values_eval, label="eval", linewidth=2, linestyle="--")

	plt.xlabel("step")
	plt.ylabel(metric)
	plt.title(metric)
	plt.legend()
	plt.tight_layout()
	plt.savefig(save_path)
	plt.close()
	return save_path


def main():
	parser = argparse.ArgumentParser(description="Plot metrics from trainer_state.json-like files.")
	parser.add_argument("--input", type=Path, default=Path("trainer_state.json"), help="Path to trainer_state JSON file.")
	parser.add_argument(
		"--metrics",
		nargs="+",
		default=["rewards/accuracies", "rewards/margins", "loss", "logps/chosen", "logps/rejected", "grad_norm"],
		help="Metric base names to plot (omit eval_ prefix). Example: rewards/accuracies rewards/margins loss.",
	)
	parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save figures.")
	args = parser.parse_args()
	if args.output_dir is None:
		args.output_dir = args.input.parent / "plots"

	log_history = load_log_history(args.input)
	metrics = infer_metrics(log_history, args.metrics)

	saved: List[Path] = []
	for metric in metrics:
		train_series, eval_series = collect_series(log_history, metric)
		try:
			saved.append(plot_metric(metric, train_series, eval_series, args.output_dir))
		except ValueError as exc:
			print(exc)

	if saved:
		print("Saved plots:")
		for path in saved:
			print(path)
	else:
		print("No plots were generated.")


if __name__ == "__main__":
	main()
