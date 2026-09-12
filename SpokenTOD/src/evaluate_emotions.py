"""Script to evaluate LLM emotion tagging accuracy against EmoWOZ ground truth."""

import argparse
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import List, Optional

import pandas as pd
from sklearn.metrics import classification_report

from augmentation.loaders.emowoz import EmoWOZLoader
from augmentation.emotion.tagger import EmotionTagger
from augmentation.emotion.prompts import build_emotion_prompt, parse_emotion_response, parse_emotion_think_response
from augmentation.constants import EMOTION_TOKENS, EMOTION_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_file(env_path: Path):
    """Manually parse .env file to set environment variables."""
    if not env_path.exists():
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip().strip("'").strip('"')
                os.environ[key.strip()] = value

def evaluate_emotions(
    data_dir: str,
    multiwoz_dir: str,
    sample_size: int = 50,
    api_key: Optional[str] = None,
    workers: int = 10,
    model: str = "gpt-4.1-mini",
    split: str = "train",
):
    """Run emotion evaluation on EmoWOZ subset."""
    data_path = Path(data_dir)
    mwoz_path = Path(multiwoz_dir)
    
    # Load .env if it exists in root
    load_env_file(Path(".env"))
    
    # Priority: 1. Argument, 2. Environment variable
    final_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not final_api_key:
        logger.error("API key not found. Please provide --api-key or set OPENAI_API_KEY in .env")
        return

    loader = EmoWOZLoader(data_path, mwoz_path, split=split)
    tagger = EmotionTagger(api_key=final_api_key, max_workers=workers, model=model)
    
    logger.info(f"Collecting utterances from EmoWOZ {split} set for balanced sampling...")
    
    buckets = {i: [] for i in range(7)}
    dialogue_pool_size = 4000  # Scan up to 4000 dialogues to fill buckets

    dialogue_count = 0
    for dialogue in loader.load():
        if dialogue_count >= dialogue_pool_size:
            break

        for i, turn in enumerate(dialogue.turns):
            if turn["role"] == "user":
                label = dialogue.emotion_labels[i]
                if 0 <= label <= 6:
                    # Collect context (all turns before this one)
                    context = []
                    for j, ctx_turn in enumerate(dialogue.turns[:i]):
                        ctx_item = dict(ctx_turn)
                        if ctx_item["role"] == "user":
                            ctx_item["emotion"] = dialogue.emotion_labels[j]
                        context.append(ctx_item)
                    buckets[label].append({
                        "text": turn["text"],
                        "dialogue_id": dialogue.id,
                        "label": label,
                        "context": context,
                    })
        dialogue_count += 1

    # Balanced sampling
    import random
    sampled_utterances = []
    
    logger.info(f"Sampling up to {sample_size} utterances per class...")
    for label, items in buckets.items():
        if not items:
            logger.warning(f"No samples found for label {label} ({EMOTION_LABELS[label]})")
            continue
            
        num_to_sample = min(len(items), sample_size)
        sampled = random.sample(items, num_to_sample)
        sampled_utterances.extend(sampled)
        logger.info(f"Label {label} ({EMOTION_LABELS[label]}): Collected {len(sampled)} samples")

    if not sampled_utterances:
        logger.error("No data collected for evaluation.")
        return

    # Shuffle sampled utterances for unpredictable order
    random.shuffle(sampled_utterances)

    y_true = []
    y_pred = []
    comparison_data = []
    
    total_to_process = len(sampled_utterances)
    logger.info(f"Tagging {total_to_process} balanced utterances using {workers} workers...")
    
    try:
        # Build prompts with context
        prompts = []
        for item in sampled_utterances:
            
            prompt = build_emotion_prompt(
                item["text"],
                context=item["context"],
            )
            prompts.append(prompt)
        # Tag with context using parallel workers
        import concurrent.futures

        def tag_single(prompt):
            response = tagger.client.chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            # return parse_emotion_response(response)
            return parse_emotion_think_response(response)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            predictions = list(executor.map(tag_single, prompts))
            
        for item, pred_label in zip(sampled_utterances, predictions):
            gt = item["label"]
            y_true.append(gt)
            y_pred.append(pred_label)

            comparison_data.append({
                "dialogue_id": item["dialogue_id"],
                "text": item["text"],
                "context_length": len(item["context"]),
                "ground_truth": gt,
                "gt_name": EMOTION_LABELS[gt],
                "predicted": pred_label,
                "pred_name": EMOTION_LABELS.get(pred_label, "unknown"),
                "match": gt == pred_label
            })

    except Exception as e:
        logger.error(f"Error during tagging: {e}")
        import traceback
        traceback.print_exc()

    if not y_true:
        logger.error("No data collected for evaluation.")
        return

    # Results
    df = pd.DataFrame(comparison_data)
    
    # Save detailed CSV
    output_dir = Path("evaluation", time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "emotion_comparison.csv", index=False)
    
    # Metrics
    labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [f"{EMOTION_LABELS.get(l, f'ID {l}')}" for l in labels]
    
    report = classification_report(
        y_true, 
        y_pred, 
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    
    print("\n" + "="*50)
    print("EMOTION TAGGING EVALUATION RESULTS (EmoWOZ Labels)")
    print("="*50)
    print(f"Pool Scanned: {dialogue_count} dialogues")
    print(f"Total Utterances evaluated: {len(y_true)}")
    print("-" * 50)
    print("Classification Report:")
    print(report)
    print("="*50)
    
    def merge_label(label: int) -> int:
        return 1 if label in (1, 2) else label

    y_true_relaxed = [merge_label(l) for l in y_true]
    y_pred_relaxed = [merge_label(l) for l in y_pred]
    relaxed_labels = [0, 1, 3, 4, 5, 6]
    relaxed_names = [
        EMOTION_LABELS[0],
        "fearful_or_dissatisfied",
        EMOTION_LABELS[3],
        EMOTION_LABELS[4],
        EMOTION_LABELS[5],
        EMOTION_LABELS[6],
    ]
    relaxed_report = classification_report(
        y_true_relaxed,
        y_pred_relaxed,
        labels=relaxed_labels,
        target_names=relaxed_names,
        zero_division=0,
    )
    print("Classification Report (Relaxed 1/2 merged):")
    print(relaxed_report)
    print("="*50)

    
    # Save report with settings
    with open(output_dir / "emotion_report.txt", "w") as f:
        f.write("=== Settings ===\n")
        f.write(f"Model: {model}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Context: full\n")
        f.write(f"Few-shot context: False\n")
        f.write(f"Sample size per class: {sample_size}\n")
        f.write("\n=== Classification Report (EmoWOZ Labels) ===\n")
        f.write(report)
        f.write(f"\nTotal Utterances: {len(y_true)}")
        f.write(f"\nPool Scanned: {dialogue_count} dialogues")
        
        f.write("\n=== Classification Report (EmoWOZ Labels) ===\n")
        f.write(report)
        
        f.write("\n\n=== Classification Report (Relaxed 1/2 merged) ===\n")
        f.write(relaxed_report)
        f.write(f"\nTotal Utterances: {len(y_true)}")
        f.write(f"\nPool Scanned: {dialogue_count} dialogues")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/EmoWOZ")
    parser.add_argument("--multiwoz-dir", default="datasets/MultiWOZ_2.1")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers for real-time tagging")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model to use for emotion tagging")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "validation", "dev", "test"],
        default="train",
        help="EmoWOZ split to evaluate",
    )

    args = parser.parse_args()

    evaluate_emotions(
        args.data_dir,
        args.multiwoz_dir,
        sample_size=args.sample_size,
        api_key=args.api_key,
        workers=args.workers,
        model=args.model,
        split=args.split,
    )
