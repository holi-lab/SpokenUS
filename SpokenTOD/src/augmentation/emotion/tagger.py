"""Emotion tagger using LiteLLM-backed concurrent requests."""

import asyncio
import concurrent.futures

from augmentation.batch.client import BatchClient, LLMUsageTracker
from augmentation.constants import EMOTION_EXCLUDED, EMOTION_LABELS
from augmentation.emotion.prompts import (
    build_emotion_prompt,
    parse_emotion_response,
)


class EmotionTagger:
    """Tag emotions for dialogue utterances using GPT."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openrouter/qwen/qwen3-32b",
        max_retries: int = 5,
        max_workers: int = 10,
        usage_tracker: LLMUsageTracker | None = None,
    ):
        self.client = BatchClient(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            usage_tracker=usage_tracker,
            request_tag="emotion",
        )
        self.model = model
        self.max_workers = max_workers

    def tag_utterances(
        self,
        utterances: list[str],
        wait: bool = True,
        use_batch: bool = True,
        max_workers: int | None = None,
    ) -> list[dict]:
        """Tag emotions for a batch of utterances.

        Args:
            utterances: List of user utterances to tag
            wait: Kept for CLI compatibility; concurrent calls are always awaited
            use_batch: When True, route through the async concurrent path
            max_workers: Concurrency for the async or threaded path

        Returns:
            List of {"label": int, "name": str} dicts
        """
        if use_batch:
            workers = max_workers or self.max_workers
            return asyncio.run(
                self.tag_utterances_async(utterances, max_concurrency=workers)
            )

        if not wait:
            raise ValueError("wait=False is not supported without batch mode")

        if not use_batch:
            workers = max_workers or self.max_workers

            def tag_single(utt):
                prompt = build_emotion_prompt(utt)
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5
                )
                label = parse_emotion_response(response)
                name = EMOTION_LABELS.get(label, "neutral")
                return {"label": label, "name": name}

            if workers <= 1:
                a = [tag_single(utt) for utt in utterances]
                return a

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(tag_single, utterances))
            return results
        return []

    async def tag_utterances_async(
        self,
        utterances: list[str],
        max_concurrency: int = 50,
    ) -> list[dict]:
        """Tag emotions for a batch of utterances using async concurrent requests.

        Args:
            utterances: List of user utterances to tag
            max_concurrency: Maximum concurrent requests

        Returns:
            List of {"label": int, "name": str} dicts
        """
        if not utterances:
            return []

        # Build request list for concurrent processing
        request_list = []
        for utt in utterances:
            prompt = build_emotion_prompt(utt)
            request_list.append({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0.0,
            })

        # Run concurrent requests
        responses = await self.client.batch_completion_async(
            request_list,
            max_concurrency=max_concurrency
        )

        # Parse responses
        emotions = []
        for response in responses:
            label = parse_emotion_response(response)
            name = EMOTION_LABELS.get(label, "neutral")
            emotions.append({"label": label, "name": name})

        return emotions

    def should_tag(self, dataset: str) -> bool:
        """Check if dataset needs emotion tagging.

        Args:
            dataset: Dataset name

        Returns:
            True if tagging needed (not in EMOTION_EXCLUDED)
        """
        return dataset not in EMOTION_EXCLUDED


def tag_dialogue_emotions(
    turns: list[dict],
    dataset: str,
    tagger: EmotionTagger | None = None,
    existing_labels: list[int] | None = None,
) -> list[dict]:
    """Tag emotions for dialogue turns.

    Args:
        turns: List of turn dicts with "role" and "text"
        dataset: Dataset name
        tagger: EmotionTagger instance (created if None)
        existing_labels: Pre-existing labels (for EmoWOZ)

    Returns:
        List of turns with added emotion field
    """
    # If dataset already has labels, use them
    if existing_labels:
        result = []
        for i, turn in enumerate(turns):
            new_turn = dict(turn)
            if turn["role"] == "user" and i < len(existing_labels):
                label = existing_labels[i]
                if label >= 0:  # -1 means system turn
                    new_turn["emotion"] = {
                        "label": label,
                        "name": EMOTION_LABELS.get(label, "neutral"),
                    }
            result.append(new_turn)
        # Continue to propagate to cross-turn segments below
    else:
        # Skip if dataset excluded
        if dataset in EMOTION_EXCLUDED:
            return turns

        # Create tagger if needed
        if tagger is None:
            tagger = EmotionTagger()

        # Collect user utterances (skip cross-turn segments - they inherit emotion)
        user_indices = []
        user_texts = []
        for i, turn in enumerate(turns):
            if turn["role"] == "user" and not turn.get("segment"):
                user_indices.append(i)
                user_texts.append(turn["text"])

        # Tag emotions
        emotions = tagger.tag_utterances(user_texts, use_batch=False)

        # Apply to turns
        result = list(turns)
        for idx, emotion in zip(user_indices, emotions):
            result[idx] = dict(result[idx])
            result[idx]["emotion"] = emotion

    # Propagate emotion to cross-turn segments (inherit from previous turn)
    last_emotion = None
    for i, turn in enumerate(result):
        if turn["role"] == "user":
            if turn.get("segment"):
                # Cross-turn segment: inherit previous emotion
                if last_emotion is None:
                    last_emotion = {"label": 0, "name": EMOTION_LABELS.get(0, "neutral")}
                result[i] = dict(result[i])
                result[i]["emotion"] = last_emotion
            else:
                # Regular turn: update last_emotion
                last_emotion = turn.get("emotion")

    return result
