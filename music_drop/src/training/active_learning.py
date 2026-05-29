import random

from .train import train_model
from .labeling import label_batch, load_labeled_samples, save_labeled_samples
from .sampling import build_pool, select_queries, pool_filter


def active_learning_loop(
    train_track_ids,
    rounds=10,
    initial_label_count=50,
    batch_size=20,
    split="train",
):
    # Load previously saved labeled samples
    labeled_samples = load_labeled_samples(split=split)

    # Build current unlabeled pool and filter out already labeled samples
    unlabeled_pool = pool_filter(build_pool(train_track_ids), labeled_samples)

    print(f"Initial unlabeled pool size: {len(unlabeled_pool)}")

    # If this is the first run, label an initial seed batch
    if len(labeled_samples) == 0:
        if len(unlabeled_pool) == 0:
            print("No samples available to label.")
            return None, []

        seed = random.sample(
            unlabeled_pool,
            min(initial_label_count, len(unlabeled_pool))
        )
        seed = label_batch(seed)
        save_labeled_samples(seed, split=split)

        labeled_samples.extend(seed)

        seed_keys = {(s.track_id, s.beat_idx) for s in seed}
        unlabeled_pool = [
            s for s in unlabeled_pool
            if (s.track_id, s.beat_idx) not in seed_keys
        ]

    model = None

    for round_idx in range(rounds):
        print(f"\n=== Round {round_idx + 1}/{rounds} ===")

        if len(labeled_samples) == 0:
            print("No labeled samples available for training.")
            break

        model = train_model(labeled_samples)

        if len(unlabeled_pool) == 0:
            print("No unlabeled samples left.")
            break

        query_samples = select_queries(model, unlabeled_pool, batch_size=batch_size)
        if len(query_samples) == 0:
            print("No query samples selected.")
            break

        query_samples = label_batch(query_samples)
        save_labeled_samples(query_samples, split=split)

        labeled_samples.extend(query_samples)

        query_keys = {(s.track_id, s.beat_idx) for s in query_samples}
        unlabeled_pool = [
            s for s in unlabeled_pool
            if (s.track_id, s.beat_idx) not in query_keys
        ]

        print(f"Labeled samples total: {len(labeled_samples)}")

    return model, labeled_samples
