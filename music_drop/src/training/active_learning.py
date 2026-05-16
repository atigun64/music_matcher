import random
from .train import train_model
from .evaluate import evaluate_model

from .labeling import label_batch

from .sampling import build_pool, select_queries

def build_labeled_eval_set():
    pass

def active_learning_loop(
    train_track_ids,
    val_track_ids,
    rounds=10,
    initial_label_count=50,
    batch_size=20
):
    unlabeled_pool = build_pool(train_track_ids, heuristic_per_track=10, background_per_track=10)

    # Seed labels
    seed = random.sample(unlabeled_pool, min(initial_label_count, len(unlabeled_pool)))
    seed = label_batch(seed)

    labeled_samples = seed
    unlabeled_pool = [s for s in unlabeled_pool if (s.track_id, s.beat_idx) not in {(x.track_id, x.beat_idx) for x in seed}]

    # Proper validation set: label once and keep fixed
    val_pool = build_labeled_eval_set(val_track_ids, heuristic_per_track=10, background_per_track=10)

    best_ap = -1.0
    best_model = None
    patience = 3
    bad_rounds = 0

    for round_idx in range(rounds):
        print(f"\n=== Round {round_idx + 1}/{rounds} ===")

        model = train_model(labeled_samples)
        val_ap = evaluate_model(model, val_pool)
        print(f"Validation AP: {val_ap:.4f}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_model = model
            bad_rounds = 0
        else:
            bad_rounds += 1

        if bad_rounds >= patience:
            print("No improvement for a while, stopping.")
            break

        query_samples = select_queries(model, unlabeled_pool, batch_size=batch_size)
        if len(query_samples) == 0:
            print("No unlabeled samples left.")
            break

        query_samples = label_batch(query_samples)
        labeled_samples.extend(query_samples)

        query_keys = {(s.track_id, s.beat_idx) for s in query_samples}
        unlabeled_pool = [s for s in unlabeled_pool if (s.track_id, s.beat_idx) not in query_keys]

        print(f"Labeled samples total: {len(labeled_samples)}")

    return best_model, labeled_samples
