import torch

def compute_trust_scores(updates, epsilon=1e-6):
    """
    Computes trust scores based on deviation from mean update.
    Implements the trust computation described in Section 4.
    """

    mean_update = {}
    for key in updates[0]:
        mean_update[key] = sum(u[key] for u in updates) / len(updates)

    distances = []
    for u in updates:
        d = 0.0
        for key in u:
            d += torch.norm(u[key] - mean_update[key]).item()
        distances.append(d)

    raw_trust = [1.0 / (d + epsilon) for d in distances]
    total = sum(raw_trust)
    normalized_trust = [t / total for t in raw_trust]

    return normalized_trust
