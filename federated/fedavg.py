def fedavg(updates):
    """
    Standard federated averaging.
    Used as baseline in experimental evaluation.
    """
    aggregated = {}
    for key in updates[0]:
        aggregated[key] = sum(u[key] for u in updates) / len(updates)
    return aggregated
