def label_flipping(labels):
    """
    Implements label flipping attack.
    """
    return 1 - labels

def sign_flipping(update):
    """
    Implements sign flipping model poisoning.
    """
    return {k: -v for k, v in update.items()}

def scaling_attack(update, factor=10):
    """
    Implements scaling based poisoning attack.
    """
    return {k: factor * v for k, v in update.items()}
