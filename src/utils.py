def score_label(score: float) -> str:
    """Convert a 0-1 score into a simple confidence label."""
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Moderate"
    elif score >= 0.40:
        return "Low"
    return "Very Low"
