from llama_index.llms.llama_cpp import LlamaCPP

from src.config import (
    PHI3_MODEL_PATH,
    CONTEXT_WINDOW,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


def load_phi3_llm():
    """Load local Phi-3 model through LlamaCPP."""
    return LlamaCPP(
        model_path=PHI3_MODEL_PATH,
        context_window=CONTEXT_WINDOW,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        verbose=False,
    )
