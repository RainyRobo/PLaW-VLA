# Derived from openpi (Copyright 2024 Physical Intelligence, Inc.; Apache-2.0).
# Modified for PLaW-VLA by the PLaW-VLA authors, 2026.
import inspect
import transformers


def check_whether_transformers_replace_is_installed_correctly():
    """Check the pinned version and adaptive normalization API required by the policy."""
    if transformers.__version__ != "5.0.0":
        return False
    from transformers.models.gemma.modeling_gemma import GemmaModel, GemmaRMSNorm

    return (
        "cond" in inspect.signature(GemmaRMSNorm.forward).parameters
        and "adarms_cond" in inspect.signature(GemmaModel.forward).parameters
    )
