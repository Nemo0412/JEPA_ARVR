#!/usr/bin/env python3
"""B17 gaze/spatial-selector evaluation entrypoint.

Temporal-allocation experiments use ``eval_stream_mtp_temporal_prune.py``.
This entrypoint admits only the registered gaze/spatial protocol family.
"""
from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    PROTOCOL_ID,
    PROTOCOL_VARIANTS,
    RANDOM_PROTOCOL_ID,
    main,
)


if __name__ == "__main__":
    main(
        allowed_protocol_ids=(PROTOCOL_ID, RANDOM_PROTOCOL_ID),
        default_protocol_id=PROTOCOL_ID,
        default_variants=PROTOCOL_VARIANTS,
    )
