#!/usr/bin/env python3
"""B17 temporal-allocation evaluation entrypoint.

This entrypoint is isolated from the gaze/spatial experiment surface and
admits only the registered temporal-allocation protocol.
"""
from app.hdepic_lora_action_anticipation.eval_stream_mtp_fixed_budget_prune import (
    TEMPORAL_PROTOCOL_ID,
    TEMPORAL_PROTOCOL_VARIANTS,
    main,
)


if __name__ == "__main__":
    main(
        allowed_protocol_ids=(TEMPORAL_PROTOCOL_ID,),
        default_protocol_id=TEMPORAL_PROTOCOL_ID,
        default_variants=TEMPORAL_PROTOCOL_VARIANTS,
    )
