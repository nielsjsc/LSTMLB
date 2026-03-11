"""
Pitcher Post-Processing Pipeline
=================================

Previously applied output regression, aging constraints, and FIP/ERA/SIERA
reconstruction as a post-processing step on the final predictions DataFrame.

As of March 2026, ALL of these adjustments now happen INSIDE the autoregressive
prediction loop in core/pitcher_prediction.py. This ensures that:

1. Output-regressed component rates feed back into the model (preventing
   compounding of extreme values for short-career pitchers).
2. Aging constraints are applied each year BEFORE the prediction feeds back,
   stopping trend extrapolation from compounding across years.
3. HR/FB, BABIP, and other rate stats that drive FIP/ERA are constrained
   in-loop, so derived stats inherit realistic inputs at every step.

The post_process_pitcher_predictions() function is retained as a pass-through
for backward compatibility with any callers that still invoke it.
"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def post_process_pitcher_predictions(
    pitcher_df: pd.DataFrame,
    role: str,
    pitching_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    No-op pass-through — all post-processing now happens in the autoregressive
    prediction loop (core/pitcher_prediction.py).

    Retained for backward compatibility.
    """
    logger.info(
        f"{role} post-processing: all adjustments now applied in-loop, "
        f"returning predictions unchanged ({len(pitcher_df)} rows)"
    )
    return pitcher_df.copy()
