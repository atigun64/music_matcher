from models import Alignment, PointAnnotation
from .utils import clamp01

from .drop import score_alignment_drops_final, score_alignment_drops_partial
from .bpm import score_average_bpm
from .style import score_alignment_style
from .preference import score_average_preference
from .gap import score_gap_quality_final, score_gap_quality_partial
from .overlap import score_overlap_quality

from .config import (
    WEIGHT_BPM,
    WEIGHT_DROPS,
    WEIGHT_PREFERENCE,
    WEIGHT_STYLE,
    WEIGHT_OVERLAP,
    WEIGHT_GAP,
)

def score_alignment_final(alignment: Alignment, query):
    """
    Final completed-alignment score.
    Returns [0,1].
    """
    requested_drops = [
        a for a in query.annotations
        if isinstance(a, PointAnnotation) and a.label == "drop"
    ]

    drop_score = score_alignment_drops_final(alignment, requested_drops)
    bpm_score = score_average_bpm(alignment, query.length)
    style_score = score_alignment_style(alignment, query.signature)
    preference_score = score_average_preference(alignment)
    gap_score = score_gap_quality_final(alignment, query.length)
    overlap_score = score_overlap_quality(alignment)

    critical = (
        drop_score ** 0.6 *
        gap_score ** 0.3 *
        overlap_score ** 0.1
    )
    secondary = (
        0.4 * style_score +
        0.3 * bpm_score +
        0.3 * preference_score
    )

    if critical < 0.6:
        final = critical
    else:
        final = 0.85 * critical + 0.15 * secondary
    return final


def score_alignment_partial(alignment: Alignment, query):
    """
    Partial beam-search score.
    Returns [0,1].

    Future drop requests are ignored.
    """
    requested_drops = [
        a for a in query.annotations
        if isinstance(a, PointAnnotation) and a.label == "drop"
    ]


    drop_score = score_alignment_drops_partial(alignment, requested_drops)
    bpm_score = score_average_bpm(alignment, query.length)
    style_score = score_alignment_style(alignment, query.signature)
    preference_score = score_average_preference(alignment)
    gap_score = score_gap_quality_partial(alignment, query.length)
    overlap_score = score_overlap_quality(alignment)

    critical = (
        drop_score ** 0.6 *
        gap_score ** 0.3 *
        overlap_score ** 0.1
    )
    secondary = (
        0.4 * style_score +
        0.3 * bpm_score +
        0.3 * preference_score
    )

    if critical < 0.6:
        final = critical
    else:
        final = 0.85 * critical + 0.15 * secondary
    return final
