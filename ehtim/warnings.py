"""Custom warning classes for ehtim.

Centralized home for ehtim-specific warning categories. New warning classes
should be added here so users can suppress categories via the standard
warnings machinery:

    import warnings
    import ehtim.warnings as ehw
    warnings.filterwarnings('ignore', category=ehw.MixedPolConventionWarning)
"""


class MixedPolConventionWarning(UserWarning):
    """Emitted on first non-trivial polrep conversion or Jones application
    each session.

    Notes the polarization-basis convention currently in use (defined in
    ``ehtim/observing/pol_conventions.py``) and the ideal-feed (D = 0)
    assumption underlying most basis transforms. Suppressible:

        warnings.filterwarnings(
            'ignore', category=ehtim.warnings.MixedPolConventionWarning
        )
    """


class MixedPolClosureSkipWarning(UserWarning):
    """Emitted when closure quantities skip triangles or quadrangles whose
    feed-type combination makes the closure unphysical.

    A bispectrum requires three visibilities of the same correlation around
    a triangle, and a closure amplitude requires four visibilities of the
    same correlation around a quadrangle. RR closure phases on a triangle
    that includes a non-circular-feed station, or RR closure amplitudes on
    a quadrangle that includes one, do not physically exist without
    Jones-level conversion. The warning summarizes, per call, how many
    triangles or quadrangles were skipped and why.
    """


class MixedPolUnpackNaNWarning(UserWarning):
    """Emitted when ``Obsdata.unpack`` of a physical correlation (e.g. 'rrvis')
    on a mixed-feed observation returns NaN for rows whose feed basis does not
    measure that correlation (e.g. RR on a circular x linear baseline).

    The warning reports, per field, how many rows were NaN-filled. It is
    deliberately verbose; suppress with the standard machinery if needed.
    """
