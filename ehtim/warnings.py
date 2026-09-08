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


class FeedTagWarning(UserWarning):
    """Emitted when the AIPS AN POLTYA/POLTYB feed tags of a uvfits file are
    missing, incomplete, or not in canonical feed order.

    Feed types are read per station from POLTYA (first feed) and POLTYB
    (second feed). Four situations are reported:

    * neither column is present -- the feed basis is inferred from the STOKES
      axis (CRVAL3) instead;
    * only one of the two columns is present -- each station's pair is
      completed from the tag that is there (``POLTYA='R'`` -> ``rl``,
      ``POLTYB='Y'`` -> ``xy``). A tag that is *blank* on a column that does
      exist is an error, not a warning: a writer that emits POLTY and leaves
      it empty says nothing about the basis;
    * the tags are in reversed order (``POLTYA='L'``, ``POLTYB='R'``) --
      the pair is canonicalized to ``rl`` on read, and canonicalized (with
      the four correlation slots permuted to match) on write, because the
      uvfits STOKES axis labels its planes absolutely;
    * the station tags imply a mixed feed basis while CRVAL3 names a
      homogeneous circular (-1) or linear (-5) product block -- the station
      tags win, since for a mixed array the STOKES axis is only nominal.
    """


class PolrepOverrideWarning(UserWarning):
    """Emitted when a loader returns a different polrep than was requested.

    A mixed-feed uvfits file can only be represented as ``polrep='mixed'``:
    converting out of the mixed basis needs Jones-level D-terms and is not
    available at the data layer. Requesting any other polrep for such a file
    returns a mixed-basis Obsdata and raises this warning rather than
    silently handing back something other than what was asked for.
    """
