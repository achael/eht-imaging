"""Interactive Plotly-based plotting for ehtim.

Sits beside the matplotlib defaults in 'comp_plots.py', 'summary_plots.py',
and the 'plot_*' methods on 'Obsdata' / 'Caltable'.

Plotly is imported at module load; installing it is required to use this
module. The error message names the install command.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import plotly.colors as _plotly_colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError("Plotly is required for ehtim.plotting.interactive. Install with `pip install plotly`.") from e

import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh

if TYPE_CHECKING:
    from os import PathLike

    from ehtim.caltable import Caltable
    from ehtim.image import Image
    from ehtim.obsdata import Obsdata


# --- Shared aesthetics (BMH Style) ----------------------------------------

_THEME = {
    "width": 820,
    "height": 460,
    "font_family": ('Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'),
    "font_size": 12,
    "title_size": 16,
    # Matplotlib 'bmh' (Bayesian Methods for Hackers) palette + extensions.
    "colorway": [
        "#348ABD",
        "#A60628",
        "#7A68A6",
        "#4B8022",
        "#D55E00",
        "#CC79A7",
        "#1F471F",
        "#2C825D",
    ],
    "plot_bgcolor": "#eeeeee",
    "paper_bgcolor": "#ffffff",
    "grid_color": "#ffffff",
    "zero_color": "#ffffff",
    "edge_color": "#bcbcbc",
    "font_color": "#333333",
    "marker_edge": "rgba(0,0,0,0.4)",
    "error_color": "rgba(0,0,0,0.3)",
}

# Fallback symbols when a plot has more traces than the colorway has colors.
_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"]

# Initial colour for traces that participate in the gray↔colour legend flow.
_GRAY = "#888888"

# Fields whose values are in raw λ and should auto-scale to Gλ/Mλ.
_UV_FIELDS = frozenset({"u", "v", "uvdist"})

# Plotly-friendly axis labels for fields whose `ehc.FIELD_LABELS` entry is
# LaTeX-wrapped (`$...$`). Matplotlib renders the LaTeX; Plotly shows the
# raw string. Subscripts become Plotly's `<sub>` HTML tags. Fields not
# listed fall through to `ehc.FIELD_LABELS`.
_PLOTLY_FIELD_LABELS = {
    "u": "u",
    "v": "v",
    "uvdist": "u-v Distance",
    "tau1": "τ<sub>1</sub>",
    "tau2": "τ<sub>2</sub>",
    "el1": "Elevation Angle<sub>1</sub>",
    "el2": "Elevation Angle<sub>2</sub>",
    "hr_ang1": "Hour Angle<sub>1</sub>",
    "hr_ang2": "Hour Angle<sub>2</sub>",
    "par_ang1": "Parallactic Angle<sub>1</sub>",
    "par_ang2": "Parallactic Angle<sub>2</sub>",
    "sigma": "σ",
    "qsigma": "σ<sub>Q</sub>",
    "usigma": "σ<sub>U</sub>",
    "vsigma": "σ<sub>V</sub>",
    "sigma_phase": "σ<sub>phase</sub>",
    "qsigma_phase": "σ<sub>Q phase</sub>",
    "usigma_phase": "σ<sub>U phase</sub>",
    "vsigma_phase": "σ<sub>V phase</sub>",
    "psigma_phase": "σ<sub>P phase</sub>",
    "msigma_phase": "σ<sub>m phase</sub>",
    "rrsigma": "σ<sub>RR</sub>",
    "rrsigma_phase": "σ<sub>RR phase</sub>",
    "llsigma": "σ<sub>LL</sub>",
    "llsigma_phase": "σ<sub>LL phase</sub>",
    "rlsigma": "σ<sub>RL</sub>",
    "rlsigma_phase": "σ<sub>RL phase</sub>",
    "lrsigma": "σ<sub>LR</sub>",
    "lrsigma_phase": "σ<sub>LR phase</sub>",
    "rrllsigma": "σ<sub>RR/LL</sub>",
    "rrllsigma_phase": "σ<sub>RR/LL phase</sub>",
}


def _field_label(field: str) -> str:
    """Plotly-friendly label for `field`, fallback to `ehc.FIELD_LABELS`."""
    if field in _PLOTLY_FIELD_LABELS:
        return _PLOTLY_FIELD_LABELS[field]
    return ehc.FIELD_LABELS.get(field, field.capitalize())


def _format_uv_axis(values: np.ndarray) -> tuple[np.ndarray, str]:
    """Auto-scale a uv array (λ) to Gλ or Mλ based on its dynamic range.

    Returns the rescaled array and the unit label. Threshold: switch to Gλ
    iff any finite |value| ≥ 1e9, else Mλ.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, "λ"
    vmax = float(np.max(np.abs(finite)))
    if vmax >= 1e9:
        return arr / 1e9, "Gλ"
    return arr / 1e6, "Mλ"


def _apply_theme(
    fig,
    *,
    title,
    xaxis_title,
    yaxis_title,
    rangex=None,
    rangey=None,
    y_type="linear",
    legend_title="",
    show_legend=True,
    add_reset_button=False,
):
    """Apply the BMH visual theme to a plotly Figure.

    `add_reset_button=True` installs a "Show all / reset" button that
    restores managed (gray-tagged) traces to visible + gray. The set of
    managed indices is read from `fig.data` at theme-application time:
    every trace with `meta.legend_kind="gray"` is included.
    """
    layout = dict(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=_THEME["title_size"], family=_THEME["font_family"], color=_THEME["font_color"]),
        ),
        xaxis=dict(
            title=xaxis_title,
            range=rangex,
            gridcolor=_THEME["grid_color"],
            zerolinecolor=_THEME["zero_color"],
            zerolinewidth=1.5,
            linecolor=_THEME["edge_color"],
            linewidth=1,
            ticks="inside",
            tickcolor=_THEME["edge_color"],
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            title=yaxis_title,
            range=rangey,
            type=y_type,
            gridcolor=_THEME["grid_color"],
            zerolinecolor=_THEME["zero_color"],
            zerolinewidth=1.5,
            linecolor=_THEME["edge_color"],
            linewidth=1,
            ticks="inside",
            tickcolor=_THEME["edge_color"],
            showline=True,
            mirror=True,
        ),
        template="none",
        plot_bgcolor=_THEME["plot_bgcolor"],
        paper_bgcolor=_THEME["paper_bgcolor"],
        font=dict(family=_THEME["font_family"], size=_THEME["font_size"], color=_THEME["font_color"]),
        margin=dict(l=70, r=160, t=80 if add_reset_button else 60, b=60),
        width=_THEME["width"],
        height=_THEME["height"],
        hovermode="closest",
        colorway=_THEME["colorway"],
        showlegend=show_legend,
        legend=dict(
            title=dict(text=legend_title),
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            font=dict(size=11, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    # The "Color" / "Show all / reset" toolbar is injected by the
    # post-script JS as HTML buttons (siblings of the plotly div), so we
    # don't add anything to fig.layout here. `add_reset_button` only
    # changes the top margin so the injected toolbar has breathing room.
    fig.update_layout(**layout)


_GRAY_OPACITY = 0.6
_COLOR_OPACITY = 0.85
_OPACITY_THRESHOLD = 0.7  # < threshold → "gray", >= threshold → "coloured"


# --- Click-to-highlight JS ------------------------------------------------

# Legend interaction follows an explicit two-mode design:
#
#   - **Color mode OFF (default).** Single-click on a legend entry uses
#     plotly's built-in toggle (visible ↔ legendonly). No custom JS in
#     the click path — plotly's defaults are predictable.
#
#   - **Color mode ON.** The "Color" toolbar button toggles a JS flag.
#     While the flag is on, single-click on a managed trace (one tagged
#     `meta.legend_kind="gray"` Python-side) paints it from the palette
#     (gray → coloured) or returns it to gray (coloured → gray); plotly's
#     default visibility toggle is suppressed via `return false`.
#
#   - **Show all / reset.** Restyles every managed trace back to gray +
#     visible and exits color mode.
#
# Why HTML buttons instead of plotly `updatemenus`: plotly re-renders its
# layout DOM on certain updates and (depending on plotly.js version) may
# drop custom event listeners attached via `gd.on(...)`. Siblings of the
# plotly div live outside plotly's React tree and survive re-renders.
# We also re-bind the legend handler on `plotly_afterplot` as a defensive
# safeguard.
#
# Robustness fixes (per the user's debugging notes on PR #269):
#   - `isGray()` accepts both scalar opacity and the vectorised
#     `[0.6, 0.6, ...]` form plotly emits during rapid updates.
#   - Toolbar buttons own their state explicitly (no closure-shared dict
#     that depends on plotly's normalization of color strings).
#
# `{plot_id}` is the placeholder plotly substitutes when this is passed
# as `post_script` to `to_html` / `write_html`.


def _legend_click_js(managed_indices: list[int] | None = None) -> str:
    """Build the post-script JS that wires up the Color/Reset toolbar.

    `managed_indices` are the trace indices that participate in the
    gray↔colour flow — baked into the JS as a constant so detection does
    *not* depend on `tr.meta` surviving plotly's internal restyles
    (it doesn't, reliably).
    """
    indices = list(managed_indices or [])
    palette = json.dumps(_THEME["colorway"])
    gray = json.dumps(_GRAY)
    indices_json = json.dumps(indices)
    return f"""
(function initToolbar() {{
    var gd = document.getElementById('{{plot_id}}');
    // In standalone HTML the post_script can fire before plotly finishes
    // attaching the plot div. Poll briefly until the div is present and
    // plotly-initialised, then run the toolbar setup.
    if (!gd || !gd.classList || !gd.classList.contains('js-plotly-plot')) {{
        return setTimeout(initToolbar, 50);
    }}
    var PALETTE = {palette};
    var GRAY = {gray};
    var THRESHOLD = {_OPACITY_THRESHOLD};
    var GRAY_OPACITY = {_GRAY_OPACITY};
    var COLOR_OPACITY = {_COLOR_OPACITY};
    var MANAGED_LIST = {indices_json};
    var MANAGED = {{}};
    for (var _i = 0; _i < MANAGED_LIST.length; _i++) MANAGED[MANAGED_LIST[_i]] = true;
    var colorMode = false;

    if (MANAGED_LIST.length === 0) return;

    function isManaged(idx) {{
        return MANAGED[idx] === true;
    }}

    // Robust gray check: plotly can vectorise opacity into an array
    // ([0.6, 0.6, ...]) during rapid restyles; pull a scalar safely
    // and default to "gray" on uncertainty so a hidden-but-managed
    // trace gets repainted on click.
    function isGray(tr) {{
        if (!tr || !tr.marker) return true;
        var op = tr.marker.opacity;
        if (Array.isArray(op)) op = op.length ? op[0] : undefined;
        if (typeof op !== 'number') return true;
        return op < THRESHOLD;
    }}

    // --- Toolbar (HTML siblings of the plotly div) -----------------------

    var colorBtn, resetBtn;
    function syncColorBtn() {{
        if (!colorBtn) return;
        colorBtn.textContent = colorMode
            ? 'Color: on — click a baseline'
            : 'Color: off';
        colorBtn.style.background = colorMode
            ? 'rgba(180,220,180,0.95)'
            : 'rgba(238,238,238,0.92)';
    }}

    // Look for an existing toolbar anywhere in the document (in case the
    // plot div's parent has been swapped by plotly since first insert).
    var existingBar = document.querySelector('[data-ehtim-toolbar-for="' + gd.id + '"]');
    if (!existingBar) {{
        var bar = document.createElement('div');
        bar.dataset.ehtimToolbar = '1';
        bar.setAttribute('data-ehtim-toolbar-for', gd.id);
        bar.style.cssText = (
            'margin: 6px 0 4px 8px; ' +
            'font-family: Inter, system-ui, sans-serif; ' +
            'display: flex; gap: 6px; align-items: center; ' +
            'position: relative; z-index: 10; ' +
            'visibility: visible !important;'
        );

        function makeBtn(label) {{
            var b = document.createElement('button');
            b.type = 'button';
            b.textContent = label;
            b.style.cssText = (
                'font: 11px Inter, system-ui, sans-serif; ' +
                'padding: 4px 10px; ' +
                'background: rgba(238,238,238,0.92); color: #333; ' +
                'border: 1px solid #bcbcbc; border-radius: 3px; ' +
                'cursor: pointer; user-select: none;'
            );
            b.addEventListener('mouseenter', function() {{
                b.style.borderColor = '#888';
            }});
            b.addEventListener('mouseleave', function() {{
                b.style.borderColor = '#bcbcbc';
            }});
            return b;
        }}

        colorBtn = makeBtn('Color: off');
        colorBtn.addEventListener('click', function() {{
            colorMode = !colorMode;
            syncColorBtn();
            ensureBindings();  // defensive: keep listeners live
        }});

        var colorAllBtn = makeBtn('Color all');
        colorAllBtn.addEventListener('click', function() {{
            if (MANAGED_LIST.length === 0) return;
            var colors = MANAGED_LIST.map(function(idx) {{
                return PALETTE[idx % PALETTE.length];
            }});
            var opacities = MANAGED_LIST.map(function() {{ return COLOR_OPACITY; }});
            Plotly.restyle(gd, {{
                visible: true,
                'marker.color': colors,
                'marker.opacity': opacities
            }}, MANAGED_LIST.slice());
            colorMode = false;
            syncColorBtn();
            ensureBindings();
        }});

        resetBtn = makeBtn('Show all / reset');
        resetBtn.addEventListener('click', function() {{
            if (MANAGED_LIST.length === 0) return;
            Plotly.restyle(gd, {{
                visible: true,
                'marker.color': GRAY,
                'marker.opacity': GRAY_OPACITY
            }}, MANAGED_LIST.slice());
            colorMode = false;
            syncColorBtn();
            // Plotly's restyle can drop user listeners in some builds;
            // re-attach immediately so subsequent legend clicks work.
            ensureBindings();
        }});

        bar.appendChild(colorBtn);
        bar.appendChild(colorAllBtn);
        bar.appendChild(resetBtn);
        // Insertion fallback chain: parent.insertBefore is the clean path
        // (Jupyter, plain HTML). If for any reason that fails (parent has
        // been replaced, no parent, etc.), fall back to body.prepend.
        var inserted = false;
        if (gd.parentNode && typeof gd.parentNode.insertBefore === 'function') {{
            try {{ gd.parentNode.insertBefore(bar, gd); inserted = true; }}
            catch (e) {{ /* fall through */ }}
        }}
        if (!inserted) {{
            document.body.insertBefore(bar, document.body.firstChild);
        }}
    }} else {{
        // Toolbar already exists (e.g. cell re-rendered). Grab refs so
        // the colorMode toggle stays wired. Button order: Color, Color all, Reset.
        var btns = existingBar.querySelectorAll('button');
        if (btns.length >= 3) {{ colorBtn = btns[0]; resetBtn = btns[2]; }}
    }}

    // --- Legend click handler (active only in color mode) ---------------

    // Force the legend to redraw its swatches. Single-trace restyle updates
    // marker.color in the plot but plotly skips legend regeneration as an
    // optimisation; setting showlegend to its current value re-runs doLegend.
    function refreshLegend() {{
        var sl = (gd.layout && gd.layout.showlegend !== false);
        Plotly.relayout(gd, {{'showlegend': sl}});
    }}

    function legendClick(ev) {{
        var idx = ev.curveNumber;
        if (!isManaged(idx) || !colorMode) return true;  // plotly default
        var tr = gd.data && gd.data[idx];
        var painting = isGray(tr);
        var nextColor = painting ? PALETTE[idx % PALETTE.length] : GRAY;
        var nextOpacity = painting ? COLOR_OPACITY : GRAY_OPACITY;
        Plotly.restyle(gd, {{
            'marker.color': nextColor,
            'marker.opacity': nextOpacity
        }}, [idx]).then(refreshLegend);
        return false;  // suppress plotly's hide
    }}

    // Re-bind both the legend handler and the afterplot watchdog after
    // every event that could have wiped them. Uses removeListener with
    // the same function reference so we always have exactly one of each.
    function ensureBindings() {{
        try {{ gd.removeListener('plotly_legendclick', legendClick); }}
        catch (e) {{ /* ignore */ }}
        gd.on('plotly_legendclick', legendClick);
        try {{ gd.removeListener('plotly_afterplot', ensureBindings); }}
        catch (e) {{ /* ignore */ }}
        gd.on('plotly_afterplot', ensureBindings);
    }}
    ensureBindings();
}})();
"""


def _managed_indices(fig) -> list[int]:
    """Indices of traces tagged `meta.legend_kind="gray"`.

    Computed Python-side once at display/write_html time so the JS no
    longer has to read `tr.meta` at runtime (plotly can drop meta
    during restyles in some builds).
    """
    out = []
    for i, tr in enumerate(fig.data):
        meta = getattr(tr, "meta", None)
        if meta and dict(meta).get("legend_kind") == "gray":
            out.append(i)
    return out


def _save_png_config(fig, *, scale: float = 3.0) -> dict:
    """Plotly config for the modebar PNG-export button.

    Forces `scale` (≈ dpi multiplier — 3 ≈ 300 dpi) and sizes the export
    canvas to match the figure's layout dimensions so the full legend is
    inside the saved PNG rather than clipped.
    """
    width = int(fig.layout.width) if fig.layout.width else 820
    height = int(fig.layout.height) if fig.layout.height else 460
    return {
        "toImageButtonOptions": {
            "format": "png",
            "scale": scale,
            "width": width,
            "height": height,
        },
    }


def _attach_save_hooks(fig):
    """Wrap fig.write_html so plain `fig.write_html(path)` still carries the
    Color toolbar JS + 3x PNG save config. Jupyter rendering (to_html /
    _repr_mimebundle_) is untouched.
    """
    orig_write_html = fig.write_html

    def patched_write_html(*args, **kwargs):
        kwargs.setdefault("post_script", _legend_click_js(_managed_indices(fig)))
        kwargs.setdefault("config", _save_png_config(fig))
        return orig_write_html(*args, **kwargs)

    fig.write_html = patched_write_html
    return fig


def write_html(
    fig,
    path: str | PathLike[str],
    *,
    include_plotlyjs: bool | str = True,
    save_scale: float = 3.0,
) -> None:
    """Write `fig` to an HTML file with the click-to-highlight JS embedded.

    The resulting file is self-contained and reproduces the same Color
    toolbar UX you get in a notebook via `interactive.display(fig)`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str or PathLike
        Destination .html file.
    include_plotlyjs : bool or 'cdn'
        Forwarded to plotly. True embeds plotly.js (offline-friendly but big);
        'cdn' uses a script tag (smaller file, needs internet).
    save_scale : float
        Scale for the modebar PNG-export button (3.0 ≈ 300 dpi).
    """
    js = _legend_click_js(_managed_indices(fig))
    fig.write_html(
        str(path),
        post_script=js,
        include_plotlyjs=include_plotlyjs,
        config=_save_png_config(fig, scale=save_scale),
    )


def display(fig, *, save_scale: float = 3.0) -> None:
    """Render `fig` inline in a Jupyter cell with the Color toolbar JS.

    Use this instead of letting Jupyter render `fig` directly when you
    want the gray↔colour interaction. Returns nothing — the figure is
    shown as a side effect.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    save_scale : float
        Scale for the modebar PNG-export button (3.0 ≈ 300 dpi).
    """
    try:
        from IPython.display import HTML
        from IPython.display import display as _ipy_display
    except ImportError as e:
        raise ImportError(
            "IPython is required for ehtim.plotting.interactive.display(). Install with `pip install ipython`."
        ) from e
    js = _legend_click_js(_managed_indices(fig))
    html = fig.to_html(
        post_script=js,
        include_plotlyjs="cdn",
        config=_save_png_config(fig, scale=save_scale),
    )
    _ipy_display(HTML(html))


# --- plot_bl --------------------------------------------------------------


def _extract_baseline_data(obs, site1, site2, field, sigtype, *, snrcut, ang_unit, debias, timetype):
    """Return (times, values, errors) for a baseline after SNR filtering, or None."""
    plotdata = obs.unpack_bl(site1, site2, field, ang_unit=ang_unit, debias=debias, timetype=timetype)
    errdata = obs.unpack_bl(site1, site2, sigtype, ang_unit=ang_unit, debias=debias) if sigtype else None

    mask = ~np.isnan(plotdata[field][:, 0])
    if snrcut > 0 and errdata is not None:
        if field in ehc.FIELDS_AMPS:
            mask &= plotdata[field][:, 0] / errdata[sigtype][:, 0] > snrcut
        elif field in ehc.FIELDS_PHASE:
            mask &= errdata[sigtype][:, 0] < (180.0 / np.pi / snrcut)
        elif field in ehc.FIELDS_SNRS:
            mask &= plotdata[field][:, 0] > snrcut

    if not mask.any():
        return None

    plotdata = plotdata[mask]
    if errdata is not None:
        errdata = errdata[mask]

    times = plotdata["time"][:, 0]
    values = plotdata[field][:, 0]
    errors = errdata[sigtype][:, 0] if errdata is not None else None
    return times, values, errors


def _make_baseline_trace(site1, site2, times, values, errors, *, field, timetype, value_unit_suffix):
    customdata = np.column_stack(
        [
            np.full(len(times), site1, dtype=object),
            np.full(len(times), site2, dtype=object),
            times,
            values,
            errors if errors is not None else np.full(len(times), np.nan),
        ]
    )

    error_y = (
        dict(type="data", array=errors, visible=True, thickness=1, width=2, color=_THEME["error_color"])
        if errors is not None
        else dict(visible=False)
    )

    err_line = "<br>Error: %{customdata[4]:.4g}" if errors is not None else ""

    return go.Scatter(
        x=times,
        y=values,
        mode="markers",
        name=f"{site1}-{site2}",
        marker=dict(size=7, opacity=0.85, line=dict(width=0.5, color=_THEME["marker_edge"])),
        error_y=error_y,
        customdata=customdata,
        hovertemplate=(
            f"<b>{site1}–{site2}</b><br>"
            f"{timetype}: %{{x:.3f}} hr<br>"
            f"{field}: %{{y:.4g}}{value_unit_suffix}"
            f"{err_line}<extra></extra>"
        ),
    )


def plot_bl(
    obs: Obsdata,
    site1: str | None = None,
    site2: str | None = None,
    field: str = "amp",
    *,
    debias: bool = False,
    ang_unit: str = "deg",
    timetype: str | None = None,
    snrcut: float = 0.0,
    rangex: tuple[float, float] | None = None,
    rangey: tuple[float, float] | None = None,
    show: bool = True,
) -> Any:
    """Interactive baseline time-series plot.

    Plotly counterpart of `Obsdata.plot_bl` (eht-imaging/ehtim/obsdata.py:4239).
    Two modes:

    - **Single baseline:** pass `site1` and `site2`. One trace, no legend.
    - **All baselines:** pass neither. One trace per baseline.
      Single-click a legend entry to colour it; double-click to colour all;
      use the "Show all / reset" button to restore the uniform gray state.

    `u`/`v`/`uvdist` fields are auto-scaled to Gλ or Mλ based on the max
    baseline length.

    Each point carries (site1, site2, time, value, error) as plotly
    `customdata`, surfaced on hover.
    """
    field = field.lower()
    if field not in ehc.FIELDS:
        raise ValueError(f"Unknown field {field!r}; valid fields: {ehc.FIELDS}")

    if timetype is None:
        timetype = obs.timetype
    sigtype = obsh.sigtype(field)

    # Resolve which baselines to plot.
    if site1 is None and site2 is None:
        pairs = sorted({tuple(sorted((str(a), str(b)))) for a, b in zip(obs.data["t1"], obs.data["t2"])})
        title_suffix = "all baselines"
        show_legend = True
    elif site1 is not None and site2 is not None:
        known = set(obs.tarr["site"])
        missing = [s for s in (site1, site2) if s not in known]
        if missing:
            raise ValueError(f"site(s) {missing} not in obs.tarr; available sites: {sorted(known)}")
        pairs = [(site1, site2)]
        title_suffix = f"{site1}–{site2}"
        show_legend = False
    else:
        raise ValueError("Provide both site1 and site2 for a single baseline, or neither for all baselines.")

    # First pass: extract data so we can pick a single uv-unit across baselines.
    extracted: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for s1, s2 in pairs:
        ext = _extract_baseline_data(
            obs,
            s1,
            s2,
            field,
            sigtype,
            snrcut=snrcut,
            ang_unit=ang_unit,
            debias=debias,
            timetype=timetype,
        )
        if ext is None:
            continue
        times, values, errors = ext
        extracted.append((s1, s2, times, values, errors))

    y_unit, y_div = _axis_unit_global([e[3] for e in extracted], field)
    y_suffix = f" {y_unit}" if y_unit else ""

    fig = go.Figure()
    for s1, s2, times, values, errors in extracted:
        values_s = values / y_div
        errors_s = (errors / y_div) if errors is not None else None
        trace = _make_baseline_trace(
            s1,
            s2,
            times,
            values_s,
            errors_s,
            field=field,
            timetype=timetype,
            value_unit_suffix=y_suffix,
        )
        if len(pairs) > 1:
            # Uniform muted colour so the palette never runs out on large
            # arrays. Single-click a legend entry to paint one baseline,
            # double-click to paint all, or use the "Show all / reset"
            # button to restore the uniform gray state. The `meta` tag
            # marks this trace as participating in the gray↔colour flow,
            # and marker.opacity doubles as the state indicator the JS
            # reads (`< _OPACITY_THRESHOLD` → gray) — see `_legend_click_js`.
            trace.marker.color = _GRAY
            trace.marker.opacity = _GRAY_OPACITY
            trace.meta = dict(legend_kind="gray")
        fig.add_trace(trace)

    if not fig.data:
        print(f"No valid data after filtering (snrcut={snrcut}).")
        return fig

    y_title = _field_label(field)
    if field in ehc.FIELDS_PHASE:
        y_title += f" ({ang_unit})"
    elif y_unit:
        y_title += f" ({y_unit})"

    _apply_theme(
        fig,
        title=f"{obs.source}: {title_suffix} ({field})",
        xaxis_title=f"{timetype} (hr)",
        yaxis_title=y_title,
        rangex=rangex,
        rangey=rangey,
        legend_title="Baseline",
        show_legend=show_legend,
        add_reset_button=show_legend,
    )

    if show:
        fig.show()
    return _attach_save_hooks(fig)


# --- plotall --------------------------------------------------------------


def plotall(
    obs: Obsdata,
    field1: str,
    field2: str,
    *,
    conj: bool = False,
    debias: bool = False,
    tag_bl: bool = True,
    ang_unit: str = "deg",
    timetype: str | None = None,
    snrcut: float = 0.0,
    xscale: str = "linear",
    yscale: str = "linear",
    rangex: tuple[float, float] | None = None,
    rangey: tuple[float, float] | None = None,
    show: bool = True,
) -> Any:
    """Interactive scatter of two visibility fields against each other.

    Plotly counterpart of `Obsdata.plotall` (eht-imaging/ehtim/obsdata.py:4001).
    Two modes:

    - **tag_bl=True (default):** one trace per baseline. Single-click paints
      one baseline, double-click paints all, "Show all / reset" restores
      uniform gray.
    - **tag_bl=False:** all visibilities pooled into one trace.

    `u`/`v`/`uvdist` fields on either axis auto-scale to Gλ or Mλ.

    Common uses:

    - ``plotall(obs, 'u', 'v', conj=True)`` — uv coverage with conjugates
    - ``plotall(obs, 'uvdist', 'amp')`` — amp vs baseline length (radplot)
    """
    field1 = field1.lower()
    field2 = field2.lower()
    for f in (field1, field2):
        if f not in ehc.FIELDS:
            raise ValueError(f"Unknown field {f!r}; valid fields: {ehc.FIELDS}")

    if timetype is None:
        timetype = obs.timetype

    sigtype1 = obsh.sigtype(field1)
    sigtype2 = obsh.sigtype(field2)

    def _filter_mask(data, sigx, sigy):
        # `|` not `+` for boolean OR — `+` works on numpy bool arrays but
        # promotes to int and reads as arithmetic.
        m = ~(np.isnan(data[field1]) | np.isnan(data[field2]))
        if snrcut > 0:
            for fld, sig in [(field1, sigx), (field2, sigy)]:
                if fld in ehc.FIELDS_AMPS and sig is not None:
                    m &= data[fld] / sig > snrcut
                elif fld in ehc.FIELDS_PHASE and sig is not None:
                    m &= sig < (180.0 / np.pi / snrcut)
                elif fld in ehc.FIELDS_SNRS:
                    m &= data[fld] > snrcut
        return m

    fig = go.Figure()
    # We rescale uv-fields once we know the global max; collect raw values,
    # then scale every trace consistently after the loop.
    raw_traces: list[dict] = []

    if tag_bl:
        for bl in obs.bllist(conj=conj):
            t1 = str(bl["t1"][0])
            t2 = str(bl["t2"][0])
            data = obs.unpack_dat(bl, [field1, field2], ang_unit=ang_unit, debias=debias, timetype=timetype)
            sigx = obs.unpack_dat(bl, [sigtype1], ang_unit=ang_unit)[sigtype1] if sigtype1 else None
            sigy = obs.unpack_dat(bl, [sigtype2], ang_unit=ang_unit)[sigtype2] if sigtype2 else None

            mask = _filter_mask(data, sigx, sigy)
            if not mask.any():
                continue
            data = data[mask]
            if sigx is not None:
                sigx = sigx[mask]
            if sigy is not None:
                sigy = sigy[mask]

            raw_traces.append(
                dict(
                    t1=t1,
                    t2=t2,
                    x=data[field1],
                    y=data[field2],
                    sigx=sigx,
                    sigy=sigy,
                )
            )
    else:
        data = obs.unpack([field1, field2], conj=conj, ang_unit=ang_unit, debias=debias, timetype=timetype)
        sigx = obs.unpack(sigtype1, conj=conj, ang_unit=ang_unit)[sigtype1] if sigtype1 else None
        sigy = obs.unpack(sigtype2, conj=conj, ang_unit=ang_unit)[sigtype2] if sigtype2 else None

        mask = _filter_mask(data, sigx, sigy)
        data = data[mask]
        if sigx is not None:
            sigx = sigx[mask]
        if sigy is not None:
            sigy = sigy[mask]
        if len(data) > 0:
            raw_traces.append(
                dict(
                    t1=None,
                    t2=None,
                    x=data[field1],
                    y=data[field2],
                    sigx=sigx,
                    sigy=sigy,
                )
            )

    # Global scaling factor for each axis (uv fields only).
    x_unit, x_div = _axis_unit_global([t["x"] for t in raw_traces], field1)
    y_unit, y_div = _axis_unit_global([t["y"] for t in raw_traces], field2)

    # u-vs-v: pick a single symmetric range driven by max(|u|,|v|) so the
    # data is centred and both axes share the same span (square frame).
    if {field1, field2} <= {"u", "v"} and rangex is None and rangey is None and raw_traces:
        scaled_max = 0.0
        for t in raw_traces:
            if len(t["x"]):
                scaled_max = max(scaled_max, float(np.max(np.abs(t["x"] / x_div))))
            if len(t["y"]):
                scaled_max = max(scaled_max, float(np.max(np.abs(t["y"] / y_div))))
        if scaled_max > 0:
            pad = 0.05 * scaled_max
            rangex = (-scaled_max - pad, scaled_max + pad)
            rangey = (-scaled_max - pad, scaled_max + pad)

    for t in raw_traces:
        x = t["x"] / x_div
        y = t["y"] / y_div
        sigx = t["sigx"] / x_div if t["sigx"] is not None else None
        sigy = t["sigy"] / y_div if t["sigy"] is not None else None
        name = f"{t['t1']}-{t['t2']}" if t["t1"] is not None else "all baselines"

        if t["t1"] is not None:
            customdata = np.column_stack(
                [
                    np.full(len(x), t["t1"], dtype=object),
                    np.full(len(x), t["t2"], dtype=object),
                    sigx if sigx is not None else np.full(len(x), np.nan),
                    sigy if sigy is not None else np.full(len(x), np.nan),
                ]
            )
            hover_label = f"<b>{t['t1']}–{t['t2']}</b><br>"
        else:
            customdata = None
            hover_label = ""

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(
                    size=6,
                    opacity=_GRAY_OPACITY if tag_bl else 0.7,
                    color=_GRAY if tag_bl else None,
                    line=dict(width=0.5, color=_THEME["marker_edge"]),
                ),
                error_x=(
                    dict(type="data", array=sigx, visible=True, thickness=1, width=2, color=_THEME["error_color"])
                    if sigx is not None
                    else dict(visible=False)
                ),
                error_y=(
                    dict(type="data", array=sigy, visible=True, thickness=1, width=2, color=_THEME["error_color"])
                    if sigy is not None
                    else dict(visible=False)
                ),
                customdata=customdata,
                # Tag participates in the gray↔colour flow (see _legend_click_js).
                meta=dict(legend_kind="gray") if tag_bl else None,
                hovertemplate=(
                    f"{hover_label}"
                    f"{field1}: %{{x:.4g}}{(' ' + x_unit) if x_unit else ''}<br>"
                    f"{field2}: %{{y:.4g}}{(' ' + y_unit) if y_unit else ''}"
                    "<extra></extra>"
                ),
            )
        )

    if not fig.data:
        print(f"No valid data after filtering (snrcut={snrcut}).")
        return fig

    x_title = _field_label(field1)
    y_title = _field_label(field2)
    if field1 in ehc.FIELDS_PHASE:
        x_title += f" ({ang_unit})"
    elif x_unit:
        x_title += f" ({x_unit})"
    if field2 in ehc.FIELDS_PHASE:
        y_title += f" ({ang_unit})"
    elif y_unit:
        y_title += f" ({y_unit})"

    _apply_theme(
        fig,
        title=f"{obs.source}: {field1} vs {field2}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        rangex=rangex,
        rangey=rangey,
        legend_title="Baseline" if tag_bl else "",
        show_legend=tag_bl,
        add_reset_button=tag_bl,
    )

    if xscale == "log":
        fig.update_xaxes(type="log")
    if yscale == "log":
        fig.update_yaxes(type="log")

    # u-vs-v coverage plot: lock data:pixel ratio so equal-span renders square.
    # Also size the figure so the (square) plot area fills the available canvas
    # without leaving a wide blank strip between the plot and the legend.
    if {field1, field2} <= {"u", "v"} and xscale != "log" and yscale != "log":
        fig.update_xaxes(scaleanchor="y", scaleratio=1.0, constrain="domain")
        # Plot area = width - L - R = height - T - B   →   square plot.
        # _apply_theme uses margin l=70, r=160, t=80, b=60 (with reset button).
        side = 460  # target plot side in px
        fig.update_layout(width=side + 70 + 160, height=side + 80 + 60)

    if show:
        fig.show()
    return _attach_save_hooks(fig)


def _axis_unit_global(arrays: list[np.ndarray], field: str) -> tuple[str, float]:
    """Choose (unit_label, divisor) for a uv-field given many traces; else ('', 1)."""
    if field not in _UV_FIELDS or not arrays:
        return "", 1.0
    finite = np.concatenate([np.abs(np.asarray(a, dtype=float)) for a in arrays if len(a)])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "λ", 1.0
    vmax = float(np.max(finite))
    if vmax >= 1e9:
        return "Gλ", 1e9
    return "Mλ", 1e6


# --- plot_gains -----------------------------------------------------------


def plot_gains(
    caltable: Caltable,
    sites: list[str] | str = "all",
    gain_type: str = "amp",
    pol: str = "R",
    *,
    timetype: str | None = None,
    ang_unit: str = "deg",
    yscale: str = "log",
    rangex: tuple[float, float] | None = None,
    rangey: tuple[float, float] | None = None,
    show: bool = True,
) -> Any:
    """Interactive gains-vs-time plot.

    Plotly counterpart of `Caltable.plot_gains` (eht-imaging/ehtim/caltable.py:153).
    One trace per (site, pol) — click legend entries to toggle visibility.

    Parameters
    ----------
    caltable : Caltable
    sites : list[str] or 'all'
        'all' / empty list means every site in `caltable.data`.
    gain_type : {'amp', 'phase'}
    pol : {'R', 'L', 'both'}
        'both' draws R and L as separate traces per site.
        TODO: schema-coupled — replace with pol1/pol2 once the cal-table
        mixed-pol schema lands.
    timetype : {'GMST', 'UTC'} or None
        None means use `caltable.timetype`.
    ang_unit : {'deg', 'rad'}
        Phase unit when `gain_type='phase'`.
    yscale : {'log', 'lin'}
        Only honored for `gain_type='amp'`; phase plots are always linear.
    """
    if gain_type not in ("amp", "phase"):
        raise ValueError(f"gain_type must be 'amp' or 'phase', got {gain_type!r}")
    # TODO: schema-coupled — replace R/L with pol1/pol2 lookup after the
    # cal-table mixed-pol rewrite.
    if pol not in ("R", "L", "both"):
        raise ValueError(f"pol must be 'R', 'L', or 'both', got {pol!r}")

    if isinstance(sites, str):
        sites = sorted(caltable.data.keys()) if sites.lower() == "all" else [sites]
    elif len(sites) == 0:
        sites = sorted(caltable.data.keys())

    if timetype is None:
        timetype = caltable.timetype

    pol_channels = ("R", "L") if pol == "both" else (pol,)
    angle_div = (np.pi / 180.0) if ang_unit == "deg" else 1.0

    fig = go.Figure()
    for site in sites:
        times = caltable.data[site]["time"]
        if timetype.upper() == "UTC" and caltable.timetype.upper() == "GMST":
            times = obsh.gmst_to_utc(times, caltable.mjd)
        elif timetype.upper() == "GMST" and caltable.timetype.upper() == "UTC":
            times = obsh.utc_to_gmst(times, caltable.mjd)

        for j, pol_ch in enumerate(pol_channels):
            # TODO: schema-coupled — replace with caltable.data[site][pol_ch]
            # once the table keys by configurable basis names instead of R/L.
            key = "rscale" if pol_ch == "R" else "lscale"
            gains_complex = caltable.data[site][key]

            if gain_type == "amp":
                gains = np.abs(gains_complex)
            else:
                gains = np.angle(gains_complex) / angle_div

            customdata = np.column_stack(
                [
                    np.full(len(times), site, dtype=object),
                    np.full(len(times), pol_ch, dtype=object),
                    gains_complex.real,
                    gains_complex.imag,
                ]
            )

            trace_name = f"{site} ({pol_ch})" if pol == "both" else site
            symbol = _SYMBOLS[j % len(_SYMBOLS)] if pol == "both" else "circle"

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=gains,
                    mode="markers",
                    name=trace_name,
                    marker=dict(size=7, opacity=0.85, symbol=symbol, line=dict(width=0.5, color=_THEME["marker_edge"])),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                        f"{timetype}: %{{x:.3f}} hr<br>"
                        f"{gain_type}: %{{y:.4g}}<br>"
                        "Re/Im: %{customdata[2]:.3g} + %{customdata[3]:.3g}i"
                        "<extra></extra>"
                    ),
                )
            )

    if gain_type == "amp":
        y_title = "|G|"
        y_type = "log" if yscale == "log" else "linear"
    else:
        y_title = f"arg(G) ({ang_unit})"
        y_type = "linear"

    _apply_theme(
        fig,
        title=f"Caltable gains — {gain_type}, pol={pol}",
        xaxis_title=f"{timetype} (hr)",
        yaxis_title=y_title,
        rangex=rangex,
        rangey=rangey,
        y_type=y_type,
        legend_title="Site" if pol != "both" else "Site (pol)",
    )

    if show:
        fig.show()
    return _attach_save_hooks(fig)


# --- dashboard ------------------------------------------------------------

# Data products available in the dashboard's panel-2 selector. Each one is
# (data, model) overlay; the dropdown switches which pair is visible and
# updates the (x, y) axis titles in place.
_DASH_PRODUCT_ORDER = (
    "amp_vs_uvdist",
    "vis_vs_uvdist",
    "phase_vs_uvdist",
    "chisq_vs_uvdist",
    "cphase_vs_time",
    "logcamp_vs_time",
    "cphase_vs_triarea",
    "logcamp_vs_quadarea",
)

_DASH_PRODUCT_LABELS = {
    "amp_vs_uvdist": "Amplitude vs uv-distance",
    "vis_vs_uvdist": "Re(vis) vs uv-distance",
    "phase_vs_uvdist": "Phase vs uv-distance",
    "chisq_vs_uvdist": "χ residual vs uv-distance",
    "cphase_vs_time": "Closure phase vs time",
    "logcamp_vs_time": "Log closure amp vs time",
    "cphase_vs_triarea": "Closure phase vs triangle area",
    "logcamp_vs_quadarea": "Log closure amp vs quadrangle area",
}


def _enumerate_triangles(obs: Obsdata, filter_: tuple | None, limit: int | None) -> list[tuple[str, str, str]]:
    """Return triangles sorted by sample count, optionally filtered/limited.

    `filter_=(s1, s2, s3)` → return only that triangle (if it has samples).
    `limit=None` → all triangles; otherwise the top-`limit` by sample count.
    """
    try:
        cph = obs.c_phases(mode="all", count="max")
    except Exception:
        return []
    if len(cph) == 0:
        return []
    counts: dict[tuple[str, str, str], int] = {}
    for row in cph:
        tri = tuple(sorted((str(row["t1"]), str(row["t2"]), str(row["t3"]))))
        counts[tri] = counts.get(tri, 0) + 1
    if filter_ is not None:
        key = tuple(sorted(str(s) for s in filter_))
        return [key] if key in counts else []
    sorted_tris = sorted(counts, key=lambda k: counts[k], reverse=True)
    return sorted_tris[:limit] if limit else sorted_tris


def _enumerate_quads(obs: Obsdata, filter_: tuple | None, limit: int | None) -> list[tuple[str, str, str, str]]:
    """Return quadrangles sorted by sample count, optionally filtered/limited."""
    try:
        camps = obs.c_amplitudes(mode="all", count="max", ctype="logcamp")
    except Exception:
        return []
    if len(camps) == 0:
        return []
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in camps:
        quad = tuple(sorted((str(row["t1"]), str(row["t2"]), str(row["t3"]), str(row["t4"]))))
        counts[quad] = counts.get(quad, 0) + 1
    if filter_ is not None:
        key = tuple(sorted(str(s) for s in filter_))
        return [key] if key in counts else []
    sorted_quads = sorted(counts, key=lambda k: counts[k], reverse=True)
    return sorted_quads[:limit] if limit else sorted_quads


def _pick_default_triangle(obs: Obsdata) -> tuple[str, str, str] | None:
    """Triangle with the most cphase samples; None if none exist."""
    tris = _enumerate_triangles(obs, None, 1)
    return tris[0] if tris else None


def _pick_default_quad(obs: Obsdata) -> tuple[str, str, str, str] | None:
    """Quadrangle with the most logcamp samples; None if none exist."""
    qs = _enumerate_quads(obs, None, 1)
    return qs[0] if qs else None


def _empty_spec(label: str = "") -> dict:
    return {
        "label": label,
        "x_data": np.array([]),
        "y_data": np.array([]),
        "x_model": np.array([]),
        "y_model": np.array([]),
    }


def _build_dashboard_products(
    obs: Obsdata,
    obs_model: Obsdata | None,
    *,
    triangle: tuple[str, str, str] | None,
    quadrangle: tuple[str, str, str, str] | None,
    n_triangles: int | None,
    n_quadrangles: int | None,
):
    """Build per-product trace specs for the dashboard panel-2 dropdown.

    Returns a dict keyed by `_DASH_PRODUCT_ORDER`; each value has:
        ``x_title``, ``y_title``       — panel axis labels for that product
        ``style``                       — "single" (gray data + red model) or
                                          "multi" (per-trace palette colour)
        ``traces``                      — list of trace specs (≥ 1), each a
                                          dict with ``label``, ``x_data``,
                                          ``y_data``, ``x_model``, ``y_model``

    Closure-quantity products (cphase / logcamp) enumerate all triangles
    (or quadrangles) and emit one trace per closure path. Use the
    ``triangle`` / ``quadrangle`` kwargs to filter to a specific one,
    or set ``n_triangles`` / ``n_quadrangles`` to cap the count.
    """
    products: dict[str, dict[str, Any]] = {}

    # --- uv-distance products (one spec each) ------------------------------
    udata = obs.unpack(["uvdist", "amp", "vis", "phase"])
    uvdist_scaled, uv_unit = _format_uv_axis(udata["uvdist"])
    if obs_model is not None:
        mdata = obs_model.unpack(["uvdist", "amp", "vis", "phase"])
        # Same scaling factor as data — don't let the model rescale itself.
        div = 1e9 if uv_unit == "Gλ" else 1e6
        uvdist_m_scaled = mdata["uvdist"] / div
    else:
        mdata = None
        uvdist_m_scaled = np.array([])

    def _single(x_data, y_data, x_model, y_model, x_title, y_title):
        return dict(
            x_title=x_title,
            y_title=y_title,
            style="single",
            traces=[dict(label="data", x_data=x_data, y_data=y_data, x_model=x_model, y_model=y_model)],
        )

    products["amp_vs_uvdist"] = _single(
        uvdist_scaled,
        udata["amp"],
        uvdist_m_scaled,
        (mdata["amp"] if mdata is not None else np.array([])),
        f"uv-distance ({uv_unit})",
        "Amplitude (Jy)",
    )
    products["vis_vs_uvdist"] = _single(
        uvdist_scaled,
        np.real(udata["vis"]),
        uvdist_m_scaled,
        (np.real(mdata["vis"]) if mdata is not None else np.array([])),
        f"uv-distance ({uv_unit})",
        "Re(vis) (Jy)",
    )
    products["phase_vs_uvdist"] = _single(
        uvdist_scaled,
        udata["phase"],
        uvdist_m_scaled,
        (mdata["phase"] if mdata is not None else np.array([])),
        f"uv-distance ({uv_unit})",
        "Phase (deg)",
    )

    if mdata is not None and len(udata["amp"]) == len(mdata["amp"]):
        sigma = obs.unpack(["sigma"])["sigma"]
        sigma_safe = np.where(sigma > 0, sigma, np.nan)
        residual = (udata["amp"] - mdata["amp"]) / sigma_safe
        products["chisq_vs_uvdist"] = _single(
            uvdist_scaled,
            residual,
            np.array([]),
            np.array([]),
            f"uv-distance ({uv_unit})",
            "(data − model) / σ",
        )
    else:
        products["chisq_vs_uvdist"] = _single(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            f"uv-distance ({uv_unit})",
            "(data − model) / σ",
        )

    # --- closure-phase products (one spec per triangle) --------------------
    triangles = _enumerate_triangles(obs, triangle, n_triangles)

    cph_time_specs: list[dict] = []
    cph_area_specs: list[dict] = []
    all_tri_areas: list[np.ndarray] = []

    # First pass: pull raw cphase data + raw areas (un-scaled) so we can
    # pick a single area unit across all triangles.
    cph_buf: list[tuple[tuple[str, str, str], Any, Any, np.ndarray]] = []
    for tri in triangles:
        try:
            cph_data = obs.cphase_tri(*tri)
        except Exception:
            continue
        if len(cph_data) == 0:
            continue
        try:
            cph_model = obs_model.cphase_tri(*tri) if obs_model is not None else None
        except Exception:
            cph_model = None
        if cph_model is not None and len(cph_model) == 0:
            cph_model = None
        tri_area = obsh.uv_area_triangle(cph_data["u1"], cph_data["v1"], cph_data["u2"], cph_data["v2"])
        cph_buf.append((tri, cph_data, cph_model, tri_area))
        all_tri_areas.append(tri_area)

    if cph_buf:
        # Pick a global area unit (Gλ² vs Mλ²) by linearizing to baseline.
        flat_area = np.concatenate(all_tri_areas) if all_tri_areas else np.array([])
        _, tri_unit = _format_uv_axis(np.sqrt(flat_area))
        area_div = (1e9**2) if tri_unit == "Gλ" else (1e6**2)
        for tri, cph_data, cph_model, tri_area in cph_buf:
            tri_str = "–".join(tri)
            x_model_t = cph_model["time"] if cph_model is not None else np.array([])
            y_model_t = cph_model["cphase"] if cph_model is not None else np.array([])
            cph_time_specs.append(
                dict(
                    label=tri_str,
                    x_data=cph_data["time"],
                    y_data=cph_data["cphase"],
                    x_model=x_model_t,
                    y_model=y_model_t,
                )
            )
            tri_area_disp = tri_area / area_div
            tri_area_model = np.array([])
            if cph_model is not None and len(cph_model) > 0:
                tri_area_model = (
                    obsh.uv_area_triangle(cph_model["u1"], cph_model["v1"], cph_model["u2"], cph_model["v2"]) / area_div
                )
            cph_area_specs.append(
                dict(
                    label=tri_str,
                    x_data=tri_area_disp,
                    y_data=cph_data["cphase"],
                    x_model=tri_area_model,
                    y_model=y_model_t,
                )
            )
        x_title_area = f"triangle uv-area ({tri_unit}²)"
    else:
        x_title_area = "triangle uv-area (Gλ²)"

    products["cphase_vs_time"] = dict(
        x_title="time (hr)",
        y_title="Closure phase (deg)",
        style="multi",
        traces=cph_time_specs or [_empty_spec()],
    )
    products["cphase_vs_triarea"] = dict(
        x_title=x_title_area,
        y_title="Closure phase (deg)",
        style="multi",
        traces=cph_area_specs or [_empty_spec()],
    )

    # --- log closure-amplitude products (one spec per quadrangle) ----------
    quads = _enumerate_quads(obs, quadrangle, n_quadrangles)

    camp_time_specs: list[dict] = []
    camp_area_specs: list[dict] = []
    camp_buf: list[tuple[tuple[str, str, str, str], Any, Any, np.ndarray]] = []
    for quad in quads:
        try:
            camp_data = obs.camp_quad(*quad, ctype="logcamp")
        except Exception:
            continue
        if len(camp_data) == 0:
            continue
        try:
            camp_model = obs_model.camp_quad(*quad, ctype="logcamp") if obs_model is not None else None
        except Exception:
            camp_model = None
        if camp_model is not None and len(camp_model) == 0:
            camp_model = None
        quad_area = obsh.uv_area_quadrangle(
            camp_data["u1"], camp_data["v1"], camp_data["u2"], camp_data["v2"], camp_data["u3"], camp_data["v3"]
        )
        camp_buf.append((quad, camp_data, camp_model, quad_area))

    if camp_buf:
        flat_qarea = np.concatenate([q[3] for q in camp_buf])
        _, quad_unit = _format_uv_axis(np.sqrt(flat_qarea))
        qarea_div = (1e9**2) if quad_unit == "Gλ" else (1e6**2)
        for quad, camp_data, camp_model, quad_area in camp_buf:
            quad_str = "–".join(quad)
            x_model_t = camp_model["time"] if camp_model is not None else np.array([])
            y_model_t = camp_model["camp"] if camp_model is not None else np.array([])
            camp_time_specs.append(
                dict(
                    label=quad_str,
                    x_data=camp_data["time"],
                    y_data=camp_data["camp"],
                    x_model=x_model_t,
                    y_model=y_model_t,
                )
            )
            quad_area_disp = quad_area / qarea_div
            quad_area_model = np.array([])
            if camp_model is not None and len(camp_model) > 0:
                quad_area_model = (
                    obsh.uv_area_quadrangle(
                        camp_model["u1"],
                        camp_model["v1"],
                        camp_model["u2"],
                        camp_model["v2"],
                        camp_model["u3"],
                        camp_model["v3"],
                    )
                    / qarea_div
                )
            camp_area_specs.append(
                dict(
                    label=quad_str,
                    x_data=quad_area_disp,
                    y_data=camp_data["camp"],
                    x_model=quad_area_model,
                    y_model=y_model_t,
                )
            )
        qarea_title = f"quadrangle uv-area ({quad_unit}²)"
    else:
        qarea_title = "quadrangle uv-area (Gλ²)"

    products["logcamp_vs_time"] = dict(
        x_title="time (hr)",
        y_title="Log closure amplitude",
        style="multi",
        traces=camp_time_specs or [_empty_spec()],
    )
    products["logcamp_vs_quadarea"] = dict(
        x_title=qarea_title,
        y_title="Log closure amplitude",
        style="multi",
        traces=camp_area_specs or [_empty_spec()],
    )

    return products


def _pol_ticks_traces(im: Image, *, nvec: int, pcut: float, mcut: float,
                      colour_by_m: bool, m_bins: int = 12,
                      coord_scale: float = 1.0,
                      coord_origin: tuple[float, float] = (0.0, 0.0),
                      colorbar_kwargs: dict | None = None) -> list:
    """Return EVPA tick traces overlaying the Stokes-I heatmap.

    Tick length ∝ |P|; tick color encodes |m| (rainbow) when
    colour_by_m=True. `coord_scale`/`coord_origin` map pixel indices to
    plot coords — pass (psize_uas, (x0, y0)) for μas axes, else defaults
    keep pixel coords.
    """
    try:
        imvec = np.asarray(im.imvec, dtype=float).copy()
        qvec = np.asarray(im.qvec, dtype=float)
        uvec = np.asarray(im.uvec, dtype=float)
    except (AttributeError, KeyError):
        return []
    if qvec.size == 0 or uvec.size == 0 or (np.all(qvec == 0) and np.all(uvec == 0)):
        return []

    ydim, xdim = im.ydim, im.xdim
    thin = max(1, xdim // nvec)
    safe_I = np.where(imvec > 0, imvec, np.nan)
    p = np.abs(qvec + 1j * uvec)
    m = p / safe_I

    I2 = imvec.reshape(ydim, xdim)
    P2 = p.reshape(ydim, xdim)
    M2 = m.reshape(ydim, xdim)

    Imax = np.nanmax(I2)
    mask = (I2 > pcut * Imax) & (M2 > mcut) & np.isfinite(M2)
    sub = mask[::thin, ::thin]
    if not sub.any():
        return []

    ys, xs = np.mgrid[:ydim:thin, :xdim:thin]
    x_centers = xs[sub].astype(float)
    y_centers = ys[sub].astype(float)
    angle = np.angle(qvec + 1j * uvec) / 2.0
    angle2 = angle.reshape(ydim, xdim)[::thin, ::thin][sub]
    # Tick direction = (-sin θ, cos θ) per Image.display convention.
    dx = -np.sin(angle2)
    dy = np.cos(angle2)

    p_sub = P2[::thin, ::thin][sub]
    p_max = np.nanmax(p_sub) if np.nanmax(p_sub) > 0 else 1.0
    L = thin * 0.70 * (p_sub / p_max)

    xs_a = x_centers - 0.5 * L * dx
    xs_b = x_centers + 0.5 * L * dx
    ys_a = y_centers - 0.5 * L * dy
    ys_b = y_centers + 0.5 * L * dy

    # Pixel -> plot coords (heatmap axis units).
    x0, y0 = coord_origin
    xs_a = x0 + xs_a * coord_scale
    xs_b = x0 + xs_b * coord_scale
    ys_a = y0 + ys_a * coord_scale
    ys_b = y0 + ys_b * coord_scale

    if not colour_by_m:
        # Single white-line trace (no m colorbar).
        seg_x = np.empty(3 * len(x_centers))
        seg_y = np.empty(3 * len(x_centers))
        seg_x[0::3] = xs_a
        seg_x[1::3] = xs_b
        seg_x[2::3] = np.nan
        seg_y[0::3] = ys_a
        seg_y[1::3] = ys_b
        seg_y[2::3] = np.nan
        return [go.Scatter(
            x=seg_x.tolist(), y=seg_y.tolist(),
            mode="lines",
            line=dict(color="white", width=2.5),
            hoverinfo="skip", showlegend=False, name="EVPA",
        )]

    # Per-bin colored line traces: discretise m, one trace per non-empty bin.
    m_sub = M2[::thin, ::thin][sub]
    m_min = 0.0
    m_max = float(max(np.nanmax(m_sub), 0.01))
    edges = np.linspace(m_min, m_max, m_bins + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    span = max(m_max - m_min, 1e-12)
    colors = _plotly_colors.sample_colorscale("Rainbow", (midpoints - m_min) / span)

    traces: list = []
    for i in range(m_bins):
        if i < m_bins - 1:
            in_bin = (m_sub >= edges[i]) & (m_sub < edges[i + 1])
        else:
            in_bin = (m_sub >= edges[i]) & (m_sub <= edges[i + 1])
        if not in_bin.any():
            continue
        n = int(in_bin.sum())
        seg_x = np.empty(3 * n)
        seg_y = np.empty(3 * n)
        seg_x[0::3] = xs_a[in_bin]
        seg_x[1::3] = xs_b[in_bin]
        seg_x[2::3] = np.nan
        seg_y[0::3] = ys_a[in_bin]
        seg_y[1::3] = ys_b[in_bin]
        seg_y[2::3] = np.nan
        traces.append(go.Scatter(
            x=seg_x.tolist(), y=seg_y.tolist(),
            mode="lines",
            line=dict(color=colors[i], width=2.5),
            hoverinfo="skip", showlegend=False,
            name=f"EVPA m={midpoints[i]:.2f}",
        ))

    # Invisible scatter that owns the shared rainbow colorbar.
    cbar = colorbar_kwargs or dict(
        title=dict(text="|m|"), x=0.40, y=0.78, yanchor="middle",
        len=0.32, thickness=10,
    )
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            size=0,
            color=[m_min, m_max],
            colorscale="Rainbow",
            cmin=m_min, cmax=m_max,
            colorbar=cbar,
            showscale=True,
        ),
        hoverinfo="skip", showlegend=False, name="|m|",
    ))
    return traces


def dashboard(
    im: Image,
    obs: Obsdata,
    caltable: Caltable,
    *,
    pol: str = "R",
    show_model: bool = True,
    ttype: str = "direct",
    plotp: bool = True,
    nvec: int = 20,
    pcut: float = 0.1,
    mcut: float = 0.01,
    vec_cfun: bool = True,
    triangle: tuple[str, str, str] | None = None,
    quadrangle: tuple[str, str, str, str] | None = None,
    n_triangles: int | None = 12,
    n_quadrangles: int | None = 12,
    default_product: str = "amp_vs_uvdist",
    show: bool = True,
) -> Any:
    """Reconstruction dashboard — image + data fit + gains + D-terms.

    2x2 layout: inspect a full reconstruction's model + data + calibration
    + leakage at a glance.

    Parameters
    ----------
    im : Image
        Reconstructed (or model) image. Stokes-I heatmap is always shown;
        with `plotp=True`, EVPA ticks are overlaid (mirroring
        `Image.display(plotp=True)`).
    obs : Obsdata
        Observation. The panel-2 data product is computed from `obs`; if
        `show_model=True`, the corresponding model from
        `im.observe_same_nonoise(obs, ttype=ttype)` is overlaid.
    caltable : Caltable
        Calibration table; provides per-site gains and D-terms (from `tarr`).
    pol : {'R', 'L'}
        Which gain channel to plot. TODO: schema-coupled — replace with
        pol1/pol2 once the cal-table mixed-pol schema lands.
    show_model : bool
        Whether to compute and overlay model traces in panel 2.
    ttype : {'direct', 'fast', 'nfft'}
        Transform type for the model-visibility computation.
    plotp : bool
        Overlay EVPA polarization ticks on the Stokes-I image and show the
        colorbar (Jy/px). Requires `im` to carry Q/U.
    nvec : int
        Approximate number of tick samples across the image (default 20).
    pcut, mcut : float
        Tick gating: keep pixels where I > pcut · Imax AND |P|/I > mcut.
    vec_cfun : bool
        Colour ticks by |m| via a small marker overlay with a side colorbar.
    triangle : (s1, s2, s3) or None
        If provided, the cphase products show *only* this triangle.
        None (default) shows all triangles overlaid, each in a palette
        colour, capped at `n_triangles`.
    quadrangle : (s1, s2, s3, s4) or None
        Same semantics for the logcamp products.
    n_triangles, n_quadrangles : int or None
        Cap on the number of distinct closure paths shown when
        `triangle` / `quadrangle` are None. None = unlimited.
        Default 12 keeps the legend readable on typical EHT arrays.
    default_product : str
        Initially-visible data product in panel 2. One of
        `interactive._DASH_PRODUCT_ORDER`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if pol not in ("R", "L"):
        raise ValueError(f"pol must be 'R' or 'L', got {pol!r}")
    if default_product not in _DASH_PRODUCT_ORDER:
        raise ValueError(f"default_product must be one of {_DASH_PRODUCT_ORDER}, got {default_product!r}")

    obs_model = im.observe_same_nonoise(obs, ttype=ttype) if show_model else None

    products = _build_dashboard_products(
        obs,
        obs_model,
        triangle=triangle,
        quadrangle=quadrangle,
        n_triangles=n_triangles,
        n_quadrangles=n_quadrangles,
    )

    source_name = str(getattr(im, "source", "") or getattr(obs, "source", "") or "")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            source_name or "Image",
            _DASH_PRODUCT_LABELS[default_product],
            f"Gains (|G|, pol={pol})",
            "D-terms (complex plane)",
        ),
        # Narrower left column so the (square) image fills its cell; column
        # 2 picks up the slack and gives the panel + D-terms more room.
        # horizontal_spacing kept generous so the panel-2 y-axis labels +
        # title don't crowd against the Stokes-I heatmap edge / colorbar.
        column_widths=[0.40, 0.60],
        horizontal_spacing=0.20,
        vertical_spacing=0.14,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]],
    )

    # --- Panel 1: image ---
    # RA/Dec offsets in μas, like Image.display().
    img2d = im.imvec.reshape(im.ydim, im.xdim)
    psize_uas = im.psize / ehc.RADPERUAS
    x_uas = (np.arange(im.xdim) - (im.xdim - 1) / 2.0) * psize_uas
    y_uas = (np.arange(im.ydim) - (im.ydim - 1) / 2.0) * psize_uas
    # Black backdrop covering the panel-1 cell so the "hot" heatmap reads
    # against black rather than the figure's gray plot_bgcolor (visible as
    # strips when scaleanchor shrinks the plot domain).
    # Panel-1 paper bounds derived from make_subplots config above.
    _p1_x = (0.0, 0.40 * (1.0 - 0.20))
    _p1_y = (1.0 - (1.0 - 0.14) / 2.0, 1.0)
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=_p1_x[0], x1=_p1_x[1], y0=_p1_y[0], y1=_p1_y[1],
        fillcolor="black", line=dict(width=0), layer="below",
    )
    fig.add_trace(go.Heatmap(
        z=img2d,
        x=x_uas,
        y=y_uas,
        colorscale="hot",
        hovertemplate="RA=%{x:.1f} μas<br>Dec=%{y:.1f} μas<br>I=%{z:.3g}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Jy/px"),
            x=0.34, y=0.785, yanchor="middle",
            len=0.43, thickness=10,
            exponentformat="power",
            showexponent="all",
        ),
    ), row=1, col=1)

    # Build pol traces unconditionally (when im has Q/U) so the toggle
    # button can hide/show them at runtime.
    pol_traces = _pol_ticks_traces(
        im, nvec=nvec, pcut=pcut, mcut=mcut, colour_by_m=vec_cfun,
        coord_scale=psize_uas,
        coord_origin=(-(im.xdim - 1) / 2.0 * psize_uas, -(im.ydim - 1) / 2.0 * psize_uas),
        colorbar_kwargs=dict(
            title=dict(text="|m|"),
            x=0.41, y=0.785, yanchor="middle",
            len=0.43, thickness=10,
        ),
    )
    pol_trace_indices: list[int] = []
    for trace in pol_traces:
        trace.visible = bool(plotp)
        fig.add_trace(trace, row=1, col=1)
        pol_trace_indices.append(len(fig.data) - 1)

    # Panel 2: data product selector
    # Each product owns 1+ trace pairs; closure products enumerate all
    # triangles/quadrangles so the user sees every closure quantity, not just one
    # Only the `default_product`'s traces are visible at start; the
    # dropdown switches which group is shown.
    panel2_indices: dict[str, list[int]] = {key: [] for key in _DASH_PRODUCT_ORDER}
    panel2_start = len(fig.data)
    for key in _DASH_PRODUCT_ORDER:
        p = products[key]
        is_default = key == default_product
        style = p.get("style", "multi")
        for i, spec in enumerate(p["traces"]):
            if style == "single":
                data_color = _GRAY
                data_opacity = 0.6
                model_color = _THEME["colorway"][1]
                data_name = "data"
                model_name = "model"
                model_symbol = "circle"
                show_model_in_legend = True
            else:
                data_color = _THEME["colorway"][i % len(_THEME["colorway"])]
                data_opacity = 0.85
                model_color = data_color
                data_name = spec["label"]
                model_name = f"{spec['label']} (model)"
                model_symbol = "x"
                # Share a legendgroup so toggling one legend entry hides
                # both the data and model trace for that triangle/quad.
                show_model_in_legend = False
            legendgroup = f"{key}_{i}"

            fig.add_trace(
                go.Scatter(
                    x=spec["x_data"],
                    y=spec["y_data"],
                    mode="markers",
                    name=data_name,
                    marker=dict(
                        size=5,
                        color=data_color,
                        opacity=data_opacity,
                        line=dict(width=0.3, color=_THEME["marker_edge"]),
                    ),
                    hovertemplate=("<b>" + data_name + "</b><br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"),
                    legend="legend2",
                    legendgroup=legendgroup,
                    visible=is_default,
                ),
                row=1,
                col=2,
            )
            panel2_indices[key].append(len(fig.data) - 1)

            x_model = spec.get("x_model")
            if x_model is not None and len(x_model) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=spec["x_model"],
                        y=spec["y_model"],
                        mode="markers",
                        name=model_name,
                        marker=dict(
                            size=5,
                            color=model_color,
                            opacity=0.55,
                            symbol=model_symbol,
                            line=dict(width=0.3, color=_THEME["marker_edge"]),
                        ),
                        hovertemplate=("<b>" + model_name + "</b><br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"),
                        legend="legend2",
                        legendgroup=legendgroup,
                        showlegend=show_model_in_legend,
                        visible=is_default,
                    ),
                    row=1,
                    col=2,
                )
                panel2_indices[key].append(len(fig.data) - 1)
    panel2_end = len(fig.data)

    # --- Panel 3: gains per site (own legend) ---
    # TODO: schema-coupled — replace 'rscale'/'lscale' once mixed-pol lands.
    pol_key = "rscale" if pol == "R" else "lscale"
    for site in sorted(caltable.data.keys()):
        gain = np.abs(caltable.data[site][pol_key])
        fig.add_trace(
            go.Scatter(
                x=caltable.data[site]["time"],
                y=gain,
                mode="markers",
                name=site,
                marker=dict(size=5, line=dict(width=0.3, color=_THEME["marker_edge"])),
                hovertemplate=(f"<b>{site}</b><br>t=%{{x:.2f}} hr<br>|G|=%{{y:.3g}}<extra></extra>"),
                legend="legend3",
            ),
            row=2,
            col=1,
        )

    # --- Panel 4: D-terms (R and L) in complex plane (own legend) ---
    # TODO: schema-coupled — tarr['dr'] / tarr['dl'] move to caltable.dterms
    # in MixPol Phase 1; this lookup needs to follow.
    tarr = caltable.tarr
    sites = [str(s) for s in tarr["site"]]
    fig.add_trace(
        go.Scatter(
            x=np.real(tarr["dr"]),
            y=np.imag(tarr["dr"]),
            mode="markers+text",
            text=sites,
            textposition="top center",
            textfont=dict(size=9, color=_THEME["font_color"]),
            name="D_R",
            marker=dict(size=10, symbol="circle", line=dict(width=0.5, color=_THEME["marker_edge"])),
            hovertemplate=("<b>%{text}</b><br>Re(D_R)=%{x:.3g}<br>Im(D_R)=%{y:.3g}<extra></extra>"),
            legend="legend4",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(tarr["dl"]),
            y=np.imag(tarr["dl"]),
            mode="markers+text",
            text=sites,
            textposition="bottom center",
            textfont=dict(size=9, color=_THEME["font_color"]),
            name="D_L",
            marker=dict(size=10, symbol="square", line=dict(width=0.5, color=_THEME["marker_edge"])),
            hovertemplate=("<b>%{text}</b><br>Re(D_L)=%{x:.3g}<br>Im(D_L)=%{y:.3g}<extra></extra>"),
            legend="legend4",
        ),
        row=2,
        col=2,
    )

    # --- Layout (multi-legend: one per scatter panel) ---
    # Each panel's legend sits just OUTSIDE the panel (in the gap between
    # cols, or in the right margin) rather than inside, so it doesn't
    # occlude data. Right margin is generous to give the panel-1,2 and
    # panel-2,2 legends room.
    fig.update_layout(
        template="none",
        plot_bgcolor=_THEME["plot_bgcolor"],
        paper_bgcolor=_THEME["paper_bgcolor"],
        font=dict(family=_THEME["font_family"], size=_THEME["font_size"], color=_THEME["font_color"]),
        width=1280,
        height=1000,
        margin=dict(l=70, r=200, t=80, b=60),
        showlegend=True,
        colorway=_THEME["colorway"],
        legend2=dict(  # data/model (panel 1,2 = top-right): just outside right edge
            x=1.01, y=0.99,
            xanchor="left", yanchor="top",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
        ),
        legend3=dict(  # gains (panel 2,1 = bottom-left): in the col-gap
            x=0.33, y=0.43,
            xanchor="left", yanchor="top",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
        ),
        legend4=dict(  # D-terms (panel 2,2 = bottom-right): just outside right edge
            x=1.01, y=0.43,
            xanchor="left", yanchor="top",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
        ),
    )

    # Common axis styling per subplot.
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(
            gridcolor=_THEME["grid_color"],
            linecolor=_THEME["edge_color"],
            tickcolor=_THEME["edge_color"],
            showline=True,
            mirror=True,
            row=r,
            col=c,
        )
        fig.update_yaxes(
            gridcolor=_THEME["grid_color"],
            linecolor=_THEME["edge_color"],
            tickcolor=_THEME["edge_color"],
            showline=True,
            mirror=True,
            row=r,
            col=c,
        )

    # Per-panel axis labels and aspect.
    # Panel 1: RA increases left, Dec increases up. Heatmap y is in pixel
    # order top-down already, so flip via autorange='reversed' to put
    # Dec increasing upward.
    fig.update_xaxes(title_text="RA offset (μas)", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text="Dec offset (μas)", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text=products[default_product]["x_title"], row=1, col=2)
    fig.update_yaxes(title_text=products[default_product]["y_title"], row=1, col=2)
    fig.update_xaxes(title_text="time (hr)", row=2, col=1)
    fig.update_yaxes(title_text="|G|", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Re(D)", row=2, col=2, zeroline=True, zerolinecolor=_THEME["edge_color"])
    fig.update_yaxes(
        title_text="Im(D)",
        row=2,
        col=2,
        scaleanchor="x4",
        scaleratio=1,
        zeroline=True,
        zerolinecolor=_THEME["edge_color"],
    )

    # --- Data-product dropdown for panel 2 ---
    # Each product owns an arbitrary number of traces (1 for the uvdist
    # products; N for the closure products, one per triangle/quad). The
    # dropdown sets visibility per index from the precomputed group map.
    buttons = []
    for key in _DASH_PRODUCT_ORDER:
        p = products[key]
        selected = set(panel2_indices[key])
        vis = []
        for i in range(len(fig.data)):
            if i < panel2_start or i >= panel2_end:
                # Outside panel 2 (image + ticks + gains + D-terms): leave on.
                vis.append(True)
            else:
                vis.append(i in selected)
        buttons.append(
            dict(
                label=_DASH_PRODUCT_LABELS[key],
                method="update",
                args=[
                    {"visible": vis},
                    {"xaxis2.title.text": p["x_title"], "yaxis2.title.text": p["y_title"]},
                ],
            )
        )

    updatemenus_list = [
        dict(
            type="dropdown",
            direction="down",
            buttons=buttons,
            showactive=True,
            # Top-left corner — the title is centered (x=0.5), so anchor
            # the dropdown to the far left to keep them separated.
            x=0.0,
            xanchor="left",
            y=1.06,
            yanchor="bottom",
            pad=dict(l=4, r=4, t=2, b=2),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
            font=dict(size=10, color=_THEME["font_color"]),
        ),
    ]
    # Pol toggle: hides/shows the EVPA ticks + |m| colorbar overlay. Default
    # state matches the `plotp` parameter; click flips between args/args2.
    if pol_trace_indices:
        updatemenus_list.append(
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.20, xanchor="left",
                y=1.06, yanchor="bottom",
                pad=dict(l=4, r=4, t=2, b=2),
                bgcolor="rgba(238,238,238,0.85)",
                bordercolor=_THEME["edge_color"],
                borderwidth=1,
                font=dict(size=10, color=_THEME["font_color"]),
                buttons=[dict(
                    label="Toggle polarization",
                    method="restyle",
                    args=[{"visible": not bool(plotp)}, pol_trace_indices],
                    args2=[{"visible": bool(plotp)}, pol_trace_indices],
                )],
            )
        )
    fig.update_layout(updatemenus=updatemenus_list)

    if show:
        fig.show()
    return _attach_save_hooks(fig)
