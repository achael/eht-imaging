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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError(
        "Plotly is required for ehtim.plotting.interactive. "
        "Install with `pip install plotly`."
    ) from e

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
    "font_family": ('Inter, system-ui, -apple-system, BlinkMacSystemFont, '
                    '"Segoe UI", Roboto, sans-serif'),
    "font_size": 12,
    "title_size": 16,
    # Matplotlib 'bmh' (Bayesian Methods for Hackers) palette + extensions.
    "colorway": [
        "#348ABD", "#A60628", "#7A68A6", "#4B8022",
        "#D55E00", "#CC79A7", "#1F471F", "#2C825D",
    ],
    "plot_bgcolor": "#eeeeee",
    "paper_bgcolor": "#ffffff",
    "grid_color":   "#ffffff",
    "zero_color":   "#ffffff",
    "edge_color":   "#bcbcbc",
    "font_color":   "#333333",
    "marker_edge":  "rgba(0,0,0,0.4)",
    "error_color":  "rgba(0,0,0,0.3)",
}

# Fallback symbols when a plot has more traces than the colorway has colors.
_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"]

# Initial colour for traces that participate in the gray↔colour legend flow.
_GRAY = "#888888"

# Fields whose values are in raw λ and should auto-scale to Gλ/Mλ.
_UV_FIELDS = frozenset({"u", "v", "uvdist"})


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


def _apply_theme(fig, *, title, xaxis_title, yaxis_title,
                 rangex=None, rangey=None, y_type="linear",
                 legend_title="", show_legend=True,
                 add_reset_button=False, n_traces=0):
    """Apply the BMH visual theme to a plotly Figure.

    `add_reset_button=True` installs a "Show all / reset" button that
    restores visibility for every trace and paints them all gray —
    pairs with the click-to-colour JS in `_legend_click_js`.
    """
    layout = dict(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=_THEME["title_size"],
                             family=_THEME["font_family"],
                             color=_THEME["font_color"])),
        xaxis=dict(title=xaxis_title, range=rangex,
                   gridcolor=_THEME["grid_color"],
                   zerolinecolor=_THEME["zero_color"],
                   zerolinewidth=1.5,
                   linecolor=_THEME["edge_color"],
                   linewidth=1,
                   ticks="inside",
                   tickcolor=_THEME["edge_color"],
                   showline=True,
                   mirror=True),
        yaxis=dict(title=yaxis_title, range=rangey, type=y_type,
                   gridcolor=_THEME["grid_color"],
                   zerolinecolor=_THEME["zero_color"],
                   zerolinewidth=1.5,
                   linecolor=_THEME["edge_color"],
                   linewidth=1,
                   ticks="inside",
                   tickcolor=_THEME["edge_color"],
                   showline=True,
                   mirror=True),
        template="none",
        plot_bgcolor=_THEME["plot_bgcolor"],
        paper_bgcolor=_THEME["paper_bgcolor"],
        font=dict(family=_THEME["font_family"],
                  size=_THEME["font_size"],
                  color=_THEME["font_color"]),
        margin=dict(l=70, r=160, t=80 if add_reset_button else 60, b=60),
        width=_THEME["width"], height=_THEME["height"],
        hovermode="closest",
        colorway=_THEME["colorway"],
        showlegend=show_legend,
        legend=dict(
            title=dict(text=legend_title),
            orientation="v",
            x=1.02, y=1, xanchor="left", yanchor="top",
            font=dict(size=11, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
            itemsizing="constant",
        ),
    )
    if add_reset_button and n_traces > 0:
        layout["updatemenus"] = [_reset_button_menu(n_traces)]
    fig.update_layout(**layout)


def _reset_button_menu(n_traces: int) -> dict:
    """Build the 'Show all / reset' updatemenu spec for `n_traces` traces."""
    return dict(
        type="buttons",
        direction="left",
        buttons=[
            dict(
                label="Show all / reset",
                method="restyle",
                args=[{"visible": True,
                       "marker.color": [_GRAY] * n_traces}],
            ),
        ],
        showactive=False,
        x=0.0, xanchor="left",
        y=1.10, yanchor="bottom",
        pad=dict(l=4, r=4, t=2, b=2),
        bgcolor="rgba(238,238,238,0.85)",
        bordercolor=_THEME["edge_color"],
        borderwidth=1,
        font=dict(size=10, color=_THEME["font_color"]),
    )


# --- Click-to-highlight JS ------------------------------------------------

# Legend interaction (replaces the old tri-state cycle):
#   single click  → gray ↔ colour for a trace that started gray,
#                   colour → hidden ↔ colour for already-coloured traces
#   double click  → paint *all* traces from the palette
#   "Show all / reset" button (top-left) → reset everything to gray + visible.
#
# `{plot_id}` is the placeholder plotly substitutes when this is passed as
# `post_script` to `to_html` / `write_html`.


def _legend_click_js() -> str:
    palette = json.dumps(_THEME["colorway"])
    gray = json.dumps(_GRAY)
    return f"""
(function() {{
    var gd = document.getElementById('{{plot_id}}');
    if (!gd) return;
    var PALETTE = {palette};
    var GRAY = {gray};
    gd.on('plotly_legendclick', function(ev) {{
        var idx = ev.curveNumber;
        var tr = gd.data[idx];
        var base = (tr.marker && tr.marker.color) || '';
        var hidden = (tr.visible === 'legendonly');
        if (hidden) {{
            Plotly.restyle(gd, {{visible: true,
                                'marker.color': PALETTE[idx % PALETTE.length]}}, [idx]);
            return false;
        }}
        if (base === GRAY) {{
            Plotly.restyle(gd, {{'marker.color': PALETTE[idx % PALETTE.length]}}, [idx]);
            return false;
        }}
        return true;  // already-coloured + visible → plotly default toggle (hide)
    }});
    gd.on('plotly_legenddoubleclick', function(ev) {{
        var n = gd.data.length;
        var colors = [];
        for (var i = 0; i < n; i++) colors.push(PALETTE[i % PALETTE.length]);
        Plotly.restyle(gd, {{visible: true, 'marker.color': colors}});
        return false;
    }});
}})();
"""


def write_html(fig, path: str | PathLike[str], *,
               include_plotlyjs: bool | str = True) -> None:
    """Write `fig` to an HTML file with the click-to-highlight JS embedded.

    The resulting file is self-contained and reproduces the same legend-click
    UX you get in a notebook via `interactive.display(fig)`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    path : str or PathLike
        Destination .html file.
    include_plotlyjs : bool or 'cdn'
        Forwarded to plotly. True embeds plotly.js (offline-friendly but big);
        'cdn' uses a script tag (smaller file, needs internet).
    """
    fig.write_html(str(path), post_script=_legend_click_js(),
                   include_plotlyjs=include_plotlyjs)


def display(fig) -> None:
    """Render `fig` inline in a Jupyter cell with click-to-highlight JS.

    Use this instead of letting Jupyter render `fig` directly when you want
    the legend single/double-click highlight UX.
    """
    try:
        from IPython.display import HTML
        from IPython.display import display as _ipy_display
    except ImportError as e:
        raise ImportError(
            "IPython is required for ehtim.plotting.interactive.display(). "
            "Install with `pip install ipython`."
        ) from e
    html = fig.to_html(post_script=_legend_click_js(), include_plotlyjs="cdn")
    _ipy_display(HTML(html))


# --- plot_bl --------------------------------------------------------------

def _extract_baseline_data(obs, site1, site2, field, sigtype, *,
                           snrcut, ang_unit, debias, timetype):
    """Return (times, values, errors) for a baseline after SNR filtering, or None."""
    plotdata = obs.unpack_bl(site1, site2, field, ang_unit=ang_unit,
                             debias=debias, timetype=timetype)
    errdata = (obs.unpack_bl(site1, site2, sigtype,
                             ang_unit=ang_unit, debias=debias)
               if sigtype else None)

    mask = ~np.isnan(plotdata[field][:, 0])
    if snrcut > 0 and errdata is not None:
        if field in ehc.FIELDS_AMPS:
            mask &= (plotdata[field][:, 0] / errdata[sigtype][:, 0] > snrcut)
        elif field in ehc.FIELDS_PHASE:
            mask &= (errdata[sigtype][:, 0] < (180.0 / np.pi / snrcut))
        elif field in ehc.FIELDS_SNRS:
            mask &= (plotdata[field][:, 0] > snrcut)

    if not mask.any():
        return None

    plotdata = plotdata[mask]
    if errdata is not None:
        errdata = errdata[mask]

    times = plotdata["time"][:, 0]
    values = plotdata[field][:, 0]
    errors = errdata[sigtype][:, 0] if errdata is not None else None
    return times, values, errors


def _make_baseline_trace(site1, site2, times, values, errors, *,
                         field, timetype, value_unit_suffix):
    customdata = np.column_stack([
        np.full(len(times), site1, dtype=object),
        np.full(len(times), site2, dtype=object),
        times, values,
        errors if errors is not None else np.full(len(times), np.nan),
    ])

    error_y = (dict(type="data", array=errors, visible=True,
                    thickness=1, width=2, color=_THEME["error_color"])
               if errors is not None else dict(visible=False))

    err_line = ("<br>Error: %{customdata[4]:.4g}"
                if errors is not None else "")

    return go.Scatter(
        x=times, y=values, mode="markers",
        name=f"{site1}-{site2}",
        marker=dict(size=7, opacity=0.85,
                    line=dict(width=0.5, color=_THEME["marker_edge"])),
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
        pairs = sorted({tuple(sorted((str(a), str(b))))
                        for a, b in zip(obs.data["t1"], obs.data["t2"])})
        title_suffix = "all baselines"
        show_legend = True
    elif site1 is not None and site2 is not None:
        known = set(obs.tarr["site"])
        missing = [s for s in (site1, site2) if s not in known]
        if missing:
            raise ValueError(
                f"site(s) {missing} not in obs.tarr; "
                f"available sites: {sorted(known)}"
            )
        pairs = [(site1, site2)]
        title_suffix = f"{site1}–{site2}"
        show_legend = False
    else:
        raise ValueError(
            "Provide both site1 and site2 for a single baseline, "
            "or neither for all baselines."
        )

    # First pass: extract data so we can pick a single uv-unit across baselines.
    extracted: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for s1, s2 in pairs:
        ext = _extract_baseline_data(
            obs, s1, s2, field, sigtype,
            snrcut=snrcut, ang_unit=ang_unit, debias=debias, timetype=timetype,
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
            s1, s2, times, values_s, errors_s,
            field=field, timetype=timetype, value_unit_suffix=y_suffix,
        )
        if len(pairs) > 1:
            # Uniform muted colour so the palette never runs out on large
            # arrays. Single-click a legend entry to paint one baseline,
            # double-click to paint all, or use the "Show all / reset"
            # button to restore the uniform gray state.
            trace.marker.color = _GRAY
            trace.marker.opacity = 0.6
        fig.add_trace(trace)

    if not fig.data:
        print(f"No valid data after filtering (snrcut={snrcut}).")
        return fig

    y_title = ehc.FIELD_LABELS.get(field, field.capitalize())
    if field in ehc.FIELDS_PHASE:
        y_title += f" ({ang_unit})"
    elif y_unit:
        y_title += f" ({y_unit})"

    _apply_theme(
        fig,
        title=f"{obs.source}: {title_suffix} ({field})",
        xaxis_title=f"{timetype} (hr)",
        yaxis_title=y_title,
        rangex=rangex, rangey=rangey,
        legend_title="Baseline",
        show_legend=show_legend,
        add_reset_button=show_legend,
        n_traces=len(fig.data),
    )

    if show:
        fig.show()
    return fig


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
            data = obs.unpack_dat(bl, [field1, field2],
                                  ang_unit=ang_unit, debias=debias,
                                  timetype=timetype)
            sigx = (obs.unpack_dat(bl, [sigtype1], ang_unit=ang_unit)[sigtype1]
                    if sigtype1 else None)
            sigy = (obs.unpack_dat(bl, [sigtype2], ang_unit=ang_unit)[sigtype2]
                    if sigtype2 else None)

            mask = _filter_mask(data, sigx, sigy)
            if not mask.any():
                continue
            data = data[mask]
            if sigx is not None:
                sigx = sigx[mask]
            if sigy is not None:
                sigy = sigy[mask]

            raw_traces.append(dict(
                t1=t1, t2=t2,
                x=data[field1], y=data[field2],
                sigx=sigx, sigy=sigy,
            ))
    else:
        data = obs.unpack([field1, field2], conj=conj,
                          ang_unit=ang_unit, debias=debias, timetype=timetype)
        sigx = (obs.unpack(sigtype1, conj=conj, ang_unit=ang_unit)[sigtype1]
                if sigtype1 else None)
        sigy = (obs.unpack(sigtype2, conj=conj, ang_unit=ang_unit)[sigtype2]
                if sigtype2 else None)

        mask = _filter_mask(data, sigx, sigy)
        data = data[mask]
        if sigx is not None:
            sigx = sigx[mask]
        if sigy is not None:
            sigy = sigy[mask]
        if len(data) > 0:
            raw_traces.append(dict(
                t1=None, t2=None,
                x=data[field1], y=data[field2],
                sigx=sigx, sigy=sigy,
            ))

    # Global scaling factor for each axis (uv fields only).
    x_unit, x_div = _axis_unit_global([t["x"] for t in raw_traces], field1)
    y_unit, y_div = _axis_unit_global([t["y"] for t in raw_traces], field2)

    for t in raw_traces:
        x = t["x"] / x_div
        y = t["y"] / y_div
        sigx = t["sigx"] / x_div if t["sigx"] is not None else None
        sigy = t["sigy"] / y_div if t["sigy"] is not None else None
        name = f"{t['t1']}-{t['t2']}" if t["t1"] is not None else "all baselines"

        if t["t1"] is not None:
            customdata = np.column_stack([
                np.full(len(x), t["t1"], dtype=object),
                np.full(len(x), t["t2"], dtype=object),
                sigx if sigx is not None else np.full(len(x), np.nan),
                sigy if sigy is not None else np.full(len(x), np.nan),
            ])
            hover_label = f"<b>{t['t1']}–{t['t2']}</b><br>"
        else:
            customdata = None
            hover_label = ""

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=name,
            marker=dict(size=6,
                        opacity=0.6 if tag_bl else 0.7,
                        color=_GRAY if tag_bl else None,
                        line=dict(width=0.5, color=_THEME["marker_edge"])),
            error_x=(dict(type="data", array=sigx, visible=True,
                          thickness=1, width=2, color=_THEME["error_color"])
                     if sigx is not None else dict(visible=False)),
            error_y=(dict(type="data", array=sigy, visible=True,
                          thickness=1, width=2, color=_THEME["error_color"])
                     if sigy is not None else dict(visible=False)),
            customdata=customdata,
            hovertemplate=(
                f"{hover_label}"
                f"{field1}: %{{x:.4g}}{(' ' + x_unit) if x_unit else ''}<br>"
                f"{field2}: %{{y:.4g}}{(' ' + y_unit) if y_unit else ''}"
                "<extra></extra>"
            ),
        ))

    if not fig.data:
        print(f"No valid data after filtering (snrcut={snrcut}).")
        return fig

    x_title = ehc.FIELD_LABELS.get(field1, field1.capitalize())
    y_title = ehc.FIELD_LABELS.get(field2, field2.capitalize())
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
        rangex=rangex, rangey=rangey,
        legend_title="Baseline" if tag_bl else "",
        show_legend=tag_bl,
        add_reset_button=tag_bl,
        n_traces=len(fig.data),
    )

    if xscale == "log":
        fig.update_xaxes(type="log")
    if yscale == "log":
        fig.update_yaxes(type="log")

    if show:
        fig.show()
    return fig


def _axis_unit_global(arrays: list[np.ndarray], field: str) -> tuple[str, float]:
    """Choose (unit_label, divisor) for a uv-field given many traces; else ('', 1)."""
    if field not in _UV_FIELDS or not arrays:
        return "", 1.0
    finite = np.concatenate([np.abs(np.asarray(a, dtype=float))
                             for a in arrays if len(a)])
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
        sites = (sorted(caltable.data.keys())
                 if sites.lower() == "all" else [sites])
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

            customdata = np.column_stack([
                np.full(len(times), site, dtype=object),
                np.full(len(times), pol_ch, dtype=object),
                gains_complex.real,
                gains_complex.imag,
            ])

            trace_name = f"{site} ({pol_ch})" if pol == "both" else site
            symbol = (_SYMBOLS[j % len(_SYMBOLS)]
                      if pol == "both" else "circle")

            fig.add_trace(go.Scatter(
                x=times, y=gains, mode="markers",
                name=trace_name,
                marker=dict(size=7, opacity=0.85, symbol=symbol,
                            line=dict(width=0.5, color=_THEME["marker_edge"])),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                    f"{timetype}: %{{x:.3f}} hr<br>"
                    f"{gain_type}: %{{y:.4g}}<br>"
                    "Re/Im: %{customdata[2]:.3g} + %{customdata[3]:.3g}i"
                    "<extra></extra>"
                ),
            ))

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
        rangex=rangex, rangey=rangey, y_type=y_type,
        legend_title="Site" if pol != "both" else "Site (pol)",
    )

    if show:
        fig.show()
    return fig


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
    "amp_vs_uvdist":       "Amplitude vs uv-distance",
    "vis_vs_uvdist":       "Re(vis) vs uv-distance",
    "phase_vs_uvdist":     "Phase vs uv-distance",
    "chisq_vs_uvdist":     "χ residual vs uv-distance",
    "cphase_vs_time":      "Closure phase vs time",
    "logcamp_vs_time":     "Log closure amp vs time",
    "cphase_vs_triarea":   "Closure phase vs triangle area",
    "logcamp_vs_quadarea": "Log closure amp vs quadrangle area",
}


def _pick_default_triangle(obs: Obsdata) -> tuple[str, str, str] | None:
    """Return the triangle with the most cphase samples; None if none exist."""
    try:
        cph = obs.c_phases(mode="all", count="max")
    except Exception:
        return None
    if len(cph) == 0:
        return None
    keys: dict[tuple[str, str, str], int] = {}
    for row in cph:
        tri = tuple(sorted((str(row["t1"]), str(row["t2"]), str(row["t3"]))))
        keys[tri] = keys.get(tri, 0) + 1
    best = max(keys, key=keys.get)
    return best


def _pick_default_quad(obs: Obsdata) -> tuple[str, str, str, str] | None:
    try:
        camps = obs.c_amplitudes(mode="all", count="max", ctype="logcamp")
    except Exception:
        return None
    if len(camps) == 0:
        return None
    keys: dict[tuple[str, str, str, str], int] = {}
    for row in camps:
        quad = tuple(sorted((str(row["t1"]), str(row["t2"]),
                             str(row["t3"]), str(row["t4"]))))
        keys[quad] = keys.get(quad, 0) + 1
    best = max(keys, key=keys.get)
    return best


def _build_dashboard_products(
    obs: Obsdata,
    obs_model: Obsdata | None,
    *,
    triangle: tuple[str, str, str] | None,
    quadrangle: tuple[str, str, str, str] | None,
):
    """Compute (x_data, y_data, x_model, y_model, x_title, y_title, label) for
    each product. Returns a list keyed by `_DASH_PRODUCT_ORDER`. Products with
    no data (e.g. cphase when no triangle has data) are returned with empty
    arrays — they still get traces, just nothing to plot."""
    products: dict[str, dict[str, Any]] = {}

    # uv-distance unit shared between vis/amp/phase/chisq panels.
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

    products["amp_vs_uvdist"] = dict(
        x_data=uvdist_scaled, y_data=udata["amp"],
        x_model=uvdist_m_scaled, y_model=(mdata["amp"] if mdata is not None else np.array([])),
        x_title=f"uv-distance ({uv_unit})", y_title="Amplitude (Jy)",
    )
    products["vis_vs_uvdist"] = dict(
        x_data=uvdist_scaled, y_data=np.real(udata["vis"]),
        x_model=uvdist_m_scaled,
        y_model=(np.real(mdata["vis"]) if mdata is not None else np.array([])),
        x_title=f"uv-distance ({uv_unit})", y_title="Re(vis) (Jy)",
    )
    products["phase_vs_uvdist"] = dict(
        x_data=uvdist_scaled, y_data=udata["phase"],
        x_model=uvdist_m_scaled,
        y_model=(mdata["phase"] if mdata is not None else np.array([])),
        x_title=f"uv-distance ({uv_unit})", y_title="Phase (deg)",
    )

    # χ residual = (data − model) / sigma_amp. Requires model.
    if mdata is not None and len(udata["amp"]) == len(mdata["amp"]):
        sigma = obs.unpack(["sigma"])["sigma"]
        sigma_safe = np.where(sigma > 0, sigma, np.nan)
        residual = (udata["amp"] - mdata["amp"]) / sigma_safe
        products["chisq_vs_uvdist"] = dict(
            x_data=uvdist_scaled, y_data=residual,
            x_model=np.array([]), y_model=np.array([]),
            x_title=f"uv-distance ({uv_unit})", y_title="(data − model) / σ",
        )
    else:
        products["chisq_vs_uvdist"] = dict(
            x_data=np.array([]), y_data=np.array([]),
            x_model=np.array([]), y_model=np.array([]),
            x_title=f"uv-distance ({uv_unit})", y_title="(data − model) / σ",
        )

    # Closure phase vs time + triangle area.
    if triangle is not None:
        try:
            cph_data = obs.cphase_tri(*triangle)
        except Exception:
            cph_data = None
        if obs_model is not None and cph_data is not None:
            try:
                cph_model = obs_model.cphase_tri(*triangle)
            except Exception:
                cph_model = None
        else:
            cph_model = None
    else:
        cph_data = None
        cph_model = None

    if cph_data is not None and len(cph_data) > 0:
        tri_str = "–".join(triangle) if triangle else ""
        products["cphase_vs_time"] = dict(
            x_data=cph_data["time"], y_data=cph_data["cphase"],
            x_model=(cph_model["time"] if cph_model is not None else np.array([])),
            y_model=(cph_model["cphase"] if cph_model is not None else np.array([])),
            x_title=f"time (hr) — triangle {tri_str}", y_title="Closure phase (deg)",
        )
        tri_area = obsh.uv_area_triangle(
            cph_data["u1"], cph_data["v1"], cph_data["u2"], cph_data["v2"])
        _, tri_unit = _format_uv_axis(np.sqrt(tri_area))
        # Display the area itself (squared), with units derived from the
        # one-baseline auto-scale to keep numbers readable.
        area_div = (1e9 ** 2) if tri_unit == "Gλ" else (1e6 ** 2)
        tri_area_disp = tri_area / area_div
        tri_area_model = np.array([])
        if cph_model is not None and len(cph_model) > 0:
            tri_area_model = obsh.uv_area_triangle(
                cph_model["u1"], cph_model["v1"],
                cph_model["u2"], cph_model["v2"]) / area_div
        products["cphase_vs_triarea"] = dict(
            x_data=tri_area_disp, y_data=cph_data["cphase"],
            x_model=tri_area_model,
            y_model=(cph_model["cphase"] if cph_model is not None else np.array([])),
            x_title=f"triangle uv-area ({tri_unit}²)",
            y_title="Closure phase (deg)",
        )
    else:
        # Empty placeholders so the dropdown can still switch to them.
        for key in ("cphase_vs_time", "cphase_vs_triarea"):
            products[key] = dict(
                x_data=np.array([]), y_data=np.array([]),
                x_model=np.array([]), y_model=np.array([]),
                x_title=("time (hr)" if "time" in key
                         else "triangle uv-area (Gλ²)"),
                y_title="Closure phase (deg)",
            )

    # Log closure amplitude vs time + quad area.
    if quadrangle is not None:
        try:
            camp_data = obs.camp_quad(*quadrangle, ctype="logcamp")
        except Exception:
            camp_data = None
        if obs_model is not None and camp_data is not None:
            try:
                camp_model = obs_model.camp_quad(*quadrangle, ctype="logcamp")
            except Exception:
                camp_model = None
        else:
            camp_model = None
    else:
        camp_data = None
        camp_model = None

    if camp_data is not None and len(camp_data) > 0:
        quad_str = "–".join(quadrangle) if quadrangle else ""
        products["logcamp_vs_time"] = dict(
            x_data=camp_data["time"], y_data=camp_data["camp"],
            x_model=(camp_model["time"] if camp_model is not None else np.array([])),
            y_model=(camp_model["camp"] if camp_model is not None else np.array([])),
            x_title=f"time (hr) — quad {quad_str}", y_title="Log closure amplitude",
        )
        quad_area = obsh.uv_area_quadrangle(
            camp_data["u1"], camp_data["v1"],
            camp_data["u2"], camp_data["v2"],
            camp_data["u3"], camp_data["v3"])
        _, quad_unit = _format_uv_axis(np.sqrt(quad_area))
        area_div = (1e9 ** 2) if quad_unit == "Gλ" else (1e6 ** 2)
        quad_area_disp = quad_area / area_div
        quad_area_model = np.array([])
        if camp_model is not None and len(camp_model) > 0:
            quad_area_model = obsh.uv_area_quadrangle(
                camp_model["u1"], camp_model["v1"],
                camp_model["u2"], camp_model["v2"],
                camp_model["u3"], camp_model["v3"]) / area_div
        products["logcamp_vs_quadarea"] = dict(
            x_data=quad_area_disp, y_data=camp_data["camp"],
            x_model=quad_area_model,
            y_model=(camp_model["camp"] if camp_model is not None else np.array([])),
            x_title=f"quadrangle uv-area ({quad_unit}²)",
            y_title="Log closure amplitude",
        )
    else:
        for key in ("logcamp_vs_time", "logcamp_vs_quadarea"):
            products[key] = dict(
                x_data=np.array([]), y_data=np.array([]),
                x_model=np.array([]), y_model=np.array([]),
                x_title=("time (hr)" if "time" in key
                         else "quadrangle uv-area (Gλ²)"),
                y_title="Log closure amplitude",
            )

    return products


def _pol_ticks_traces(im: Image, *, nvec: int, pcut: float, mcut: float,
                      colour_by_m: bool):
    """Build (lines_trace, dots_trace) for EVPA tick overlay; either may be None.

    Mirrors the per-pixel sampling, gating, and angle convention from
    `Image.display` (eht-imaging/ehtim/image.py:3820–3860). The dots_trace
    is only built when `colour_by_m=True`, and surfaces |m| via a colorbar.
    """
    try:
        imvec = np.asarray(im.imvec, dtype=float).copy()
        qvec = np.asarray(im.qvec, dtype=float)
        uvec = np.asarray(im.uvec, dtype=float)
    except (AttributeError, KeyError):
        return None, None
    if qvec.size == 0 or uvec.size == 0 or not np.any(qvec) and not np.any(uvec):
        return None, None

    ydim, xdim = im.ydim, im.xdim
    thin = max(1, xdim // nvec)
    # Avoid divide-by-zero in m gating.
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
        return None, None

    # Pixel-coordinate grids matching Plotly's heatmap (z[i, j] → x=j, y=i).
    ys, xs = np.mgrid[:ydim:thin, :xdim:thin]
    x_centers = xs[sub].astype(float)
    y_centers = ys[sub].astype(float)
    angle = np.angle(qvec + 1j * uvec) / 2.0
    angle2 = angle.reshape(ydim, xdim)[::thin, ::thin][sub]
    # Convention from Image.display: tick direction = (-sin θ, cos θ).
    dx = -np.sin(angle2)
    dy = np.cos(angle2)

    # Segment length: half a thinned-pixel, scaled by |P|/Pmax.
    p_sub = P2[::thin, ::thin][sub]
    p_norm = p_sub / np.nanmax(p_sub) if np.nanmax(p_sub) > 0 else p_sub
    L = thin * 0.45 * p_norm

    xs_a = x_centers - 0.5 * L * dx
    xs_b = x_centers + 0.5 * L * dx
    ys_a = y_centers - 0.5 * L * dy
    ys_b = y_centers + 0.5 * L * dy

    # Build a single line trace with None separators between segments.
    seg_x = np.empty(3 * len(x_centers))
    seg_y = np.empty(3 * len(x_centers))
    seg_x[0::3] = xs_a
    seg_x[1::3] = xs_b
    seg_x[2::3] = np.nan
    seg_y[0::3] = ys_a
    seg_y[1::3] = ys_b
    seg_y[2::3] = np.nan

    lines = go.Scatter(
        x=seg_x.tolist(), y=seg_y.tolist(),
        mode="lines",
        line=dict(color="white", width=1.4),
        hoverinfo="skip",
        showlegend=False,
        name="EVPA",
    )

    dots = None
    if colour_by_m:
        m_sub = M2[::thin, ::thin][sub]
        dots = go.Scatter(
            x=x_centers.tolist(), y=y_centers.tolist(),
            mode="markers",
            marker=dict(
                size=4,
                color=m_sub.tolist(),
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="|m|"),
                    x=0.40, y=0.32, yanchor="middle", len=0.30,
                    thickness=8,
                ),
                showscale=True,
                line=dict(width=0),
            ),
            hovertemplate="|m|=%{marker.color:.3f}<extra></extra>",
            showlegend=False,
            name="|m|",
        )
    return lines, dots


def dashboard(
    im: Image,
    obs: Obsdata,
    caltable: Caltable,
    *,
    pol: str = "R",
    show_model: bool = True,
    ttype: str = "direct",
    plotp: bool = False,
    nvec: int = 20,
    pcut: float = 0.1,
    mcut: float = 0.01,
    vec_cfun: bool = False,
    triangle: tuple[str, str, str] | None = None,
    quadrangle: tuple[str, str, str, str] | None = None,
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
        Triangle used for closure-phase products. None auto-picks the
        triangle with the most samples.
    quadrangle : (s1, s2, s3, s4) or None
        Quadrangle used for closure-amplitude products. None auto-picks.
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
        raise ValueError(
            f"default_product must be one of {_DASH_PRODUCT_ORDER}, "
            f"got {default_product!r}"
        )

    if triangle is None:
        triangle = _pick_default_triangle(obs)
    if quadrangle is None:
        quadrangle = _pick_default_quad(obs)

    obs_model = im.observe_same_nonoise(obs, ttype=ttype) if show_model else None

    products = _build_dashboard_products(
        obs, obs_model,
        triangle=triangle, quadrangle=quadrangle,
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Image (Stokes I)",
            _DASH_PRODUCT_LABELS[default_product],
            f"Gains (|G|, pol={pol})",
            "D-terms (complex plane)",
        ),
        # Narrower left column so the (square) image fills its cell; column
        # 2 picks up the slack and gives the panel + D-terms more room.
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.13, vertical_spacing=0.14,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
    )

    # --- Panel 1: image ---
    img2d = im.imvec.reshape(im.ydim, im.xdim)
    heatmap_kwargs = dict(
        z=img2d, colorscale="hot",
        hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.3g}<extra></extra>",
    )
    if plotp:
        heatmap_kwargs.update(
            showscale=True,
            colorbar=dict(
                title=dict(text="Jy/px"),
                x=0.40, y=0.78, yanchor="middle", len=0.32,
                thickness=10,
            ),
        )
    else:
        heatmap_kwargs["showscale"] = False
    fig.add_trace(go.Heatmap(**heatmap_kwargs), row=1, col=1)

    if plotp:
        lines, dots = _pol_ticks_traces(
            im, nvec=nvec, pcut=pcut, mcut=mcut, colour_by_m=vec_cfun)
        if lines is not None:
            fig.add_trace(lines, row=1, col=1)
        if dots is not None:
            fig.add_trace(dots, row=1, col=1)

    # --- Panel 2: data product selector (data + model traces per product) ---
    # Two traces per product, in `_DASH_PRODUCT_ORDER`. Only the
    # `default_product` pair is visible at start; the dropdown toggles.
    panel2_start = len(fig.data)
    for key in _DASH_PRODUCT_ORDER:
        p = products[key]
        is_default = (key == default_product)
        fig.add_trace(go.Scatter(
            x=p["x_data"], y=p["y_data"], mode="markers",
            name="data",
            marker=dict(size=4, color=_GRAY, opacity=0.6,
                        line=dict(width=0.3, color=_THEME["marker_edge"])),
            hovertemplate=("data<br>" + "x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"),
            legend="legend2",
            visible=is_default,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=p["x_model"], y=p["y_model"], mode="markers",
            name="model",
            marker=dict(size=4, color=_THEME["colorway"][1], opacity=0.85,
                        line=dict(width=0.3, color=_THEME["marker_edge"])),
            hovertemplate=("model<br>" + "x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>"),
            legend="legend2",
            visible=is_default,
        ), row=1, col=2)
    panel2_end = len(fig.data)

    # --- Panel 3: gains per site (own legend) ---
    # TODO: schema-coupled — replace 'rscale'/'lscale' once mixed-pol lands.
    pol_key = "rscale" if pol == "R" else "lscale"
    for site in sorted(caltable.data.keys()):
        gain = np.abs(caltable.data[site][pol_key])
        fig.add_trace(go.Scatter(
            x=caltable.data[site]["time"], y=gain,
            mode="markers", name=site,
            marker=dict(size=5,
                        line=dict(width=0.3, color=_THEME["marker_edge"])),
            hovertemplate=(
                f"<b>{site}</b><br>t=%{{x:.2f}} hr<br>|G|=%{{y:.3g}}<extra></extra>"
            ),
            legend="legend3",
        ), row=2, col=1)

    # --- Panel 4: D-terms (R and L) in complex plane (own legend) ---
    # TODO: schema-coupled — tarr['dr'] / tarr['dl'] move to caltable.dterms
    # in MixPol Phase 1; this lookup needs to follow.
    tarr = caltable.tarr
    sites = [str(s) for s in tarr["site"]]
    fig.add_trace(go.Scatter(
        x=np.real(tarr["dr"]), y=np.imag(tarr["dr"]),
        mode="markers+text", text=sites, textposition="top center",
        textfont=dict(size=9, color=_THEME["font_color"]),
        name="D_R",
        marker=dict(size=10, symbol="circle",
                    line=dict(width=0.5, color=_THEME["marker_edge"])),
        hovertemplate=(
            "<b>%{text}</b><br>Re(D_R)=%{x:.3g}<br>Im(D_R)=%{y:.3g}<extra></extra>"
        ),
        legend="legend4",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=np.real(tarr["dl"]), y=np.imag(tarr["dl"]),
        mode="markers+text", text=sites, textposition="bottom center",
        textfont=dict(size=9, color=_THEME["font_color"]),
        name="D_L",
        marker=dict(size=10, symbol="square",
                    line=dict(width=0.5, color=_THEME["marker_edge"])),
        hovertemplate=(
            "<b>%{text}</b><br>Re(D_L)=%{x:.3g}<br>Im(D_L)=%{y:.3g}<extra></extra>"
        ),
        legend="legend4",
    ), row=2, col=2)

    # --- Layout (multi-legend: one per scatter panel) ---
    fig.update_layout(
        title=dict(text=f"Reconstruction dashboard — {obs.source}",
                   x=0.5, xanchor="center",
                   font=dict(size=_THEME["title_size"],
                             family=_THEME["font_family"],
                             color=_THEME["font_color"])),
        template="none",
        plot_bgcolor=_THEME["plot_bgcolor"],
        paper_bgcolor=_THEME["paper_bgcolor"],
        font=dict(family=_THEME["font_family"], size=_THEME["font_size"],
                  color=_THEME["font_color"]),
        width=1280, height=1000,
        margin=dict(l=70, r=200, t=120, b=60),
        showlegend=True,
        colorway=_THEME["colorway"],
        # legend2 (data/model), legend3 (gains), legend4 (D-terms).
        legend2=dict(
            title=dict(text="Panel 2"),
            x=1.02, y=0.98, xanchor="left", yanchor="top",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"], borderwidth=1,
        ),
        legend3=dict(
            title=dict(text="Sites"),
            x=1.02, y=0.50, xanchor="left", yanchor="middle",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"], borderwidth=1,
        ),
        legend4=dict(
            title=dict(text="D-terms"),
            x=1.02, y=0.04, xanchor="left", yanchor="bottom",
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"], borderwidth=1,
        ),
    )

    # Common axis styling per subplot.
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(gridcolor=_THEME["grid_color"],
                         linecolor=_THEME["edge_color"],
                         tickcolor=_THEME["edge_color"],
                         showline=True, mirror=True, row=r, col=c)
        fig.update_yaxes(gridcolor=_THEME["grid_color"],
                         linecolor=_THEME["edge_color"],
                         tickcolor=_THEME["edge_color"],
                         showline=True, mirror=True, row=r, col=c)

    # Per-panel axis labels and aspect.
    fig.update_xaxes(title_text="x (pixel)", row=1, col=1)
    fig.update_yaxes(title_text="y (pixel)", row=1, col=1,
                     scaleanchor="x", scaleratio=1, autorange="reversed")
    fig.update_xaxes(title_text=products[default_product]["x_title"], row=1, col=2)
    fig.update_yaxes(title_text=products[default_product]["y_title"], row=1, col=2)
    fig.update_xaxes(title_text="time (hr)", row=2, col=1)
    fig.update_yaxes(title_text="|G|", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Re(D)", row=2, col=2, zeroline=True,
                     zerolinecolor=_THEME["edge_color"])
    fig.update_yaxes(title_text="Im(D)", row=2, col=2,
                     scaleanchor="x4", scaleratio=1,
                     zeroline=True, zerolinecolor=_THEME["edge_color"])

    # --- Data-product dropdown for panel 2 ---
    buttons = []
    for key in _DASH_PRODUCT_ORDER:
        p = products[key]
        vis = [tr.visible if tr.visible is not False else False
               for tr in fig.data]
        # Anything outside panel 2 (image + ticks + gains + D-terms) stays
        # visible regardless of selection.
        for i in range(len(fig.data)):
            if i < panel2_start or i >= panel2_end:
                vis[i] = True
            else:
                # Two consecutive traces (data, model) per product.
                product_idx = (i - panel2_start) // 2
                vis[i] = (_DASH_PRODUCT_ORDER[product_idx] == key)
        buttons.append(dict(
            label=_DASH_PRODUCT_LABELS[key],
            method="update",
            args=[
                {"visible": vis},
                {"xaxis2.title.text": p["x_title"],
                 "yaxis2.title.text": p["y_title"]},
            ],
        ))

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                showactive=True,
                x=0.58, xanchor="left",
                y=1.06, yanchor="bottom",
                pad=dict(l=4, r=4, t=2, b=2),
                bgcolor="rgba(238,238,238,0.85)",
                bordercolor=_THEME["edge_color"], borderwidth=1,
                font=dict(size=10, color=_THEME["font_color"]),
            ),
        ],
    )

    if show:
        fig.show()
    return fig
