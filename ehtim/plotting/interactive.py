"""Interactive Plotly-based plotting for ehtim.

Sits beside the matplotlib defaults in 'comp_plots.py', 'summary_plots.py',
and the 'plot_*' methods on 'Obsdata' / 'Caltable'.
Plotly is imported lazily so this module imports cleanly without plotly installed;
the import only fires on first call to a plotting function.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

import ehtim.const_def as ehc
import ehtim.observing.obs_helpers as obsh

if TYPE_CHECKING:
    from os import PathLike

    from ehtim.caltable import Caltable
    from ehtim.image import Image
    from ehtim.obsdata import Obsdata

def _require_plotly():
    """Import and return `plotly.graph_objects`, with an install hint on failure."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "Plotly is required for ehtim.plotting.interactive. "
            "Install with `pip install plotly`."
        ) from e
    return go


def _require_plotly_subplots():
    """Lazily import `plotly.subplots.make_subplots`."""
    try:
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "Plotly is required for ehtim.plotting.interactive. "
            "Install with `pip install plotly`."
        ) from e
    return make_subplots


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
    # Outline + error-bar colours tuned for the light bg.
    "marker_edge":  "rgba(0,0,0,0.4)",
    "error_color":  "rgba(0,0,0,0.3)",
}

# Fallback symbols when a plot has more traces than the colorway has colors.
_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"]


def _apply_theme(fig, *, title, xaxis_title, yaxis_title,
                 rangex=None, rangey=None, y_type="linear",
                 legend_title="", show_legend=True):
    """Apply the BMH visual theme to a plotly Figure."""
    fig.update_layout(
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
        template="none", # Strip default light templates
        plot_bgcolor=_THEME["plot_bgcolor"],
        paper_bgcolor=_THEME["paper_bgcolor"],
        font=dict(family=_THEME["font_family"],
                  size=_THEME["font_size"],
                  color=_THEME["font_color"]),
        margin=dict(l=70, r=160, t=60, b=60),
        width=_THEME["width"], height=_THEME["height"],
        hovermode="closest",
        colorway=_THEME["colorway"],
        showlegend=show_legend,
        legend=dict(
            title=dict(text=legend_title),
            orientation="v",
            x=1.02, y=1, xanchor="left", yanchor="top",
            font=dict(size=11, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)", # Matches plot bg, slightly transparent
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
            itemsizing="constant",
        ),
    )
# --- Click-to-highlight JS ------------------------------------------------

# Tri-state legend click for uniform-gray traces:
#   gray default → single click paints it from PALETTE
#                → second click hides it
#                → third click restores gray
# Traces that started non-gray (e.g. plot_gains) fall through to plotly's
# default legend-click toggle so we don't fight existing colouring.
#
# `{plot_id}` is a placeholder plotly substitutes for the div id when this is
# passed as `post_script=` to `write_html` / `to_html`. Curly braces in the JS
# itself are escaped by doubling so the raw f-string doesn't eat them.
_GRAY = "#888888"


def _legend_click_js() -> str:
    palette = json.dumps(_THEME["colorway"])
    gray = json.dumps(_GRAY)
    return f"""
(function() {{
    var gd = document.getElementById('{{plot_id}}');
    if (!gd) return;
    var PALETTE = {palette};
    var GRAY = {gray};
    var states = {{}};
    gd.on('plotly_legendclick', function(ev) {{
        var idx = ev.curveNumber;
        var cur = states[idx];
        if (cur !== undefined) {{
            if (cur === 'color') {{
                Plotly.restyle(gd, {{visible: 'legendonly'}}, [idx]);
                states[idx] = 'hidden';
            }} else if (cur === 'hidden') {{
                Plotly.restyle(gd, {{visible: true, 'marker.color': GRAY}}, [idx]);
                states[idx] = 'gray';
            }} else {{
                Plotly.restyle(gd, {{'marker.color': PALETTE[idx % PALETTE.length]}}, [idx]);
                states[idx] = 'color';
            }}
            return false;
        }}
        var tr = gd.data[idx];
        var base = (tr.marker && tr.marker.color) || '';
        if (base === GRAY) {{
            Plotly.restyle(gd, {{'marker.color': PALETTE[idx % PALETTE.length]}}, [idx]);
            states[idx] = 'color';
            return false;
        }}
        return true;  // already-coloured trace → plotly default toggle
    }});
}})();
"""


def write_html(fig, path: str | PathLike[str], *,
               include_plotlyjs: bool | str = True) -> None:
    """Write `fig` to an HTML file with the click-to-highlight JS embedded.

    The resulting file is self-contained and reproduces the same legend-click
    UX you get in a notebook via `interactive.display(fig)`. Use this instead
    of `fig.write_html(...)` whenever you want to share a clickable HTML.

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
    the legend tri-state highlight. Returns nothing — the figure is shown as
    a side effect.
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

def _build_baseline_trace(go, obs, site1, site2, field, sigtype, *,
                          snrcut, ang_unit, debias, timetype):
    """Return a plotly Scatter for one baseline, or None if no valid data."""
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
            f"{field}: %{{y:.4g}}"
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
    - **All baselines:** pass neither. One trace per baseline that appears
      in `obs.data`; click legend entries to toggle, double-click to isolate.

    Each point carries (site1, site2, time, value, error) as plotly
    `customdata`, surfaced on hover.
    """
    go = _require_plotly()

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

    fig = go.Figure()
    for s1, s2 in pairs:
        trace = _build_baseline_trace(
            go, obs, s1, s2, field, sigtype,
            snrcut=snrcut, ang_unit=ang_unit, debias=debias, timetype=timetype,
        )
        if trace is None:
            continue
        if len(pairs) > 1:
            # Uniform muted colour so the palette never runs out on large
            # arrays. Double-click a legend entry to isolate one baseline.
            trace.marker.color = _GRAY
            trace.marker.opacity = 0.6
        fig.add_trace(trace)

    if not fig.data:
        print(f"No valid data after filtering (snrcut={snrcut}).")
        return fig

    y_title = ehc.FIELD_LABELS.get(field, field.capitalize())
    if field in ehc.FIELDS_PHASE:
        y_title += f" ({ang_unit})"

    _apply_theme(
        fig,
        title=f"{obs.source}: {title_suffix} ({field})",
        xaxis_title=f"{timetype} (hr)",
        yaxis_title=y_title,
        rangex=rangex, rangey=rangey,
        legend_title="Baseline",
        show_legend=show_legend,
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

    - **tag_bl=True (default):** one trace per baseline, click legend to toggle.
    - **tag_bl=False:** all visibilities pooled into one trace.

    Common uses:

    - ''plotall(obs, 'u', 'v', conj=True)'' — uv coverage with conjugates
    - ''plotall(obs, 'uvdist', 'amp')'' — amp vs baseline length (radplot)
    """
    go = _require_plotly()

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
        m = ~(np.isnan(data[field1]) + np.isnan(data[field2]))
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

            x = data[field1]
            y = data[field2]
            customdata = np.column_stack([
                np.full(len(x), t1, dtype=object),
                np.full(len(x), t2, dtype=object),
                sigx if sigx is not None else np.full(len(x), np.nan),
                sigy if sigy is not None else np.full(len(x), np.nan),
            ])

            # Uniform muted colour so the palette never runs out on large
            # arrays. Double-click a legend entry to isolate one baseline.
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                name=f"{t1}-{t2}",
                marker=dict(size=6, opacity=0.6, color=_GRAY,
                            line=dict(width=0.5, color=_THEME["marker_edge"])),
                error_x=(dict(type="data", array=sigx, visible=True,
                              thickness=1, width=2, color=_THEME["error_color"])
                         if sigx is not None else dict(visible=False)),
                error_y=(dict(type="data", array=sigy, visible=True,
                              thickness=1, width=2, color=_THEME["error_color"])
                         if sigy is not None else dict(visible=False)),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{t1}–{t2}</b><br>"
                    f"{field1}: %{{x:.4g}}<br>"
                    f"{field2}: %{{y:.4g}}"
                    "<extra></extra>"
                ),
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
            fig.add_trace(go.Scatter(
                x=data[field1], y=data[field2], mode="markers",
                name="all baselines",
                marker=dict(size=6, opacity=0.7,
                            line=dict(width=0.5, color=_THEME["marker_edge"])),
                error_x=(dict(type="data", array=sigx, visible=True,
                              thickness=1, width=2, color=_THEME["error_color"])
                         if sigx is not None else dict(visible=False)),
                error_y=(dict(type="data", array=sigy, visible=True,
                              thickness=1, width=2, color=_THEME["error_color"])
                         if sigy is not None else dict(visible=False)),
                hovertemplate=(
                    f"{field1}: %{{x:.4g}}<br>"
                    f"{field2}: %{{y:.4g}}"
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
    if field2 in ehc.FIELDS_PHASE:
        y_title += f" ({ang_unit})"

    _apply_theme(
        fig,
        title=f"{obs.source}: {field1} vs {field2}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        rangex=rangex, rangey=rangey,
        legend_title="Baseline" if tag_bl else "",
        show_legend=tag_bl,
    )

    if xscale == "log":
        fig.update_xaxes(type="log")
    if yscale == "log":
        fig.update_yaxes(type="log")

    if show:
        fig.show()
    return fig


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
        TODO: replace with pol1/pol2 once the cal-table mixed-pol schema lands.
    timetype : {'GMST', 'UTC'} or None
        None means use `caltable.timetype`.
    ang_unit : {'deg', 'rad'}
        Phase unit when `gain_type='phase'`.
    yscale : {'log', 'lin'}
        Only honored for `gain_type='amp'`; phase plots are always linear.
    """
    go = _require_plotly()

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
            # When plotting both pols, give L a distinct symbol so it's
            # visually separable from R within the same colour.
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

def dashboard(
    im: Image,
    obs: Obsdata,
    caltable: Caltable,
    *,
    pol: str = "R",
    show_model: bool = True,
    ttype: str = "direct",
    show: bool = True,
) -> Any:
    """Reconstruction dashboard — image + data fit + gains + D-terms.

    2x2 layout requested by Rohan at the 2026-05-13 meeting: inspect a full
    reconstruction's model + data + calibration + leakage at a glance.

    Parameters
    ----------
    im : Image
        Reconstructed (or model) Stokes-I image.
    obs : Obsdata
        Observation. The amp-vs-uvdist panel plots 'obs' amplitudes; if
        'show_model=True', the model visibilities predicted by 'im at the
        same (u,v) points are overlaid.
    caltable : Caltable
        Calibration table; provides per-site gains and D-terms (from `tarr`).
    pol : {'R', 'L'}
        Which gain channel to plot. TODO: pol1/pol2 once the cal-table
        mixed-pol schema lands.
    show_model : bool
        Overlay model visibility amplitudes from 'im.observe_same_nonoise(obs)'
        on the amp-vs-uvdist panel. Default True (Rohan's data-fit ask).
    ttype : {'direct', 'fast', 'nfft'}
        Transform type for the model-visibility computation. 'direct' is the
        safest default; 'nfft' / 'fast' are faster but need their backends.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    make_subplots = _require_plotly_subplots()

    if pol not in ("R", "L"):
        raise ValueError(f"pol must be 'R' or 'L', got {pol!r}")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Image (Stokes I)",
            "Amplitude vs uv-distance",
            f"Gains (|G|, pol={pol})",
            "D-terms (complex plane)",
        ),
        # Narrower left column so the (square) image fills its cell; column
        # 2 picks up the slack and gives radplot + D-terms more room.
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.13, vertical_spacing=0.14,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
    )

    # --- Panel 1: image ---
    img2d = im.imvec.reshape(im.ydim, im.xdim)
    fig.add_trace(go.Heatmap(
        z=img2d, colorscale="hot", showscale=False,
        hovertemplate="x=%{x}<br>y=%{y}<br>I=%{z:.3g}<extra></extra>",
    ), row=1, col=1)

    # --- Panel 2: amp vs uvdist, with optional model overlay ---
    udata = obs.unpack(["uvdist", "amp"])
    fig.add_trace(go.Scatter(
        x=udata["uvdist"], y=udata["amp"], mode="markers",
        name="data",
        marker=dict(size=4, color=_GRAY, opacity=0.6,
                    line=dict(width=0.3, color=_THEME["marker_edge"])),
        hovertemplate="data<br>uvdist=%{x:.2e} λ<br>amp=%{y:.3g} Jy<extra></extra>",
        legendgroup="ampuv",
    ), row=1, col=2)
    if show_model:
        obs_model = im.observe_same_nonoise(obs, ttype=ttype)
        mdata = obs_model.unpack(["uvdist", "amp"])
        fig.add_trace(go.Scatter(
            x=mdata["uvdist"], y=mdata["amp"], mode="markers",
            name="model",
            marker=dict(size=4, color=_THEME["colorway"][1], opacity=0.85,
                        line=dict(width=0.3, color=_THEME["marker_edge"])),
            hovertemplate="model<br>uvdist=%{x:.2e} λ<br>amp=%{y:.3g} Jy<extra></extra>",
            legendgroup="ampuv",
        ), row=1, col=2)

    # --- Panel 3: gains per site ---
    # TODO: schema-coupled — replace 'rscale'/'lscale' once mixed-pol lands.
    pol_key = "rscale" if pol == "R" else "lscale"
    for site in sorted(caltable.data.keys()):
        gain = np.abs(caltable.data[site][pol_key])
        fig.add_trace(go.Scatter(
            x=caltable.data[site]["time"], y=gain,
            mode="markers", name=site,
            marker=dict(size=5,
                        line=dict(width=0.3, color=_THEME["marker_edge"])),
            hovertemplate=f"<b>{site}</b><br>t=%{{x:.2f}} hr<br>|G|=%{{y:.3g}}<extra></extra>",
        ), row=2, col=1)

    # --- Panel 4: D-terms (R and L) in complex plane ---
    tarr = caltable.tarr
    sites = [str(s) for s in tarr["site"]]
    fig.add_trace(go.Scatter(
        x=np.real(tarr["dr"]), y=np.imag(tarr["dr"]),
        mode="markers+text", text=sites, textposition="top center",
        textfont=dict(size=9, color=_THEME["font_color"]),
        name="D_R",
        marker=dict(size=10, symbol="circle",
                    line=dict(width=0.5, color=_THEME["marker_edge"])),
        hovertemplate="<b>%{text}</b><br>Re(D_R)=%{x:.3g}<br>Im(D_R)=%{y:.3g}<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=np.real(tarr["dl"]), y=np.imag(tarr["dl"]),
        mode="markers+text", text=sites, textposition="bottom center",
        textfont=dict(size=9, color=_THEME["font_color"]),
        name="D_L",
        marker=dict(size=10, symbol="square",
                    line=dict(width=0.5, color=_THEME["marker_edge"])),
        hovertemplate="<b>%{text}</b><br>Re(D_L)=%{x:.3g}<br>Im(D_L)=%{y:.3g}<extra></extra>",
    ), row=2, col=2)

    # --- Layout ---
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
        # Height bumped so row 1 is tall enough for the image to fill its
        # (now-narrower) cell at square pixel aspect.
        width=1200, height=1000,
        margin=dict(l=70, r=180, t=80, b=60),
        showlegend=True,
        colorway=_THEME["colorway"],
        legend=dict(
            font=dict(size=10, color=_THEME["font_color"]),
            bgcolor="rgba(238,238,238,0.85)",
            bordercolor=_THEME["edge_color"],
            borderwidth=1,
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
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
    fig.update_xaxes(title_text="uv-distance (λ)", row=1, col=2)
    fig.update_yaxes(title_text="amp (Jy)", row=1, col=2)
    fig.update_xaxes(title_text="time (hr)", row=2, col=1)
    fig.update_yaxes(title_text="|G|", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Re(D)", row=2, col=2, zeroline=True,
                     zerolinecolor=_THEME["edge_color"])
    fig.update_yaxes(title_text="Im(D)", row=2, col=2,
                     scaleanchor="x4", scaleratio=1,
                     zeroline=True, zerolinecolor=_THEME["edge_color"])

    if show:
        fig.show()
    return fig
