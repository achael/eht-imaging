# Polarization conventions in ehtim

This document is the authoritative reference for how ehtim represents
polarization. All basis transforms and Jones-matrix utilities are
implemented in
[`ehtim/observing/pol_conventions.py`](../ehtim/observing/pol_conventions.py);
this document derives the formulas and pins down the sign conventions.

If you ever need to change a convention, change it in one place
(`pol_conventions.py`) and update this document.

---

## §1. Coordinate frames and field definitions

We work in the sky-tangent frame at the source, with the right-handed
basis $(\hat e_X, \hat e_Y, \hat k)$ where $\hat k$ points from source
to observer. The electric field of a monochromatic plane wave is

$$\vec E(t) = \mathrm{Re}\,[E_X \hat e_X + E_Y \hat e_Y]\,e^{+i\omega t},$$

with $E_X, E_Y$ complex analytic amplitudes. We use the engineering /
IEEE positive-frequency convention $e^{+i\omega t}$ standard in radio
interferometry literature (e.g., TMS Ch. 4). The physics convention
$e^{-i\omega t}$ would conjugate all complex amplitudes and flip the
sign of $V$ in the §5 linear formulas; everything else is invariant.

The "feed" of an antenna
projects $\vec E$ onto a basis vector; the two pure-feed cases of
interest are **linear** (samples $E_X, E_Y$ directly) and **circular**
(samples $E_R, E_L$ — defined below).

Implemented at: convention statement in
`ehtim/observing/pol_conventions.py:18-43` (module docstring).

---

## §2. Linear-circular basis transform — the convention choice (IAU/HBS)

ehtim adopts the IAU / Hamaker-Bregman-Sault / CASA / AIPS convention
under the §1 engineering time dependence $e^{+i\omega t}$:

$$R = \frac{X + iY}{\sqrt 2}, \quad L = \frac{X - iY}{\sqrt 2}.$$

In matrix form, the basis-change matrix
$M_{\mathrm{lin}\to\mathrm{circ}}$ acting on the column vector
$(E_X, E_Y)^\top$ to produce $(E_R, E_L)^\top$ is

$$M_{\mathrm{lin}\to\mathrm{circ}} = \frac{1}{\sqrt 2}
  \begin{pmatrix} 1 & +i \\ 1 & -i \end{pmatrix}.$$

It is unitary, so
$M_{\mathrm{circ}\to\mathrm{lin}} = M_{\mathrm{lin}\to\mathrm{circ}}^\dagger$.

The IAU/HBS convention is the standard in radio interferometry
calibration software (CASA, AIPS, DIFX) and is what AIPS POLTYA/POLTYB
labels assume when they report an X/Y feed.

Under the physics time convention $e^{-i\omega t}$ the same physical
basis is written $R = (X - iY)/\sqrt 2$ (complex conjugates of the
amplitudes). Switching to that convention requires changing the
constant `BASIS_LIN_TO_CIRC` in `pol_conventions.py` and flipping the
sign of $V$ in §5; §4 is unaffected.

Implemented at: `ehtim/observing/pol_conventions.py:56-60`.

---

## §3. Visibility correlations

For two stations $i, j$ with feeds $p, q$, the visibility is

$$V^{ij}_{pq} = \langle s^p_i \, (s^q_j)^* \rangle,$$

where $s^p_i$ is the complex analytic feed-signal at station $i$ feed
$p$. With four pure-feed station pairs we have four correlations per
baseline.

The natural object is the $2\times 2$ **coherency matrix**

$$C^{ij} = \begin{pmatrix} V^{ij}_{p_1 p_1} & V^{ij}_{p_1 p_2} \\
                            V^{ij}_{p_2 p_1} & V^{ij}_{p_2 p_2} \end{pmatrix},$$

where $(p_1, p_2)$ is either $(R, L)$ or $(X, Y)$ depending on the feed
type of stations $i$ and $j$.

ehtim stores the four entries of $C$ as separate columns
(`rrvis/llvis/rlvis/lrvis` for circular, `xxvis/yyvis/xyvis/yxvis` for
linear, or the generic `p1p1vis/p2p2vis/p1p2vis/p2p1vis` title aliases).

Implemented at: dtype declarations in
`ehtim/const_def.py:99-115` (DTPOL_CIRC, DTPOL_LIN).

---

## §4. Stokes from circular feeds

The Stokes parameters $(I, Q, U, V)$ in the visibility domain are
combinations of the four circular correlations:

$$I = \tfrac{1}{2}(RR + LL), \quad V = \tfrac{1}{2}(RR - LL),$$
$$Q = \tfrac{1}{2}(RL + LR), \quad U = \tfrac{i}{2}(LR - RL).$$

These match ehtim's existing (pre-mixed-pol) circular-to-Stokes code
and define what we mean by Stokes throughout the codebase.

Implemented at: pair primitives in
`ehtim/observing/pol_conventions.py:108-115` (`circ_to_stokes_parallel`),
`:117-126` (`circ_to_stokes_cross`), `:128-131`
(`stokes_to_circ_parallel`), `:134-137` (`stokes_to_circ_cross`).
Four-component wrappers at `:176-186` (`circ_to_stokes`) and
`:188-198` (`stokes_to_circ`).

---

## §5. Stokes from linear feeds

Following §3, the linear correlator labels are
$XX \equiv \langle E_X E_X^* \rangle$,
$XY \equiv \langle E_X E_Y^* \rangle$,
$YX \equiv \langle E_Y E_X^* \rangle$,
$YY \equiv \langle E_Y E_Y^* \rangle$
(first index un-conjugated, second conjugated). This labeling, together
with §2, fixes the sign of $V$.

Substituting the §2 basis transform into the §4 definitions gives the
linear-feed analog. Starting from $R = (X + iY)/\sqrt 2$ and
$L = (X - iY)/\sqrt 2$, the four circular correlations expand to

$$RR = \tfrac{1}{2}[(XX + YY) - i(XY - YX)],$$
$$LL = \tfrac{1}{2}[(XX + YY) + i(XY - YX)],$$
$$RL = \tfrac{1}{2}[(XX - YY) + i(XY + YX)],$$
$$LR = \tfrac{1}{2}[(XX - YY) - i(XY + YX)].$$

Plugging into §4 yields the linear-feed Stokes formulas:

$$\boxed{\,I = \tfrac{1}{2}(XX + YY), \quad Q = \tfrac{1}{2}(XX - YY), \quad
       U = \tfrac{1}{2}(XY + YX), \quad V = -\tfrac{i}{2}(XY - YX).\,}$$

Inverted (to derive linear correlations from Stokes):

$$XX = I + Q, \quad YY = I - Q, \quad XY = U + iV, \quad YX = U - iV.$$

The **sign of V** depends on the §1/§2 choice: under the physics
convention ($e^{-i\omega t}$, $R = (X - iY)/\sqrt 2$), $V \to +i(XY -
YX)/2$ and accordingly $XY \to U - iV$, $YX \to U + iV$. The circular
formulas in §4 are invariant.

### §5a. Comparison to Thompson, Moran & Swenson (3rd ed.)

TMS Eq. 4.28 (engineering convention $e^{+j\omega t}$, matching our §1)
gives the cross-hand linear correlations as

$$\langle E_X E_Y^* \rangle = \tfrac{1}{2}(U + jV), \qquad
  \langle E_Y E_X^* \rangle = \tfrac{1}{2}(U - jV),$$

which agree in sign with ehtim's $XY = U + iV$, $YX = U - iV$. The
remaining difference is a factor-of-2 normalization: TMS Eq. 4.28
puts the $1/2$ on the Stokes side of every correlation (parallel and
cross), whereas ehtim §5 puts the $1/2$ on the correlation side.
Equivalently, $\text{(TMS Stokes)} = 2 \times \text{(ehtim Stokes)}$
for all of $I, Q, U, V$. A data set written under TMS Eq. 4.28
normalization must therefore have all four Stokes parameters rescaled
by $\tfrac{1}{2}$ before being ingested into the ehtim pipeline.

Implemented at: pair primitives in
`ehtim/observing/pol_conventions.py:140-146` (`lin_to_stokes_diag`),
`:149-158` (`lin_to_stokes_offdiag`), `:160-163`
(`stokes_to_lin_diag`), `:166-169` (`stokes_to_lin_offdiag`).
Four-component wrappers at `:200-210` (`lin_to_stokes`) and
`:212-222` (`stokes_to_lin`).

---

## §6. Visibility sigmas

Each visibility column has a paired sigma column (`rrsigma`/`xxsigma`/
`sigma`/etc.) carrying a per-row $1\sigma$ thermal noise estimate, in
the same units as the visibility itself. Sigmas are real and
non-negative.

For independent, complex-circular Gaussian noise on the input
correlations, propagating the §4 and §5 linear combinations gives the
per-component output sigmas:

**circular → Stokes**

$$\sigma_I = \sigma_V = \tfrac{1}{2}\sqrt{\sigma_{RR}^2 + \sigma_{LL}^2}, \quad
  \sigma_Q = \sigma_U = \tfrac{1}{2}\sqrt{\sigma_{RL}^2 + \sigma_{LR}^2}.$$

**Stokes → circular**

$$\sigma_{RR} = \sigma_{LL} = \sqrt{\sigma_I^2 + \sigma_V^2}, \quad
  \sigma_{RL} = \sigma_{LR} = \sqrt{\sigma_Q^2 + \sigma_U^2}.$$

**linear → Stokes** (under the §2 IAU/HBS convention)

$$\sigma_I = \sigma_Q = \tfrac{1}{2}\sqrt{\sigma_{XX}^2 + \sigma_{YY}^2}, \quad
  \sigma_U = \sigma_V = \tfrac{1}{2}\sqrt{\sigma_{XY}^2 + \sigma_{YX}^2}.$$

**Stokes → linear**

$$\sigma_{XX} = \sigma_{YY} = \sqrt{\sigma_I^2 + \sigma_Q^2}, \quad
  \sigma_{XY} = \sigma_{YX} = \sqrt{\sigma_U^2 + \sigma_V^2}.$$

### Known limitation: cross-component covariance is dropped

Each output component is a linear combination of (typically) two input
components. The output components are therefore **correlated** even
when the inputs are independent, but the formulas above record only
marginal variances — the full output covariance has nonzero off-diagonal
entries that we discard.

Concrete consequences:

- Round-trips do not recover the input sigmas exactly. For example,
  $\sigma_I, \sigma_Q$ from `circ_to_stokes_sigma` followed by
  `stokes_to_circ_sigma` returns
  $\sqrt{\sigma_I^2 + \sigma_V^2} = \tfrac{1}{\sqrt 2}\sqrt{\sigma_{RR}^2 + \sigma_{LL}^2}$
  for the recovered $\sigma_{RR}$, which differs from the true
  $\sigma_{RR}$ unless $\sigma_{RR} = \sigma_{LL}$.
- Per-baseline closures, weighted averages, and Bayesian likelihoods
  that mix Stokes and feed-basis quantities will be slightly
  miscalibrated if they treat the sigmas as independent across
  components.

The proper treatment propagates the full $4\times 4$ covariance
matrix:

$$\mathrm{Cov}_{\mathrm{out}} = M\,\mathrm{Cov}_{\mathrm{in}}\,M^\dagger,$$

where $M$ is the $4\times 4$ basis-transform matrix on the visibility
vector. This produces nonzero off-diagonal entries that downstream
code can use when it cares. **No current ehtim consumer reads
off-diagonal covariance**, so the per-component sigma transforms are
sufficient for bit-identical migration of existing Obsdata behavior —
but new code that mixes polreps in noise calculations should not
assume independence.

Implemented at: `ehtim/observing/pol_conventions.py:257-266`
(`circ_to_stokes_sigma`), `:268-277` (`stokes_to_circ_sigma`),
`:279-287` (`lin_to_stokes_sigma`), `:290-294` (`stokes_to_lin_sigma`).
The covariance-aware version is not yet implemented; it will arrive
when a downstream consumer needs it.

---

## §7. Cross-check: `lin → circ → stokes` vs direct `lin → stokes`

The two compositions must agree by construction (basis transforms are
linear). Concretely, starting from $(XX, YY, XY, YX)$:

1. Apply §5 inverse → $(I, Q, U, V)$
2. Apply §4 forward → $(RR, LL, RL, LR)$

vs.

1. Apply §5 (relations above) directly → $(RR, LL, RL, LR)$

Both routes produce the same circular correlations. The `lin_to_circ`
wrapper in `pol_conventions.py` is implemented as the two-step
composition `lin_to_stokes` → `stokes_to_circ` to keep a single source
of truth and make any future convention change a one-line edit.

Implemented at: `ehtim/observing/pol_conventions.py:224-226`
(`lin_to_circ`) and `:229-231` (`circ_to_lin`).
The cross-check is enforced by a round-trip test in
`tests/test_pol_conventions.py`.

---

## §8. Faraday rotation sign convention

Faraday rotation of an EVPA $\chi$ by an amount $\phi(\nu)$ (the
rotation measure times $\lambda^2$ minus the reference value) rotates
$Q + iU$ as

$$Q' + iU' = (Q + iU)\,e^{+2i\phi}.$$

The sign $+2i\phi$ is the convention used in the EHT data pipeline and
is consistent with the §2 basis choice. (Multifrequency Faraday-rotation
work is tracked in `obsdata_multifreq_plan.md`; this section is a
forward-pointer.)

Implemented at: not yet — Faraday rotation lives in calibration code
that this module does not own. When that code consolidates here,
update.

---

## §9. Jones matrices

A station's net response to an incident field $\vec E$ is a $2\times 2$
complex matrix $J$ acting on the feed-basis vector:

$$\begin{pmatrix} V_{p_1} \\ V_{p_2} \end{pmatrix}_{\text{observed}}
  = J \begin{pmatrix} E_{p_1} \\ E_{p_2} \end{pmatrix}.$$

For a baseline $(i, j)$ the observed coherency matrix is

$$C^{ij}_{\text{obs}} = J_i \, C^{ij}_{\text{true}} \, J_j^\dagger,$$

where $J_i, J_j$ are the per-station Jones matrices. Inverting gives
the calibration formula in §11.

(The full mixed-pol Jones treatment lives in Alex's mixed-pol imaging
work. This module ships the per-station algebra so that downstream
calibration code has a single home to call into.)

Implemented at: scaffolding in
`ehtim/observing/pol_conventions.py:310-330` (`jones_matrix`),
`:332-337` (`invert_jones`), `:340-357`
(`apply_inverse_jones_to_coherency`).

---

## §10. Jones factoring: $J = G \cdot (I + D)$

ehtim factors a station's Jones matrix into a diagonal complex gain and
a D-term cross-coupling matrix:

$$J = G \cdot (I + D), \quad
  G = \begin{pmatrix} g_{p_1} & 0 \\ 0 & g_{p_2} \end{pmatrix}, \quad
  I + D = \begin{pmatrix} 1 & d_{p_1} \\ d_{p_2} & 1 \end{pmatrix}.$$

Equivalently,

$$J = \begin{pmatrix} g_{p_1} & g_{p_1} d_{p_1} \\
                       g_{p_2} d_{p_2} & g_{p_2} \end{pmatrix}.$$

The $d_{p_1}, d_{p_2}$ entries are the per-feed leakage of the
*other* feed into this one (so $d_{p_1}$ is the leakage of $p_2$ into
$p_1$).

This factoring matches the convention used in CASA's polcal and in the
EHT calibration pipeline. When the first consumer (full-Jones
`applycal`) lands, verify against existing `pol_cal*` code in this
repo — flip here if a sign mismatch is found.

Implemented at: `ehtim/observing/pol_conventions.py:310-330`
(`jones_matrix`).

---

## §11. Per-baseline visibility correction

Given per-station Jones matrices $J_i, J_j$ and an observed coherency
matrix $C^{ij}_{\text{obs}}$, the calibrated coherency is

$$C^{ij}_{\text{corr}} = J_i^{-1} \, C^{ij}_{\text{obs}} \, (J_j^\dagger)^{-1}.$$

When gains and D-terms are well-determined and the feed model is
correct, $C^{ij}_{\text{corr}}$ recovers the true sky-domain coherency
matrix up to noise.

Implemented at: `ehtim/observing/pol_conventions.py:340-357`
(`apply_inverse_jones_to_coherency`).

---

## §12. Polrep-specific Jones application

The $2\times 2$ coherency matrix corresponds to different physical
quantities depending on the feed types of the two stations:

| Stations | $C_{11}$ | $C_{12}$ | $C_{21}$ | $C_{22}$ | polrep |
|---|---|---|---|---|---|
| circular–circular | $V_{RR}$ | $V_{RL}$ | $V_{LR}$ | $V_{LL}$ | `'circ'` |
| linear–linear     | $V_{XX}$ | $V_{XY}$ | $V_{YX}$ | $V_{YY}$ | `'lin'`  |
| circ–linear (i.e. station $i$ has circular feed, $j$ has linear) | $V_{RX}$ | $V_{RY}$ | $V_{LX}$ | $V_{LY}$ | `'mixed'` (per-baseline) |
| stokes (sky-frame; only meaningful at calibrated-image level) | $I$ | $Q + iU$ | $Q - iU$ | $V$ | `'stokes'` |

The Jones machinery in §9-§11 operates on the coherency matrix and is
**polrep-agnostic** — it does not know which physical quantities are
in each slot. The polrep is used only to label the slots when packing
results back into ehtim's column-storage layout (`rrvis`/`xxvis`/etc.).

For `'mixed'` polrep the slot interpretation varies per baseline, so
the per-baseline `polbasis` field on the DTPOL_MIXED dtype carries the
feed-pair label needed to route results correctly. This is set up in
the schema layer (DTPOL_MIXED definition in `ehtim/const_def.py`)
and consumed by Obsdata polrep-dispatching code.

Implemented at: dispatch logic in `Obsdata.switch_polrep` and friends
(forthcoming).
