"""Four-case energy-density I_E/Q_E/Pi display; verified caches, no transport runs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from plot_style import apply_mnras_style, get_three_column_size

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/spectra"
WINDOWS = (("th015", 0, 30), ("th045", 30, 60), ("th075", 60, 90))
AZIMUTH_FACTOR = 2*np.pi
DEFAULT_CASES = ((1., 1.), (1., 10.), (20., 1.), (20., 10.))
DEFAULT_XLIMITS = ((.001, .2), (.001, 12.), (.001, 80.), (.001, 500.))
METHODS = (("recursion + Jones", "jones", "-", False),
           ("compPSc", "comppsc", "--", False),
           ("MCRT", "mcrt", None, True))


def _tag(value):
    return f"{float(value):g}".replace(".", "p")


def _read_csv(path, is_mc):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty spectrum: {path}")
    fields = ["E_left_keV", "E_right_keV", "E_center_keV",
              "I_N_per_keV_sr", "Q_N_per_keV_sr", "Pi_signed"]
    if is_mc:
        fields += ["I_N_se_per_keV_sr", "Q_N_se_per_keV_sr", "Pi_signed_se", "count"]
    result = {field: np.array([float(row[field]) if row[field].strip() else np.nan
                               for row in rows]) for field in fields}
    result.update({field: np.array([row[field] for row in rows])
                   for field in ("boundary", "theta_label")})
    if is_mc:
        result["plot_supported"] = np.array([row["plot_supported"].lower() == "true"
                                              for row in rows])
    if not np.all(result["E_right_keV"] > result["E_left_keV"]):
        raise ValueError(f"Nonpositive energy-bin width: {path}")
    intensity, stokes_q = result["I_N_per_keV_sr"], result["Q_N_per_keV_sr"]
    valid = (intensity > 0) & np.isfinite(result["Pi_signed"])
    np.testing.assert_allclose(result["Pi_signed"][valid], stokes_q[valid] / intensity[valid],
                               rtol=2e-10, atol=2e-12, err_msg=str(path))
    return result


def load_case(temperature, tau):
    directory = CACHE / f"kte{_tag(temperature)}_tau{_tag(tau)}"
    models, audits = [], {}
    for label, key, linestyle, is_mc in METHODS:
        filename = "mcrt_project_Q.csv" if is_mc else f"{key}.csv"
        path = directory / filename
        audit_path = directory / f"{key}_audit.json"
        if not path.is_file() or not audit_path.is_file():
            raise FileNotFoundError(f"Missing cache: {path}; this display never runs simulations")
        audit = json.loads(audit_path.read_text())
        if audit["kTe_keV"] != temperature or audit["tau"] != tau:
            raise ValueError(f"Parameter mismatch: {audit_path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["sha256"]:
            raise ValueError(f"Spectrum checksum mismatch: {path}")
        models.append((label, _read_csv(path, is_mc), linestyle, is_mc))
        audits[label] = audit
    mc = audits["MCRT"]
    if (mc["seed_model"], mc["seed_temperature_eV"], mc["injection"], mc["mode"]) != (
            "blackbody", 5., "lambert", "stokes"):
        raise ValueError("Seed/transport configuration does not match the benchmark")
    if (mc["stokes_convention"] != "project/manuscript Q = -raw MCRT Q"
            or tuple(map(tuple, mc["angular_windows"])) != WINDOWS):
        raise ValueError("Stokes convention or angular windows do not match")
    if mc["total_histories"] != 400_000_000 or mc["truncated_histories"] != 0:
        raise ValueError("Incomplete MCRT cache")
    if not audits["compPSc"]["iteration_converged"]:
        raise ValueError("compPSc scattering iteration did not converge")
    if audits["compPSc"]["seed_keV"] != .005:
        raise ValueError("compPSc seed temperature does not match")
    return models, audits


def plot_four_case_niq(cases=DEFAULT_CASES, xlimits=DEFAULT_XLIMITS,
                       surfaces=("up", "down"), intensity_decades=8,
                       pi_min_count=100, pi_signal_floor=1e-12, show=True,
                       intensity_ymin=None, q_ymax=None, q_ymin=None,
                       q_xmax=None, intensity_xmax=None, pi_xmax=None,
                       q_ylim_by_surface=None, save_directory=None):
    """One 4x3 figure per surface: I_E=2*pi*E*N_I, Q_E=2*pi*E*N_Q, 100*Q/I.

    Cached densities are per injected photon, per keV, per sr. Displayed
    densities integrate over azimuth and remain averaged per direction cosine
    over each angular window, in the manuscript's local Stokes basis.
    Exactly one E factor converts photon to energy densities (E in keV).
    No extra Delta_mu/cosine factors or amplitude matching are applied.
    Explicit I/Q y limits refer to the displayed, azimuth-integrated densities.
    For MCRT Pi we use the stored ratio SE (including I/Q covariance), not
    independent-error propagation. Count/signal masks affect display only.
    Polarization y limits use deterministic curves only; MCRT points and
    error bars are retained but may extend beyond the displayed y range.
    When save_directory is provided, export one tight-boundary PDF per surface.
    """
    cases, xlimits = tuple(cases), tuple(xlimits)
    if len(cases) != 4 or len(xlimits) != 4:
        raise ValueError("This layout requires four cases and four energy ranges")
    intensity_ymin = (None,)*4 if intensity_ymin is None else tuple(intensity_ymin)
    if len(intensity_ymin) != 4 or any(
            lower is not None and (not np.isfinite(lower) or lower <= 0)
            for lower in intensity_ymin):
        raise ValueError("Provide four positive finite intensity lower limits or None")
    q_ymax = (None,)*4 if q_ymax is None else tuple(q_ymax)
    if len(q_ymax) != 4 or any(
            upper is not None and not np.isfinite(upper) for upper in q_ymax):
        raise ValueError("Provide four finite Q upper limits or None")
    q_ymin = (None,)*4 if q_ymin is None else tuple(q_ymin)
    if len(q_ymin) != 4 or any(
            lower is not None and not np.isfinite(lower) for lower in q_ymin):
        raise ValueError("Provide four finite Q lower limits or None")
    q_ylim_by_surface = {} if q_ylim_by_surface is None else dict(q_ylim_by_surface)
    for surface, limits in q_ylim_by_surface.items():
        if surface not in ("up", "down") or len(limits) != 4:
            raise ValueError("Surface-specific Q limits require up/down and four row entries")
        for pair in limits:
            if pair is not None and (len(pair) != 2 or not np.all(np.isfinite(pair))
                                     or pair[0] >= pair[1]):
                raise ValueError("Each Q limit override must be a finite increasing pair or None")
    energy_overrides = {}
    for name, limits in (("I", intensity_xmax), ("Q", q_xmax), ("Pi", pi_xmax)):
        limits = (None,)*4 if limits is None else tuple(limits)
        if len(limits) != 4 or any(
                upper is not None and (not np.isfinite(upper) or upper <= xlim[0])
                for upper, xlim in zip(limits, xlimits)):
            raise ValueError(f"Provide four finite {name} energy upper limits above the lower limits or None")
        energy_overrides[name] = limits
    if any(surface not in ("up", "down") for surface in surfaces):
        raise ValueError("Surface must be up or down")
    loaded = [load_case(float(temperature), float(tau)) for temperature, tau in cases]
    audits = {tuple(case): audit for case, (_, audit) in zip(cases, loaded)}
    apply_mnras_style()
    figures = {}
    for surface in surfaces:
        sign = "+" if surface == "up" else "-"
        fig, axes = plt.subplots(4, 3, figsize=get_three_column_size(row_height_scale=3.5),
                                 sharex=False, sharey=False)
        for row, ((temperature, tau), xlim, (models, _)) in enumerate(zip(cases, xlimits, loaded)):
            panel_xlimits = tuple(
                xlim if energy_overrides[name][row] is None
                else (xlim[0], energy_overrides[name][row])
                for name in ("I", "Q", "Pi"))
            peak_i, q_bounds, pi_bounds = 0., [0.], [0.]
            for color, (window, _, _) in enumerate(WINDOWS):
                for _, spectrum, linestyle, is_mc in models:
                    selected = (spectrum["boundary"] == surface) & (spectrum["theta_label"] == window)
                    if is_mc:
                        selected &= spectrum["plot_supported"]
                    indices = np.flatnonzero(selected)
                    indices = indices[np.argsort(spectrum["E_center_keV"][indices])]
                    energy = spectrum["E_center_keV"][indices]
                    intensity = spectrum["I_N_per_keV_sr"][indices]
                    q = spectrum["Q_N_per_keV_sr"][indices]
                    energy_weight = AZIMUTH_FACTOR*energy
                    pi = np.divide(q, intensity, out=np.full_like(q, np.nan), where=intensity > 0)
                    xerr = np.vstack((energy-spectrum["E_left_keV"][indices],
                                      spectrum["E_right_keV"][indices]-energy))
                    for col, values in enumerate((energy_weight*intensity, energy_weight*q, 100*pi)):
                        ax = axes[row, col]
                        xmin, xmax = panel_xlimits[col]
                        visible = (energy >= xmin) & (energy <= xmax)
                        valid = visible & np.isfinite(values)
                        if col == 0:
                            valid &= values > 0
                        if col == 2:
                            if is_mc:
                                valid &= spectrum["count"][indices] >= pi_min_count
                            else:
                                valid &= intensity > max(float(intensity.max()), 0.) * pi_signal_floor
                        if is_mc:
                            error_field = ("I_N_se_per_keV_sr", "Q_N_se_per_keV_sr", "Pi_signed_se")[col]
                            errors = spectrum[error_field][indices] * (100 if col == 2 else energy_weight)
                            valid &= np.isfinite(errors) & (errors >= 0)
                            ax.errorbar(energy[valid], values[valid], xerr=xerr[:, valid],
                                        yerr=errors[valid], fmt="o", mfc=f"C{color}", ms=2.5,
                                        color=f"C{color}", alpha=.25, elinewidth=.6,
                                        capsize=0, zorder=4)
                        else:
                            errors = np.zeros_like(values)
                            ax.plot(energy, np.where(valid, values, np.nan), color=f"C{color}",
                                    ls=linestyle, lw=1.35, zorder=2)
                        if valid.any():
                            if col == 0:
                                peak_i = max(peak_i, float((values+errors)[valid].max()))
                            elif col == 1 or not is_mc:
                                bounds = q_bounds if col == 1 else pi_bounds
                                bounds.extend((float((values-errors)[valid].min()),
                                               float((values+errors)[valid].max())))
            axes[row, 0].set_yscale("log")
            if peak_i > 0:
                lower = intensity_ymin[row]
                if lower is None:
                    lower = peak_i*10.**(-intensity_decades)
                if lower >= peak_i*1.4:
                    raise ValueError("Intensity lower limit must be below the upper limit")
                axes[row, 0].set_ylim(lower, peak_i*1.4)
            axes[row, 0].text(.5, .005,
                               rf"$kT_e={temperature:g}$ keV"+"\n"+rf"$\tau={tau:g}$",
                               transform=axes[row, 0].transAxes,
                               ha="center", va="bottom")
            for col, bounds in ((1, q_bounds), (2, pi_bounds)):
                lo, hi = min(bounds), max(bounds)
                margin = .10 * max(hi-lo, np.finfo(float).tiny)
                upper = q_ymax[row] if col == 1 else None
                lower = q_ymin[row] if col == 1 else None
                if upper is None:
                    upper = hi+margin
                if lower is None:
                    lower = lo-margin
                if col == 1 and surface in q_ylim_by_surface:
                    override = q_ylim_by_surface[surface][row]
                    if override is not None:
                        lower, upper = override
                if upper <= lower:
                    raise ValueError("Q upper limit must be above the lower limit")
                axes[row, col].set_ylim(lower, upper)
                axes[row, col].axhline(0, color=".55", lw=.65, zorder=0)
                axes[row, col].yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            axes[row, 1].yaxis.set_major_formatter(formatter)
            for col, ax in enumerate(axes[row]):
                ax.set_xscale("log")
                ax.set_xlim(*panel_xlimits[col])
                ax.grid(False)
                ax.tick_params(axis="y", pad=2)
                ax.set_ylabel("")
                if col == 2:
                    ax.annotate(r"$\%$", xy=(0, 1), xycoords="axes fraction",
                                xytext=(0, 3), textcoords="offset points",
                                ha="left", va="bottom", annotation_clip=False,
                                fontsize=ax.yaxis.get_offset_text().get_fontsize())
            # Every row has its own E range; keep all energy tick labels visible.
        for col, title in enumerate((rf"$I_E^{sign}(E)$", rf"$Q_E^{sign}(E)$", rf"$\Pi^{sign}(E)$")):
            axes[0, col].set_title(title, pad=8)
        angle_handles = [Line2D([], [], color=f"C{index}", label=f"{lo}–{hi}°")
                         for index, (_, lo, hi) in enumerate(WINDOWS)]
        axes[0, 1].legend(handles=angle_handles,
                          loc="upper right" if surface == "up" else "lower right", frameon=False,
                          handlelength=.9, handletextpad=.35, labelspacing=.15, borderaxespad=.25)
        method_handles = [Line2D([], [], color=".25", ls="-", label="recursion"),
                          Line2D([], [], color=".25", ls="--", label="compPSc"),
                          Line2D([], [], color=".25", ls="none", marker="o", mfc=".25",
                                 alpha=.25, ms=3., label="MCRT")]
        axes[0, 2].legend(handles=method_handles, loc="lower left", frameon=False,
                          handlelength=.9, handletextpad=.35, labelspacing=.15)
        fig.supxlabel(r"photon energy $E$ (keV)", y=.018)
        fig.subplots_adjust(left=.10, right=.98, top=.95, bottom=.075, wspace=.30, hspace=.34)
        if save_directory is not None:
            destination = Path(save_directory)
            destination.mkdir(parents=True, exist_ok=True)
            surface_name = "upper" if surface == "up" else "lower"
            fig.savefig(destination / f"four_case_energy_spectra_{surface_name}.pdf",
                        bbox_inches="tight", pad_inches=.04)
        figures[surface] = fig
        if show:
            plt.show()
            plt.close(fig)
    return figures, audits
