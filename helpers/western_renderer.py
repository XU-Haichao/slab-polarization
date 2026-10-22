"""Display the verified standard-v4 Western IQU fits; never fit or run XSPEC."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

from plot_style import apply_mnras_style, get_double_column_size

CACHE = Path(__file__).resolve().parent / "data"
KEV_TO_ERG = 1.602176634e-9
LABELS = ("NuSTAR FPMA", "NuSTAR FPMB", "NICER", "IXPE (DU1–3)")


def _read(path):
    return json.loads(path.read_text())


def _residual(ax, energy, values, halfwidth, color, size=2.5):
    ax.errorbar(energy, values, yerr=np.ones(len(energy)), xerr=halfwidth,
                fmt="o", mfc="none", ms=size, color=color, elinewidth=0.55, alpha=0.75)


def plot_western_iqu_comparison(show=True, save_path=None, *, verified_fits=None):
    """Two columns: low/high reflection; I, Q, U each with detector residuals."""
    apply_mnras_style()
    if verified_fits is None:
        raise ValueError("Pass the verified public data exports")
    fits = verified_fits
    fig = plt.figure(figsize=get_double_column_size(row_height_scale=3.2))
    outer = fig.add_gridspec(3, 2, height_ratios=(1.5, 1, 1), hspace=0.16, wspace=0.12)
    axes = {}
    for col, branch in enumerate(("low", "high")):
        cached = fits[branch]
        result = cached["result"]
        for row, stokes in enumerate(("I", "Q", "U")):
            pair = outer[row, col].subgridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.02)
            ax = fig.add_subplot(pair[0])
            residual_ax = fig.add_subplot(pair[1], sharex=ax)
            axes[branch, stokes] = (ax, residual_ax)
            if stokes == "I":
                # IXPE I uses derived response-aware optmin/25 display groups;
                # NuSTAR/NICER retain their native optmin/25 groups. All three
                # IXPE source fluxes are overlaid, not added. Fits remain native.
                for number, data in enumerate(cached["i_display"]):
                    color = f"C{min(number, 3)}"
                    calibration = cached["point"]["groups"][data["group"]-1][0]["values"][0]
                    ax.errorbar(data["energy"], np.asarray(data["data"])*KEV_TO_ERG/calibration,
                                yerr=np.asarray(data["error"])*KEV_TO_ERG/calibration,
                                xerr=data["halfwidth"], fmt=".", ms=2.5,
                                elinewidth=0.55, alpha=0.7, color=color)
                    folded = next(r for r in cached["i_folded"] if r["group"] == data["group"])
                    np.testing.assert_array_equal(data["energy"], folded["energy"])
                    np.testing.assert_array_equal(data["halfwidth"], folded["halfwidth"])
                    _residual(residual_ax, folded["energy"], folded["residual_sigma"],
                              folded["halfwidth"], color, size=1.8)
                curve = cached["unfolded"]["curve"]
                energy = np.asarray(curve["energy"])
                ax.plot(energy, np.asarray(curve["total"])*KEV_TO_ERG, color="0.15", lw=1.5)
                names = ("corona", "blackbody", "reflection") if branch == "low" else ("corona", "reflection")
                styles = {"corona": ("C4", "--"), "blackbody": ("C5", ":"), "reflection": ("C6", "-.")}
                for values, name in zip(curve["components"], names):
                    color, style = styles[name]
                    ax.plot(energy, np.asarray(values)*KEV_TO_ERG, color=color, ls=style, lw=1.3)
                ax.set(xscale="log", yscale="log", xlim=(1.5, 22))
                ax.set_title(f"{branch.capitalize()} reflection\n"
                             + rf"$\chi^2/\mathrm{{dof}}={result['chi2']:.2f}/{result['dof']}$", pad=6)
                ticks = [2, 3, 5, 8, 12, 20]
                residual_ax.xaxis.set_major_locator(FixedLocator(ticks))
                residual_ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
                residual_ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                display = cached["display"][stokes]
                curve = cached["curves"][stokes]
                energy = np.asarray(display["energy_keV"])
                halfwidth = np.diff(display["edges_keV"])/2
                ax.errorbar(energy, np.asarray(display["data"])*1e3,
                            yerr=np.asarray(display["error"])*1e3, xerr=halfwidth,
                            fmt="o", mfc="none", ms=3.8, elinewidth=0.8, color="C3")
                ax.plot(curve["energy_keV"], np.asarray(curve["model"])*1e3,
                        color="0.15", lw=1.5)
                _residual(residual_ax, energy, display["residual_sigma"], halfwidth, "C3", size=3.2)
                ax.axhline(0, color="0.6", lw=0.65)
                ax.set(xlim=(1.95, 8.05), xticks=np.arange(2, 9))
                ax.text(0.97, 0.95, rf"$\chi^2_{stokes}={result['chi2_by_stokes'][stokes]:.2f}$",
                        transform=ax.transAxes, va="top", ha="right", fontsize=12)
            if col == 0:
                ax.set_ylabel(r"$E F_E$ (erg cm$^{-2}$ s$^{-1}$)" if stokes == "I" else
                              rf"{stokes} ($10^{{-3}}$)" + "\n" + r"(count s$^{-1}$ keV$^{-1}$)")
                residual_ax.set_ylabel(r"$\Delta\chi$")
            ax.tick_params(labelbottom=False)
            residual_ax.axhline(0, color="0.45", lw=0.65)
            for level in (-3, 3):
                residual_ax.axhline(level, color="0.75", lw=0.6, ls=":")
            residual_ax.set_ylim(-4.2, 4.2)
            residual_ax.set_yticks([-3, 0, 3])
            for axis in (ax, residual_ax):
                axis.grid(False)
                axis.tick_params(axis="y", pad=2)
    # Matching physical scales across branches; no clipping of native I residuals.
    for stokes in ("I", "Q", "U"):
        left, right = [axes[b, stokes][0] for b in ("low", "high")]
        if stokes == "I":
            bounds, upper = [], []
            for cached in fits.values():
                curve = cached["unfolded"]["curve"]
                energy, total = np.asarray(curve["energy"]), np.asarray(curve["total"])*KEV_TO_ERG
                bounds.extend(total[(energy >= 1.5) & (energy <= 22) & (total > 0)])
                for data in cached["i_display"]:
                    cal = cached["point"]["groups"][data["group"]-1][0]["values"][0]
                    upper.extend((np.asarray(data["data"])+data["error"])*KEV_TO_ERG/cal)
            limits = min(bounds)/8, max(max(bounds), max(upper))*1.5
        else:
            extrema = [0.]
            for cached in fits.values():
                display = cached["display"][stokes]
                y, error = np.asarray(display["data"])*1e3, np.asarray(display["error"])*1e3
                extrema.extend(y-error)
                extrema.extend(y+error)
                extrema.extend(np.asarray(cached["curves"][stokes]["model"])*1e3)
            pad = (max(extrema)-min(extrema))*0.15
            limits = min(extrema)-pad, max(extrema)+pad
        left.set_ylim(limits)
        right.set_ylim(limits)
        right.tick_params(labelleft=False)
        axes["high", stokes][1].tick_params(labelleft=False)
        if stokes == "I":
            largest = max(max(abs(np.asarray(r["residual_sigma"])))
                          for cached in fits.values() for r in cached["i_folded"])
            for branch in fits:
                axes[branch, stokes][1].set_ylim(-max(4.2, largest+1), max(4.2, largest+1))
    # Split the shared keys between the unused regions of the two I panels.
    instruments = [Line2D([], [], color=f"C{i}", marker=".", ls="", label=label)
                   for i, label in enumerate(LABELS)]
    models = [Line2D([], [], color="0.15", label="total model"),
              Line2D([], [], color="C4", ls="--", label="corona"),
              Line2D([], [], color="C5", ls=":", label="blackbody (low only)"),
              Line2D([], [], color="C6", ls="-.", label="reflection")]
    axes["low", "I"][0].legend(
        handles=instruments, loc="lower center", bbox_to_anchor=(0.55, 0.035),
        frameon=False, handlelength=0.8, handletextpad=0.45, labelspacing=0.2,
        borderaxespad=0.3,
    )
    axes["high", "I"][0].legend(
        handles=models, loc="lower center", bbox_to_anchor=(0.50, 0.035),
        frameon=False, handlelength=1.2, handletextpad=0.5, labelspacing=0.2,
        borderaxespad=0.3,
    )
    fig.supxlabel("photon energy (keV)", x=0.56, y=0.015)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.057, top=0.935)
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches="tight", pad_inches=0.04)
    if show:
        plt.show()
        plt.close(fig)
    for branch, cached in fits.items():
        r = cached["result"]
        print(f"{branch}: chi2/dof={r['chi2']:.6f}/{r['dof']}; "
              f"native I/Q/U points={cached['counts']}; "
              f"IXPE I optmin/25 display bins={cached['ixpe_display_counts']}; standard-v4-20261007.")
    return fig, {branch: c["result"] for branch, c in fits.items()}
