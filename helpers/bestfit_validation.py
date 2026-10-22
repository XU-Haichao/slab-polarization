"""Display-only best-fit coronal benchmarks from bundled numerical exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_style import apply_mnras_style, get_double_column_size

BASE = Path(__file__).resolve().parents[1]/'data/bestfit_validation'

def load_cached_branch(branch):
    if branch not in ('low', 'high'):
        raise ValueError('Expected low or high reflection branch')
    out = BASE/branch
    parameters = json.loads((BASE/'parameters.json').read_text())[branch]
    manifest = json.loads((out/'production_manifest.json').read_text())
    audit = json.loads((out/'jones_audit.json').read_text())
    budget = json.loads((out/'mcrt_photon_budget.json').read_text())
    if manifest['parameters'] != parameters or audit['parameters'] != parameters:
        raise ValueError('Benchmark parameter mismatch')
    # Tie these benchmarks to the fitted spectra displayed immediately above.
    fitted = json.loads((BASE.parent/'western'/f'{branch}.json').read_text())
    group = fitted['point']['groups'][0]
    for key,name in (('kTe_keV','kTe'), ('tau','tau'), ('inclination_deg','incl')):
        value = next(p['values'][0] for p in group if p['name']==name)
        np.testing.assert_allclose(value,parameters[key],rtol=0,atol=1e-12)
    histories = manifest['total_histories']
    assert manifest['completed_batches'] == manifest['requested_batches'] == 128
    assert len(manifest['runs']) == 128
    assert len({r['seed'] for r in manifest['runs']}) == 128
    assert sum(r['events'] for r in manifest['runs']) == histories == 128000000
    assert budget['injected'] == budget['escaped'] == histories
    assert budget['terminated'] == 0 and budget['seeds_unique']
    mc = pd.read_csv(out/'mcrt_rebin4.csv')
    model = pd.read_csv(out/'jones_n16384.csv')
    assert (mc.total_histories == histories).all()
    for frame in (mc,model):
        assert set(frame.boundary) == {'up','down'}
        assert {'th015','th045','th075','fit'} <= set(frame.theta_label)
        assert (frame.E_right_keV > frame.E_left_keV).all()
        active = frame.I_N_per_keV_sr > 0
        np.testing.assert_allclose(frame.loc[active,'Pi_signed'],
            frame.loc[active,'Q_N_per_keV_sr']/frame.loc[active,'I_N_per_keV_sr'],
            rtol=1e-10,atol=1e-12)
    return mc,model

def plot_cached_branch(branch):
    mc,model = load_cached_branch(branch)
    apply_mnras_style()
    return plot_broad(branch,mc,model,show=True,save_figures=False)

def draw(ax, model, mc, side, window, stokes, color):
    a = model[(model.boundary == side) & (model.theta_label == window)]
    m = mc[(mc.boundary == side) & (mc.theta_label == window)]
    m = m[m.plot_supported & m.E_center_keV.between(.004, 40)]
    if stokes == 'Pi':
        ax.plot(a.E_center_keV, 100*a.Pi_signed, color=color)
        yy, ee = 100*m.Pi_signed, 100*m.Pi_signed_se
    else:
        ax.plot(a.E_center_keV, a.E_center_keV**2*a[f'{stokes}_N_per_keV_sr'], color=color)
        yy = m.E_center_keV**2*m[f'{stokes}_N_per_keV_sr']
        ee = m.E_center_keV**2*m[f'{stokes}_N_se_per_keV_sr']
    ax.errorbar(m.E_center_keV, yy, yerr=ee,
        xerr=np.array([m.E_center_keV-m.E_left_keV, m.E_right_keV-m.E_center_keV]),
        fmt='o', ms=2.3, mfc='none', color=color, lw=.65, alpha=.8)
    ax.set_xscale('log'); ax.set_xlim(.004, 40.)
    if stokes == 'I':
        ax.set_yscale('log'); ax.set_ylim(bottom=1e-7)
    else:
        ax.axhline(0, color='.6', lw=.7, zorder=0)

def intensity_limits(ax, model, mc, side, windows, low=.004, high=40.):
    a = model[(model.boundary == side) & model.theta_label.isin(windows) &
              model.E_center_keV.between(low, high)]
    m = mc[(mc.boundary == side) & mc.theta_label.isin(windows) & mc.plot_supported &
           mc.E_center_keV.between(low, high)]
    peak = max((a.E_center_keV**2*a.I_N_per_keV_sr).max(),
               (m.E_center_keV**2*(m.I_N_per_keV_sr+m.I_N_se_per_keV_sr)).max())
    ax.set_ylim(1e-7, peak*2.5)

def plot_broad(branch, mc, model, *, show=False, save_figures=False):
    fig, axes = plt.subplots(2, 2, figsize=get_double_column_size(2), sharex=True)
    for r, side in enumerate(('up', 'down')):
        for c, stokes in enumerate(('I', 'Q')):
            for k, window in enumerate(('th015', 'th045', 'th075')):
                draw(axes[r, c], model, mc, side, window, stokes, f'C{k}')
            axes[r, c].set_ylabel(rf'$E^2 {stokes}_E$ (keV photon$^{{-1}}$ sr$^{{-1}}$)')
            axes[r, c].text(.04, .94, 'upper' if side == 'up' else 'lower',
                                va='top', transform=axes[r, c].transAxes)
            if stokes == 'I':
                intensity_limits(axes[r, c], model, mc, side, ('th015','th045','th075'))
            else:
                axes[r, c].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    colors = [Line2D([], [], color=f'C{k}', label=label) for k, label in enumerate(
        (r'$0^\circ$–$30^\circ$', r'$30^\circ$–$60^\circ$', r'$60^\circ$–$90^\circ$'))]
    axes[0, 1].legend(handles=colors, loc='lower left', frameon=False)
    axes[0, 0].legend(handles=[Line2D([], [], color='.2', label='recursion + Jones'),
        Line2D([], [], color='.2', marker='o', mfc='none', ls='none', label='MCRT')],
        loc='lower left', frameon=False)
    for ax in axes[-1]: ax.set_xlabel('photon energy (keV)')
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes

def plot_cached_polarization(branch, *, show=True, save_figures=False):
    """Signed Q/I in percent; cached batch SE retains the I/Q covariance."""
    mc, model = load_cached_branch(branch)
    apply_mnras_style()
    fig, axes = plt.subplots(1, 2, figsize=get_double_column_size(), sharex=True)
    # Emphasize the model variation, rather than rare high-energy MC outliers.
    # All cached points/errors are retained; the axes clip only their display.
    pi_ylim_percent = {'up': (-1., 5.), 'down': (-1., 4.)}
    for ax, side in zip(axes, ('up', 'down')):
        for k, window in enumerate(('th015', 'th045', 'th075')):
            draw(ax, model, mc, side, window, 'Pi', f'C{k}')
        ax.set_xlabel('photon energy (keV)')
        ax.set_ylabel(r'polarization fraction $\Pi(E)$ (%)')
        ax.set_ylim(*pi_ylim_percent[side])
        ax.text(.04, .94, 'upper' if side == 'up' else 'lower',
                va='top', transform=ax.transAxes)
    axes[0].legend(handles=[Line2D([], [], color='.2', label='recursion + Jones'),
        Line2D([], [], color='.2', marker='o', mfc='none', ls='none', label='MCRT')],
        loc='center left', bbox_to_anchor=(.02, .52), frameon=False,
        handlelength=1.5, labelspacing=.25)
    colors = [Line2D([], [], color=f'C{k}', label=label) for k, label in enumerate(
        (r'$0^\circ$–$30^\circ$', r'$30^\circ$–$60^\circ$', r'$60^\circ$–$90^\circ$'))]
    axes[1].legend(handles=colors, loc='upper left', bbox_to_anchor=(.02, .86),
                   borderaxespad=.1, frameon=False, handlelength=1.5, labelspacing=.25)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
