"""Display the saved fixed-parameter chi-square scan; never rerun solvers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from plot_style import apply_mnras_style, get_double_column_size

DATA = Path(__file__).resolve().parents[1]/'data/statistics'
GROUPS = [('up', 'th000_045', 0, 45), ('up', 'th045_090', 45, 90),
          ('down', 'th000_045', 0, 45), ('down', 'th045_090', 45, 90)]
MODEL_NAMES = {'jones': 'recursion + Jones', 'comppsc': 'compPSc (base mesh)'}


def load_cached_statistics():
    path = DATA/'chi_square_fixed_parameters.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing completed scan: {path}; no automatic simulation is started.')
    data = pd.read_csv(path, float_precision='round_trip')
    provenance = json.loads((DATA/'provenance.json').read_text())
    assert len(data) == 648 and not data.duplicated(['model', 'case', 'boundary', 'window']).any()
    assert set(data.model) == set(MODEL_NAMES)
    for stokes in 'IQ':
        assert (data[f'N_{stokes}'] > 0).all()
        np.testing.assert_array_equal(data[f'dof_{stokes}'], data[f'N_{stokes}'])
        assert (data[f'fitted_parameters_{stokes}'] == 0).all()
        np.testing.assert_allclose(data[f'chi2_reduced_{stokes}'], data[f'chi2_{stokes}']/data[f'N_{stokes}'], rtol=1e-13)
        counts = data.pivot(index=['case', 'boundary', 'window'], columns='model', values=f'N_{stokes}')
        np.testing.assert_array_equal(counts.jones, counts.comppsc)
    return data, provenance


def _log_edges(values):
    mid = np.sqrt(values[:-1]*values[1:])
    return np.r_[values[0]**2/mid[0], mid, values[-1]**2/mid[-1]]


def plot_cached_chi_square(*, save_directory=None, show=True):
    data, provenance = load_cached_statistics()
    apply_mnras_style()
    temperatures = np.sort(data.kTe_keV.unique())
    taus = np.sort(data.tau.unique())
    assert len(temperatures) == len(taus) == 9
    xe, ye = _log_edges(taus), _log_edges(temperatures)
    norms = {}
    for stokes in 'IQ':
        values = data[f'chi2_reduced_{stokes}'].to_numpy()
        assert np.isfinite(values).all() and (values > 0).all()
        norms[stokes] = LogNorm(10**np.floor(np.log10(min(1., values.min()))),
                               10**np.ceil(np.log10(max(1., values.max()))))
    figures = {}
    for model, name in MODEL_NAMES.items():
        fig, axes = plt.subplots(2, 4, figsize=get_double_column_size(row_height_scale=1.7), sharex=True, sharey=True)
        for row, stokes in enumerate('IQ'):
            for col, (side, window, lo, hi) in enumerate(GROUPS):
                selected = data[(data.model == model) & (data.boundary == side) & (data.window == window)]
                grid = selected.pivot(index='kTe_keV', columns='tau', values=f'chi2_reduced_{stokes}').reindex(index=temperatures, columns=taus).to_numpy()
                assert grid.shape == (9, 9) and np.isfinite(grid).all()
                ax = axes[row, col]
                image = ax.pcolormesh(xe, ye, grid, cmap='viridis', norm=norms[stokes], shading='flat')
                for iy, ix in np.argwhere(grid <= 1):
                    ax.add_patch(Rectangle((xe[ix], ye[iy]), xe[ix+1]-xe[ix], ye[iy+1]-ye[iy],
                                           fill=False, edgecolor='white', lw=.6))
                if row == 0:
                    ax.set_title(('upper' if side == 'up' else 'lower')+f'\n{lo}–{hi}°')
                ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(False)
                ax.set_xlim(.3, 20); ax.set_ylim(1, 40)
                ax.set_xticks([.3, 1, 3, 10, 20], labels=['0.3', '1', '3', '10', '20'])
                ax.set_yticks([1, 3, 10, 40], labels=['1', '3', '10', '40'])
                ax.tick_params(axis='both', which='minor', labelbottom=False, labelleft=False)
                ax.tick_params(axis='x', labelsize=10)
            cax = fig.add_axes([.89, .55 if row == 0 else .15, .018, .32])
            bar = fig.colorbar(image, cax=cax)
            bar.set_label(r'$\chi_'+stokes+r'^2/N_'+stokes+'$')
            bar.ax.axhline(1., color='.4', lw=.8)
        for ax in axes[1, 1:]:
            ax.get_xticklabels()[0].set_visible(False)
        fig.suptitle(f'{name} vs MCRT', y=.99, fontsize=12)
        fig.supxlabel(r'optical depth $\tau$', x=.5, y=.065)
        fig.supylabel(r'electron temperature $kT_e$ (keV)', x=.02, y=.53)
        fig.text(.5, .008, r'Fixed parameters and absolute normalization; diagonal covariance; white outline: $\chi^2/N\leq1$', ha='center', fontsize=8)
        fig.subplots_adjust(left=.11, right=.86, bottom=.15, top=.85, wspace=.1, hspace=.15)
        if save_directory is not None:
            directory = Path(save_directory)
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ('pdf', 'png'):
                fig.savefig(directory/f'{model}_vs_mcrt_reduced_chi_square.{suffix}', dpi=240, bbox_inches='tight', pad_inches=.05)
        figures[model] = fig
        if show:
            plt.show()
        else:
            plt.close(fig)
    return figures, data, provenance


def plot_upper_recursion_chi_square(*, save_directory=None, show=True, model='jones'):
    """Two columns, upper surface only; I/Q rows and no cell outlines."""
    if model not in MODEL_NAMES:
        raise ValueError(f'Unknown cached model: {model}')
    data, provenance = load_cached_statistics()
    apply_mnras_style()
    temperatures = np.sort(data.kTe_keV.unique())
    taus = np.sort(data.tau.unique())
    xe, ye = _log_edges(taus), _log_edges(temperatures)
    fig, axes = plt.subplots(2, 2, figsize=get_double_column_size(row_height_scale=1.6),
                             sharex=True, sharey=True)
    for row, stokes in enumerate('IQ'):
        # Preserve the full-map color range for direct comparison with the full maps.
        values = data[f'chi2_reduced_{stokes}'].to_numpy()
        norm = LogNorm(10**np.floor(np.log10(min(1., values.min()))),
                       10**np.ceil(np.log10(max(1., values.max()))))
        for col, (side, window, lo, hi) in enumerate(GROUPS[:2]):
            selected = data[(data.model == model) & (data.boundary == side) & (data.window == window)]
            grid = selected.pivot(index='kTe_keV', columns='tau', values=f'chi2_reduced_{stokes}').reindex(index=temperatures, columns=taus).to_numpy()
            assert grid.shape == (9, 9) and np.isfinite(grid).all() and (grid > 0).all()
            ax = axes[row, col]
            image = ax.pcolormesh(xe, ye, grid, cmap='viridis', norm=norm, shading='flat')
            # No Rectangle patches: omit the white cell outlines in this saved subset.
            if row == 0:
                ax.set_title(f'{lo}–{hi}°')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(False)
            ax.set_xlim(.3, 20); ax.set_ylim(1, 40)
            ax.set_xticks([.3, 1, 3, 10, 20], labels=['0.3', '1', '3', '10', '20'])
            ax.set_yticks([1, 3, 10, 40], labels=['1', '3', '10', '40'])
            ax.tick_params(axis='both', which='minor', labelbottom=False, labelleft=False)
        cax = fig.add_axes([.89, .55 if row == 0 else .15, .018, .32])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(r'$\chi_'+stokes+r'^2/N_'+stokes+'$')
        bar.ax.axhline(1., color='.4', lw=.8)
    fig.supxlabel(r'optical depth $\tau$', x=.5, y=.065)
    fig.supylabel(r'electron temperature $kT_e$ (keV)', x=.025, y=.53)
    fig.subplots_adjust(left=.12, right=.86, bottom=.15, top=.87, wspace=.12, hspace=.15)
    if save_directory is not None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ('png', 'pdf'):
            fig.savefig(directory/f'{model}_upper_vs_mcrt_reduced_chi_square.{suffix}',
                        dpi=240, bbox_inches='tight', pad_inches=.05)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, data, provenance


def plot_upper_comppsc_chi_square(*, show=True):
    """Display the compPSc counterpart with identical reference support and color scales."""
    return plot_upper_recursion_chi_square(model='comppsc', save_directory=None, show=show)
