"""Offline display of the audited 2026-10-14 profile-reference best points."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .western_renderer import plot_western_iqu_comparison as _render

CACHE = Path(__file__).resolve().parents[1]/'data/western'


def load_verified_fits():
    manifest = json.loads((CACHE/'manifest.json').read_text())
    assert manifest['version'] == 'profile-reference-20261014'
    assert manifest['data_version'] == 'standard-v4-20261007'
    assert manifest['refitted'] is False and manifest['protected_inputs_unchanged']
    for name, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest() == digest, name
    audit = json.loads((CACHE/'parameter-intervals.json').read_text())
    fits = {}
    for branch in ('low', 'high'):
        cached = json.loads((CACHE/(branch+'.json')).read_text())
        result, point = cached['result'], cached['point']
        assert cached['audit']['full_precision_replay'] and cached['audit']['actual_loaded_files_checked']
        assert cached['audit']['no_optimizer_called'] and result['refitted'] is False
        assert result['branch_start'] == branch and result['data_points'] == 1596
        assert result['free_parameters'] == 1596-result['dof'] == (24 if branch == 'low' else 23)
        assert point['chi2'] == result['chi2'] == audit[branch]['reference_chi2']
        assert point['dof'] == result['dof'] == audit[branch]['dof']
        for row in audit[branch]['rows']:
            key = re.fullmatch(r'g(\d+)-p(\d+)', row['key'])
            if key:
                g, i = map(int, key.groups())
                np.testing.assert_allclose(point['groups'][g-1][i-1]['values'][0], row['best'], rtol=0, atol=1e-12)
        counts, by = {}, {}
        for stokes in 'IQU':
            rows = [r for r in cached['native'] if r['stokes'] == stokes]
            counts[stokes] = sum(len(r['data']) for r in rows)
            by[stokes] = 0.
            for row in rows:
                residual = (np.asarray(row['data'])-row['model'])/row['error']
                np.testing.assert_allclose(residual, row['residual_sigma'], atol=1e-10)
                by[stokes] += float(residual@residual)
            np.testing.assert_allclose(by[stokes], result['chi2_by_stokes'][stokes], atol=1e-6, rtol=0)
        assert counts == cached['counts'] == dict(I=704, Q=446, U=446)
        np.testing.assert_allclose(sum(by.values()), result['chi2'], atol=1e-4, rtol=0)
        curve = cached['unfolded']['curve']
        assert len(curve['components']) == (3 if branch == 'low' else 2)
        np.testing.assert_allclose(np.sum(curve['components'], axis=0), curve['total'], rtol=1e-8, atol=1e-12)
        for stokes in 'QU':
            display, curve = cached['display'][stokes], cached['curves'][stokes]
            assert len(display['data']) == 6
            np.testing.assert_allclose(display['edges_keV'], np.arange(2., 9.))
            covariance = np.asarray(display['covariance'])
            assert np.linalg.eigvalsh(covariance).min() >= -1e-16
            np.testing.assert_allclose(np.sqrt(np.diag(covariance)), display['error'])
            for key in ('data', 'model', 'covariance'):
                np.testing.assert_allclose(sum(np.asarray(p[key]) for p in display['DU_contributions']), display[key])
            np.testing.assert_allclose((np.asarray(display['data'])-display['model'])/display['error'], display['residual_sigma'], atol=1e-10)
            np.testing.assert_allclose(np.sum(curve['per_DU_models'], axis=0), curve['model'])
        for folded, unfolded in zip(cached['i_folded'], cached['i_display']):
            np.testing.assert_array_equal(folded['energy'], unfolded['energy'])
            np.testing.assert_array_equal(folded['halfwidth'], unfolded['halfwidth'])
            np.testing.assert_allclose((np.asarray(folded['data'])-folded['model'])/folded['error'], folded['residual_sigma'], atol=1e-10)
        fits[branch] = cached
    # The new best points change predictions/unfolding, never the measured rates.
    for key in ('native', 'i_folded'):
        for low, high in zip(fits['low'][key], fits['high'][key]):
            assert low['group'] == high['group']
            for field in ('energy', 'halfwidth', 'data', 'error'):
                np.testing.assert_array_equal(low[field], high[field])
    return fits


def plot_western_iqu_comparison(show=True, save_path=None):
    """Retain the established layout, legends, units and display grouping."""
    return _render(show=show, save_path=save_path, verified_fits=load_verified_fits())
