"""Cache-only all-history scattering-count statistics for the public companion."""
import csv
from pathlib import Path
import numpy as np


def load_mean_case(directory, tau, source, events, seed, max_scatters):
    directory = Path(directory)

    def rows(suffix):
        paths = list(directory.glob('*_' + suffix + '.csv'))
        if len(paths) != 1:
            raise FileNotFoundError(f'Restore the bundled mean-validation cache: {directory}/{suffix}')
        with paths[0].open(newline='') as stream:
            return list(csv.DictReader(stream))

    config = {row['key']: row['value'] for row in rows('cfg')}
    summary = {row['key']: row['value'] for row in rows('sum')}
    expected = dict(polarization_mode='stokes', geometry_model='slab',
                    transport_cross_section='thomson', electron_model='monoenergetic',
                    seed_photon_model='monoenergetic', slab_injection_model=source)
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'{directory}: unexpected {key}')
    for key, value in dict(num_events=events, seed=seed, max_scatters=max_scatters).items():
        if int(config[key]) != value:
            raise ValueError(f'{directory}: unexpected {key}')
    for key, value in dict(electron_gamma=1., incident_photon_energy_mec2=1e-6,
                           slab_optical_depth=tau, seed_stokes_I=1.).items():
        if not np.isclose(float(config[key]), value, rtol=5e-5, atol=0):
            raise ValueError(f'{directory}: unexpected {key}')
    if any(float(config['seed_stokes_'+s]) != 0 for s in ('Q','U','V')):
        raise ValueError('Expected an unpolarized seed source')
    n = int(summary['events_completed'])
    terminated = int(summary['terminated_at_max_scatters'])
    if n != events or int(summary['events_requested']) != n:
        raise ValueError('Incomplete simulation cache')
    if int(summary['total_escaped_up']) + int(summary['total_escaped_down']) + terminated != n:
        raise ValueError('Photon bookkeeping does not close')
    histogram = rows('sc')
    order = np.array([int(row['scatter_count']) for row in histogram], dtype=np.int64)
    count = np.array([int(row['count']) for row in histogram], dtype=np.int64)
    if count.sum() != n or np.any(count < 0) or np.any(order < 0) or np.any(order > max_scatters):
        raise ValueError('Histogram must include all injected histories, including capped ones')
    if int(order @ count) != int(summary['total_scatter_interactions']):
        raise ValueError('Scattering-count histogram and summary disagree')
    mean = float(order @ count) / n
    variance = float(((order-mean)**2) @ count) / (n-1)
    return dict(mean=mean, standard_error=np.sqrt(variance/n), events=n,
                termination_fraction=terminated/n, terminated=terminated,
                max_scatters=max_scatters)
