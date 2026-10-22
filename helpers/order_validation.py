"""Read cold-electron exact-order MCRT exports without launching simulations."""
import csv
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data/order_validation"
SOURCES = {"beam": "beam", "lambert": "lambert", "internal": "internal_iso"}
ORDERS = (1, 3, 6, 10)
MCRT_MIN_ANGULAR_BIN_COUNT = 30
MCRT_MIN_POLARIZATION_BIN_COUNT = 100


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _key_values(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return {row[0]: row[1] for row in csv.reader(stream)
                if len(row) == 2 and row[0] != "key"}


def _one(directory, suffix):
    paths = list(directory.glob("*_" + suffix + ".csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Restore the bundled exact-order MCRT cache: {directory.name}/{suffix}")
    return paths[0]


def load_order_cases():
    cases = {}
    for tau, events, seed_base in ((0.1, 5_000_000, 2026082700),
                                   (0.3, 5_000_000, 2026082800),
                                   (1.0, 3_000_000, 2026082900),
                                   (3.0, 1_000_000, 2026083000),
                                   (5.0, 1_000_000, 2026083100)):
        tag = f"{tau:g}".replace(".", "p")
        for source_index, (source, injection) in enumerate(SOURCES.items()):
            directory = DATA / injection / f"tau_{tag}"
            cfg = _key_values(_one(directory, "cfg"))
            summary = _key_values(_one(directory, "sum"))
            expected = {"mode": "run", "seed_photon_model": "monoenergetic",
                        "polarization_mode": "stokes", "geometry_model": "slab",
                        "transport_cross_section": "thomson", "electron_model": "monoenergetic",
                        "slab_injection_model": injection, "max_scatters": "32", "mu_bins": "36",
                        "num_events": str(events), "seed": str(seed_base + source_index)}
            for key, value in expected.items():
                if cfg.get(key) != value:
                    raise ValueError(f"MCRT configuration mismatch: {source}, tau={tau}, {key}")
            np.testing.assert_allclose(float(cfg["electron_gamma"]), 1.0)
            np.testing.assert_allclose(float(cfg["incident_photon_energy_mec2"]), 1e-6)
            np.testing.assert_allclose(float(cfg["slab_optical_depth"]), tau)
            assert int(summary["events_completed"]) == int(summary["events_requested"]) == events
            capped = int(summary["terminated_at_max_scatters"])
            escaped = int(summary["total_escaped_up"]) + int(summary["total_escaped_down"])
            assert escaped + capped == events
            pn = _rows(_one(directory, "Pn"))
            angular = {}
            for boundary, suffix in (("+", "du"), ("-", "dd")):
                for order in ORDERS:
                    rows = _rows(_one(directory, f"{suffix}_order_{order:03d}"))
                    record = next(row for row in pn if int(row["scatter_order"]) == order)
                    key = "escaped_up_count" if boundary == "+" else "escaped_down_count"
                    assert sum(int(row["count"]) for row in rows) == int(record[key])
                    angular[boundary, order] = rows
            cases[tau, source] = {"Pn": pn, "angular": angular,
                                  "stokes_angle": _rows(_one(directory, "stokes_order_angle")),
                                  "summary": summary, "events": events}
            print(f"MCRT cache: tau={tau:g}, {source}, histories={events:,}, "
                  f"terminated at n=32: {capped:,}; only n<=10 is used here.")
    return cases


def _pn_record(case, order):
    return next(row for row in case["Pn"] if int(row["scatter_order"]) == order)


def mcrt_escape_series(case, boundary, max_order, min_count=10):
    """Unconditional escape probabilities from integer counts, not rounded CSV P_n."""
    if boundary not in ("+", "-"):
        raise ValueError("boundary must be + or -")
    if max_order >= int(case['summary']['max_scatters']):
        raise ValueError("Requested order reaches the simulation scattering cap")
    orders = np.arange(max_order + 1)
    key = 'escaped_up_count' if boundary == '+' else 'escaped_down_count'
    counts = np.array([int(_pn_record(case, n)[key]) for n in orders])
    probability = counts / case['events']
    error = np.sqrt(probability * (1 - probability) / case['events'])
    return orders, probability, error, counts >= min_count


def mcrt_survival_series(case, max_order, min_count=10):
    """M_n = 1 - sum_{j=0}^{n-1}(P_j^+ + P_j^-), including capped survivors."""
    if max_order >= int(case['summary']['max_scatters']):
        raise ValueError("Requested order reaches the simulation scattering cap")
    escaped = np.array([
        int(_pn_record(case, j)['escaped_up_count']) + int(_pn_record(case, j)['escaped_down_count'])
        for j in range(max_order)
    ], dtype=np.int64)
    surviving = case['events'] - np.cumsum(escaped)
    assert np.all(surviving >= 0) and np.all(np.diff(surviving) <= 0)
    probability = surviving / case['events']
    error = np.sqrt(probability * (1 - probability) / case['events'])
    return np.arange(1, max_order + 1), probability, error, surviving >= min_count


def _mcrt_phi_points(case, boundary, order):
    rows = case["angular"][boundary, order]
    record = _pn_record(case, order)
    count = int(record["escaped_up_count" if boundary == "+" else "escaped_down_count"])
    mu = np.array([float(row["mu_center"]) for row in rows])
    counts = np.array([float(row["count"]) for row in rows])
    widths = np.array([float(row["mu_right"]) - float(row["mu_left"]) for row in rows])
    phi = np.divide(counts, count * widths, out=np.full_like(counts, np.nan),
                    where=(count > 0) & (widths > 0))
    return mu, phi, counts >= MCRT_MIN_ANGULAR_BIN_COUNT


def _mcrt_polarization_points(case, boundary, order):
    side = "up" if boundary == "+" else "down"
    rows = [row for row in case["stokes_angle"]
            if row["boundary"] == side and int(row["scatter_order"]) == order]
    mu = np.array([0.5 * (float(row["mu_min"]) + float(row["mu_max"])) for row in rows])
    counts = np.array([int(row["packet_count"]) for row in rows])
    # Raw archived MCRT Q is along the projected slab normal: reverse exactly once.
    pi = -np.array([float(row["Q_over_I"]) for row in rows])
    error = np.array([float(row["Q_over_I_standard_error"]) for row in rows])
    indices = np.argsort(mu)
    return mu[indices], pi[indices], error[indices], counts[indices] >= MCRT_MIN_POLARIZATION_BIN_COUNT
