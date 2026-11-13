"""Rebuild omitted boundary assets offline from the two public circuits.

Running --help or importing this helper does not construct tables or models.
decoder.py calls ensure_assets before importing the validated organizer entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
KNOWN_CIRCUITS = {
    "X": "dc6d415ae3f0b5396a7f63ba1047f024be8ce7e44b03ace65348bcee78b3c070",
    "Z": "36a24a0347bbb82362af28d816d790d5ee1cb1543fef2e9af7ada70536bf9968",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_assets(*, asset_root=None, threads=None):
    """Create missing tables only; existing tables are checked during loading."""
    # The validated temporal-chain loader resolves boundary tables relative to
    # its source tree. QEC_ASSET_ROOT may separately locate supplied row banks.
    asset_root = Path(asset_root or ROOT).resolve()
    output = asset_root/"data/target03"
    missing = [basis for basis in "XZ" if not (output/f"boundary21_{basis}.npz").is_file()]
    if not missing:
        return {"built": [], "asset_root": str(asset_root)}
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    requested = int(threads if threads is not None else os.environ.get("NUMBA_NUM_THREADS", min(48, os.cpu_count() or 1)))
    if not 1 <= requested <= 48:
        raise ValueError("Boundary bootstrap needs NUMBA_NUM_THREADS from1 to48")
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(requested)
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "GOTO_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import numba
    import stim
    from prior_model import PriorTiltDecoder
    from diagnostics.boundary21_tables import SCHEMA, build_tables

    numba.set_num_threads(min(requested, int(numba.config.NUMBA_NUM_THREADS)))
    sources = {name: _sha256(ROOT/name) for name in (
        "diagnostics/boundary21_tables.py", "diagnostics/temporal_factor_probe.py",
        "diagnostics/temporal_structure.py")}
    output.mkdir(parents=True, exist_ok=True)
    lock = output/".boundary_bootstrap.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Another boundary bootstrap holds the lock. Finish it before importing decoder; remove a stale lock only after confirming its process has ended.") from error
    os.close(descriptor)
    report = {"schema": "offline-publication-boundary-bootstrap-v1", "built": [],
              "asset_root": str(asset_root), "source_sha256": sources,
              "uses_logical_labels_or_network": False}
    try:
        for basis in missing:
            path = output/f"boundary21_{basis}.npz"
            if path.exists():
                continue
            started = time.perf_counter()
            print(f"Building missing {basis} boundary tables from the public circuit; initialization is untimed.", flush=True)
            circuit = stim.Circuit.from_file(str(asset_root/f"data/competition/{basis}_circuit.stim"))
            circuit_sha256 = hashlib.sha256(str(circuit).encode()).hexdigest()
            if circuit_sha256 != KNOWN_CIRCUITS[basis]:
                raise ValueError("Boundary bootstrap supports only the two validated public circuits")
            model = PriorTiltDecoder(circuit).model
            if model.h.shape[0] != 70 or model.logical.shape != (4, model.h.shape[1]):
                raise ValueError("Unexpected public-circuit detector/logical model dimensions")
            coordinates = circuit.get_detector_coordinates()
            times = np.asarray([coordinates[i][2] for i in range(circuit.num_detectors)])
            arrays, summaries = build_tables(model, times)
            for region, detector_bits in (("first", 21), ("middle2", 14), ("middle3", 14), ("last", 21)):
                probability, bias = arrays[region+"_probability"], arrays[region+"_bias"]
                if (len(arrays[region+"_detectors"]) != detector_bits
                        or probability.shape != (1 << detector_bits,)
                        or bias.shape != (1 << detector_bits, 4)
                        or not np.all(np.isfinite(probability)) or not np.all(np.isfinite(bias))
                        or not np.isclose(probability.sum(), 1., rtol=0., atol=1e-9)
                        or np.min(probability) < -1e-10
                        or np.max(np.abs(bias)-probability[:, None]) > 1e-10):
                    raise ValueError("Rebuilt boundary dimensions, finite masses or probability normalizers are invalid")
            metadata = {"schema": SCHEMA, "basis": basis, "source_sha256": sources,
                "circuit_sha256": circuit_sha256,
                "regions": ["first", "middle2", "middle3", "last"], "summaries": summaries,
                "minimum_probability": 1e-12, "num_variables": model.h.shape[1],
                "outside_variables_first_last": len(arrays["outside_columns"]),
                "outside_variables_all_four": len(arrays["outside_all_columns"])}
            temporary = path.with_name(path.name+f".bootstrap-{os.getpid()}")
            try:
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            report["built"].append({"basis": basis, "relative_path": str(path.relative_to(asset_root)),
                "bytes": path.stat().st_size, "sha256": _sha256(path),
                "seconds": time.perf_counter()-started})
            del arrays, model
        return report
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=min(48, os.cpu_count() or 1))
    args = parser.parse_args()
    print(json.dumps(ensure_assets(asset_root=args.asset_root, threads=args.threads), indent=2))


if __name__ == "__main__":
    main()
