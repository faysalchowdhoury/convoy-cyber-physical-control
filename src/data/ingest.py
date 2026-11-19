"""Deterministic ingestion of real public vehicle data, with provenance logging.

Sources (all keyless, public):

* NGSIM I-80 trajectories -- USDOT ITS DataHub Socrata resource ``8ect-6jqj``.
  Real 10 Hz vehicle trajectories with native leader/follower linkage
  (``preceding`` / ``following``) and ``space_headway`` / ``time_headway``.
* VED (Vehicle Energy Dataset, U. Michigan) -- real on-road OBD / powertrain
  telemetry, shipped as 7z archives in the ``gsoh/VED`` GitHub repo.

Nothing here is synthetic. NGSIM native units are US feet; they are converted to
SI metres on ingest. VED speeds are km/h, converted to m/s.

Everything is cached under ``data/raw`` and a provenance record is written to
``data/manifest.json`` (source URLs, row/vehicle counts, sampling rates).
"""
from __future__ import annotations

import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

FT_TO_M = 0.3048
KMH_TO_MS = 1.0 / 3.6

NGSIM_COLUMNS = [
    "vehicle_id", "frame_id", "global_time", "local_x", "local_y",
    "v_length", "v_vel", "v_acc", "v_class", "lane_id",
    "preceding", "following", "space_headway", "time_headway",
]


@dataclass
class Manifest:
    """Provenance of every real source pulled, serialised to manifest.json."""
    sources: dict = field(default_factory=dict)

    def add(self, name: str, **meta: Any) -> None:
        meta["retrieved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sources[name] = meta

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# NGSIM
# --------------------------------------------------------------------------- #
def _ngsim_page(cfg: dict, offset: int, timeout: int = 120) -> list[dict]:
    params = {
        "$select": ",".join(NGSIM_COLUMNS),
        "$where": f"location='{cfg['location']}'",
        "$order": "vehicle_id,frame_id",
        "$limit": cfg["page"],
        "$offset": offset,
    }
    url = cfg["base_url"] + "?" + urllib.parse.urlencode(params, safe="(),'=")
    req = urllib.request.Request(url, headers={"User-Agent": "convoy-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_ngsim(cfg: dict, manifest: Manifest | None = None):
    """Pull a bounded NGSIM I-80 slice to a cached csv.gz and return a DataFrame.

    Units are converted to SI metres on the way in.
    """
    import pandas as pd

    ncfg = cfg["ngsim"]
    out = Path(cfg["cache_dir"]) / "ngsim" / f"{ncfg['location']}_raw.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not out.exists():
        import csv
        n = 0
        with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=NGSIM_COLUMNS)
            writer.writeheader()
            offset = 0
            while n < ncfg["max_rows"]:
                rows = None
                for attempt in range(4):
                    try:
                        rows = _ngsim_page(ncfg, offset)
                        break
                    except Exception as exc:  # transient network / 5xx
                        if attempt == 3:
                            raise
                        time.sleep(3 * (attempt + 1))
                if not rows:
                    break
                for row in rows:
                    writer.writerow({c: row.get(c, "") for c in NGSIM_COLUMNS})
                n += len(rows)
                if len(rows) < ncfg["page"]:
                    break
                offset += ncfg["page"]

    df = pd.read_csv(out)
    num = ["frame_id", "global_time", "local_x", "local_y", "v_length",
           "v_vel", "v_acc", "v_class", "lane_id", "preceding", "following",
           "space_headway", "time_headway", "vehicle_id"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["vehicle_id", "frame_id", "v_vel", "lane_id"])

    # feet -> metres (positions, lengths, headway) ; ft/s -> m/s ; ft/s^2 -> m/s^2
    for c in ("local_x", "local_y", "v_length", "space_headway"):
        df[c] = df[c] * FT_TO_M
    df["v_vel"] = df["v_vel"] * FT_TO_M
    df["v_acc"] = df["v_acc"] * FT_TO_M
    df = df.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)

    if manifest is not None:
        manifest.add(
            "ngsim_i80",
            url=ncfg["base_url"], dataset_id=ncfg["dataset_id"],
            location=ncfg["location"], dt_s=ncfg["dt"],
            rows=int(len(df)), vehicles=int(df["vehicle_id"].nunique()),
            units="SI (converted from NGSIM feet)",
            note="Real 10 Hz trajectories with native preceding/following linkage.",
        )
    return df


# --------------------------------------------------------------------------- #
# VED
# --------------------------------------------------------------------------- #
VED_USE_COLS = [
    "VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]",
    "Engine RPM[RPM]", "Absolute Load[%]", "Fuel Rate[L/hr]",
    "HV Battery Current[A]", "HV Battery Voltage[V]", "HV Battery SOC[%]",
]


def fetch_ved(cfg: dict, manifest: Manifest | None = None):
    """Download + extract a bounded VED slice to a cached csv.gz; return DataFrame.

    Columns are the real OBD / powertrain signals used by the energy model and the
    cross-layer onboard-consistency check.
    """
    import pandas as pd

    vcfg = cfg["ved"]
    raw_dir = Path(cfg["cache_dir"]) / "ved"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed = raw_dir / "ved_slice.csv.gz"

    if not processed.exists():
        archive = raw_dir / "VED_DynamicData_Part1.7z"
        if not archive.exists():
            req = urllib.request.Request(vcfg["archive_url"],
                                         headers={"User-Agent": "convoy-research/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(archive, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        import py7zr
        import tempfile
        tmp = tempfile.mkdtemp()
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extract(path=tmp, targets=[vcfg["week_file"]])
        wk = pd.read_csv(os.path.join(tmp, vcfg["week_file"]))
        wk = wk[[c for c in VED_USE_COLS if c in wk.columns]].copy()
        # keep a deterministic slice of real vehicles with enough samples
        counts = wk.groupby("VehId").size()
        keep = counts[counts >= vcfg["min_trip_samples"]].index[: vcfg["n_vehicles"]]
        wk = wk[wk["VehId"].isin(keep)].reset_index(drop=True)
        wk.to_csv(processed, index=False)
        os.remove(os.path.join(tmp, vcfg["week_file"]))

    df = pd.read_csv(processed)
    df["speed_ms"] = df["Vehicle Speed[km/h]"] * KMH_TO_MS

    if manifest is not None:
        manifest.add(
            "ved",
            url=vcfg["archive_url"], week_file=vcfg["week_file"],
            rows=int(len(df)), vehicles=int(df["VehId"].nunique()),
            signals=[c for c in VED_USE_COLS if c in df.columns],
            note="Real on-road OBD/powertrain telemetry (U. Michigan VED).",
        )
    return df


def load_can_dropin(cfg: dict):
    """Return real labeled CAN frames if the optional HCRL drop-in is present.

    Looks for CSV files in ``can.manual_dir``. Returns ``None`` when absent, in
    which case the cross-layer IDS runs on real VED OBD streams with injected,
    labeled bus attacks (disclosed in the README).
    """
    import pandas as pd

    d = Path(cfg["can"]["manual_dir"])
    if not d.exists():
        return None
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        return None
    frames = [pd.read_csv(p) for p in csvs]
    return pd.concat(frames, ignore_index=True)
