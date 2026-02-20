"""Download and filter OSM waterway data from Geofabrik for SWORD regions."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

GEOFABRIK_URLS: dict[str, str] = {
    "NA": "https://download.geofabrik.de/north-america-latest.osm.pbf",
    "SA": "https://download.geofabrik.de/south-america-latest.osm.pbf",
    "EU": "https://download.geofabrik.de/europe-latest.osm.pbf",
    "AF": "https://download.geofabrik.de/africa-latest.osm.pbf",
    "AS": "https://download.geofabrik.de/asia-latest.osm.pbf",
    "OC": "https://download.geofabrik.de/australia-oceania-latest.osm.pbf",
}

ALL_REGIONS = list(GEOFABRIK_URLS.keys())


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
    else:
        mb = downloaded / 1e6
        print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)


def download_pbf(region: str, output_dir: Path) -> Path:
    """Download a Geofabrik PBF for the given SWORD region.

    Skips download if the file already exists and is non-empty.
    """
    region = region.upper()
    if region not in GEOFABRIK_URLS:
        raise ValueError(
            f"Unknown region '{region}'. Valid regions: {', '.join(ALL_REGIONS)}"
        )

    url = GEOFABRIK_URLS[region]
    pbf_dir = output_dir / "pbf"
    pbf_dir.mkdir(parents=True, exist_ok=True)
    dest = pbf_dir / f"{region.lower()}-latest.osm.pbf"

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{region}] PBF already exists: {dest}")
        return dest

    print(f"[{region}] Downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()  # newline after progress
    print(f"[{region}] Saved to {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def filter_waterways(pbf_path: Path, output_path: Path) -> Path:
    """Filter a PBF to only waterway=river ways using osmium.

    Skips if the output file already exists and is non-empty.
    """
    if not shutil.which("osmium"):
        raise FileNotFoundError(
            "osmium not found on PATH. Install with: "
            "apt install osmium-tool  /  brew install osmium-tool"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Filtered PBF already exists: {output_path}")
        return output_path

    print(f"  Filtering waterways from {pbf_path.name}")
    subprocess.run(
        [
            "osmium",
            "tags-filter",
            str(pbf_path),
            "w/waterway=river",
            "-o",
            str(output_path),
            "--overwrite",
        ],
        check=True,
    )
    print(f"  Filtered to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def pbf_to_gpkg(filtered_pbf: Path, output_path: Path) -> Path:
    """Convert a filtered PBF to GeoPackage (lines layer) using ogr2ogr.

    Overwrites existing output.
    """
    if not shutil.which("ogr2ogr"):
        raise FileNotFoundError(
            "ogr2ogr not found on PATH. Install GDAL with: "
            "apt install gdal-bin  /  brew install gdal"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Converting to GPKG: {output_path.name}")
    subprocess.run(
        [
            "ogr2ogr",
            "-f",
            "GPKG",
            str(output_path),
            str(filtered_pbf),
            "lines",
            "-overwrite",
        ],
        check=True,
    )
    print(f"  GPKG ready: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def acquire_region(region: str, data_dir: str | Path = "data/osm") -> Path:
    """Full pipeline for one region: download PBF, filter waterways, convert to GPKG.

    Returns the path to the final GeoPackage file.
    """
    region = region.upper()
    data_dir = Path(data_dir)

    print(f"=== Acquiring OSM waterways for {region} ===")

    pbf_path = download_pbf(region, data_dir)

    filtered_dir = data_dir / "filtered"
    filtered_path = filtered_dir / f"{region.lower()}-rivers.osm.pbf"
    filter_waterways(pbf_path, filtered_path)

    gpkg_dir = data_dir / "gpkg"
    gpkg_path = gpkg_dir / f"{region.lower()}-rivers.gpkg"
    pbf_to_gpkg(filtered_path, gpkg_path)

    print(f"=== {region} complete: {gpkg_path} ===\n")
    return gpkg_path


def acquire_all(
    data_dir: str | Path = "data/osm", regions: list[str] | None = None
) -> dict[str, Path]:
    """Run acquire_region for each region. Returns {region: gpkg_path}.

    If regions is None, processes all 6 SWORD regions.
    """
    if regions is None:
        regions = ALL_REGIONS
    else:
        regions = [r.upper() for r in regions]

    results: dict[str, Path] = {}
    for region in regions:
        results[region] = acquire_region(region, data_dir)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and filter OSM waterway data from Geofabrik."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--region",
        type=str,
        choices=ALL_REGIONS,
        help="SWORD region to acquire (NA, SA, EU, AF, AS, OC)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_regions",
        help="Acquire all 6 SWORD regions",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/osm",
        help="Base directory for OSM data (default: data/osm)",
    )

    args = parser.parse_args()

    if args.all_regions:
        acquire_all(data_dir=args.data_dir)
    else:
        acquire_region(args.region, data_dir=args.data_dir)
