"""Download the project's soccer dataset from Roboflow Universe.

Default project: new-workspace-4dtyu/football-players-ball-ojdzs-s4i2s
(classes: ball, player, referee).

Usage:
    ROBOFLOW_API_KEY=xxxx python3 scripts/download_data.py

The dataset is public, but Roboflow still requires a (free) API key to download.
Get one at: https://app.roboflow.com  ->  Settings -> API Key (Private API Key).

Downloads into ./datasets/football-players-ball-ojdzs/ in YOLOv8 format:
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
"""
import os
import sys
from pathlib import Path

from roboflow import Roboflow

# Defaults target the project dataset (football-players-ball-ojdzs). Override via
# env vars to pull a different Roboflow project into its own folder:
#   RF_WORKSPACE, RF_PROJECT, RF_OUTDIR
WORKSPACE = os.environ.get("RF_WORKSPACE", "new-workspace-4dtyu")
PROJECT = os.environ.get("RF_PROJECT", "football-players-ball-ojdzs-s4i2s")
DOWNLOAD_FORMAT = "yolov8"
_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets" / "football-players-ball-ojdzs"
OUT_DIR = Path(os.environ.get("RF_OUTDIR", str(_DEFAULT_OUT)))


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: set ROBOFLOW_API_KEY first, e.g.\n"
            "    ROBOFLOW_API_KEY=your_key python scripts/download_data.py"
        )

    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    # Use the latest version available for the project.
    version_number = project.versions()[0].version.split("/")[-1]
    print(f"Downloading {PROJECT} version {version_number} ...")
    version = project.version(int(version_number))
    dataset = version.download(DOWNLOAD_FORMAT, location=str(OUT_DIR))
    print(f"Downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
