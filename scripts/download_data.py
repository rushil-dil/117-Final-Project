"""Download the football-players-detection dataset from Roboflow Universe.

Usage:
    ROBOFLOW_API_KEY=xxxx python scripts/download_data.py

The dataset is public, but Roboflow still requires a (free) API key to download.
Get one at: https://app.roboflow.com  ->  Settings -> API Key (Private API Key).

Downloads into ./datasets/football-players-detection/ in YOLOv8 format:
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
"""
import os
import sys
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "roboflow-jvuqo"
PROJECT = "football-players-detection-3zvbc"
DOWNLOAD_FORMAT = "yolov8"
OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "football-players-detection"


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
