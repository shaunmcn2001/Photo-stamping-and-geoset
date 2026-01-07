# Photo Stamp + GeoSet

This app stamps dates onto photos and builds a Google Earth KMZ ("geoset") using the photos' GPS EXIF data. It can also sort stamped photos into property folders based on polygons from a boundary KMZ.

## What it does

- Stamps the photo date (EXIF `DateTimeOriginal`, or file mtime fallback) bottom-right.
- Builds `geoset.kmz` with embedded photos + thumbnail icons.
- Optional: Sorts stamped photos into property folders using a boundary KMZ.

## Quick start (Windows)

Option A (recommended for download zip):
1. Download the repo as a ZIP from GitHub and extract it.
2. Double-click `SETUP.bat` (creates `~/Downloads/Photo Processing/Automation Scripts`).
3. The app launches. On first run, approve package install if prompted.

Option B (run from anywhere):
1. Double-click `RUN_ONE_CLICK.bat` (or run `python auto_stamp_and_kmz.py`).
2. On first run, approve package install if prompted.

Then:
1. Click **Create New Job...** and add JPGs, ZIPs, or folders (drag and drop is supported).
2. Select the job and click **Start**.
3. (Optional) Enable **Sort by Boundary KMZ**, pick the KMZ, set buffer distance, then **Start**.

You can also use **New Job from Folder...** or drop files/folders onto the main window to create a job quickly.

Workflow folder is created at `~/Downloads/Photo Processing` (with `Automation Scripts/` inside).
If Python is not installed, the run scripts will open the Microsoft Store page for Python.

Optional portable Python:
- If you place a portable Python at `Automation Scripts/python/python.exe`, the app will use it first.
- For ZIP installs, you can also include a `python/` folder in the same directory as `SETUP.bat`, and it will be copied into `Automation Scripts/python/`.
See `PORTABLE_PYTHON.md` for steps to build a portable Python with pip.

## Inputs

- Supported photo types: `.jpg` / `.jpeg` only.
- ZIPs: only JPGs inside ZIPs are imported; internal folder paths are ignored.
- Folders: JPGs (and ZIPs) inside the folder are imported.
- GPS required for KMZ placement. Photos without GPS go to `_NoGPS` (sorting mode).

## Boundary KMZ format (sorting mode)

- Each property should be a Folder containing one or more polygons.
- If the KMZ has a single top-level folder, its child folders are treated as properties.
- Polygons are unioned per property (all polygons inside a property folder).
- Points on the boundary are included ("covers" instead of "contains").

## Output structure

Each job folder looks like:

```
<Job Name>/
  Originals/           Original photos
  Stamped/             Stamped photos (or property folders when sorting)
  geoset.kmz           KMZ with placemarks + embedded photos
```

When sorting by boundary KMZ, `Stamped/` looks like:

```
Stamped/
  _cache/              Stamped originals used for KMZ build
  <Property Name>/     Sorted photos per property
  _Unassigned/         GPS but not inside any polygon
  _NoGPS/              No GPS data
```

## Requirements

- Python 3.9+ recommended
- Packages (auto-installed on first run):
  - Pillow
  - shapely (sorting mode)
  - pyproj (sorting mode)
  - tkinterdnd2 (drag and drop)

If needed, you can install manually:

```
python -m pip install --user -r requirements.txt
```

## Tips for speed

- Keep photos on a local SSD (avoid network drives).
- Only enable sorting when needed (it adds geometry work).
- Large jobs: split into smaller batches for faster feedback.

## Troubleshooting

- **No placemarks in KMZ**: Photos likely lack GPS EXIF.
- **Sorting disabled**: Install `shapely` + `pyproj` and restart.
- **Odd property names**: Folder names are sanitized for Windows compatibility.

## Notes

- Thumbnails are embedded inside `geoset.kmz` (no separate `Thumbs` folder on disk).
- Cancel stops after in-progress photos complete.
