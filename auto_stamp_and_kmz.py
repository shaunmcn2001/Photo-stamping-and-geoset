import os
import re
import io
import sys
import shutil
import zipfile
import threading
import queue
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ==========================================================
# WORKFLOW (shareable) — lives in Downloads
# ==========================================================
WORKFLOW_ROOT_NAME = "Photo Processing"
SCRIPTS_DIR_NAME = "Automation Scripts"

IMAGE_EXTS = {".jpg", ".jpeg"}  # KMZ/Google Earth friendly

# Stamp style: bottom-right, yellow, NO outline, subtle shadow
FONT_SIZE_RATIO = 0.058
PAD_RATIO = 0.020
SHADOW_OFFSET_RATIO = 0.004
SHADOW_BLUR_RATIO = 0.003
SHADOW_ALPHA = 140

# KMZ thumbnail icons embedded in KMZ (NOT saved to disk)
THUMB_SIZE = 128  # 128x128 PNG

# Performance: parallel stamping
MAX_WORKERS = max(2, min(8, (os.cpu_count() or 4)))

# EXIF tag numbers
TAG_ORIENTATION = 274
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME = 306
TAG_GPSINFO = 34853

# Hide console window for subprocess on Windows
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Pillow globals (set after dependency check)
Image = ImageDraw = ImageFont = ImageOps = ImageFilter = ExifTags = None

# Shapely/pyproj globals (set after dependency check)
Polygon = Point = unary_union = shp_transform = STRtree = Transformer = None


# ---------------------------
# Paths
# ---------------------------
def get_downloads_path() -> Path:
    home = Path.home()
    p = home / "Downloads"
    return p if p.exists() else home


def get_workflow_root() -> Path:
    return get_downloads_path() / WORKFLOW_ROOT_NAME


def get_scripts_dir() -> Path:
    return get_workflow_root() / SCRIPTS_DIR_NAME


def write_text_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_run_bat(scripts_dir: Path) -> None:
    """
    Launch GUI without terminal (pythonw if available).
    """
    bat = scripts_dir / "RUN_ONE_CLICK.bat"
    bat_content = r"""@echo off
setlocal
cd /d "%~dp0"

where pythonw >nul 2>&1
if %errorlevel%==0 (
  pythonw auto_stamp_and_kmz.py
  exit /b
)

python auto_stamp_and_kmz.py
endlocal
"""
    bat.write_text(bat_content, encoding="utf-8")


def ensure_workflow_layout_and_drop_scripts() -> None:
    root = get_workflow_root()
    scripts = get_scripts_dir()
    root.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)

    readme_root = root / "README.txt"
    write_text_if_missing(
        readme_root,
        f"""Photo Processing Workflow (Downloads)
=================================

Workflow folder:
  {root}

How to use:
  1) Run: {scripts}\\RUN_ONE_CLICK.bat
  2) Click "Create New Job..." and import photos or ZIPs
  3) Select the job in the dropdown
  4) (Optional) tick "Sort by boundary KMZ" and choose boundary KMZ
  5) Click Start (only the selected job is processed)

Outputs inside each job folder:
  Originals\\                 (original photos)
  Stamped\\                   (stamped photos OR property subfolders)
  geoset.kmz                  (Google Earth KMZ)

NOTE:
  Thumbnail icons are embedded INSIDE geoset.kmz (no Thumbs folder on disk).
""",
    )

    req = scripts / "requirements.txt"
    write_text_if_missing(req, "Pillow>=9.0.0\nshapely>=2.0.0\npyproj>=3.6.0\n")

    write_run_bat(scripts)

    # Copy this script into Automation Scripts for sharing
    try:
        src = Path(__file__).resolve()
        dst = scripts / "auto_stamp_and_kmz.py"
        if src != dst:
            shutil.copy2(str(src), str(dst))
    except Exception:
        pass


# ---------------------------
# Dependency install/check (Store Python compatible)
# ---------------------------
def module_exists(modname: str) -> bool:
    return importlib.util.find_spec(modname) is not None


def run_pip_install(args: List[str]) -> Tuple[bool, str]:
    """
    Runs pip using *this* python.
    Uses --user for Store Python compatibility.
    """
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=CREATE_NO_WINDOW,
        )
        out = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
        return (p.returncode == 0), out.strip()
    except Exception as e:
        return False, str(e)


def restart_self():
    """
    Restart the script to make newly installed packages importable.
    """
    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            creationflags=CREATE_NO_WINDOW,
            cwd=str(Path(__file__).resolve().parent),
        )
    except Exception:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve())])
    raise SystemExit(0)


def ensure_requirements_on_first_launch(parent: tk.Tk) -> None:
    """
    On first open: if any required module missing, prompt and install.
    """
    ensure_workflow_layout_and_drop_scripts()
    req_path = get_scripts_dir() / "requirements.txt"

    missing = []
    if not module_exists("PIL"):
        missing.append("Pillow (PIL)")
    if not module_exists("shapely"):
        missing.append("shapely")
    if not module_exists("pyproj"):
        missing.append("pyproj")

    if not missing:
        return

    msg = (
        "First run setup:\n\n"
        "This tool needs a few Python packages to run:\n"
        f"  - {', '.join(missing)}\n\n"
        "Install them now? (Recommended)\n\n"
        "No terminal window will open."
    )
    if not messagebox.askyesno("Install requirements?", msg, parent=parent):
        if "Pillow (PIL)" in missing:
            messagebox.showerror(
                "Cannot continue",
                "Pillow is required to run.\n\nPlease install packages, then re-open the app.",
                parent=parent,
            )
            raise SystemExit(1)
        return

    run_pip_install(["install", "--user", "--upgrade", "pip"])
    ok, out = run_pip_install(["install", "--user", "-r", str(req_path)])

    if not ok:
        messagebox.showerror(
            "Install failed",
            "Could not install required packages.\n\nDetails:\n" + (out or "(no output)"),
            parent=parent,
        )
        raise SystemExit(1)

    messagebox.showinfo(
        "Installed",
        "Requirements installed successfully.\n\nRestarting the app now...",
        parent=parent,
    )
    restart_self()


def ensure_sort_deps_or_prompt(parent: tk.Tk) -> bool:
    if module_exists("shapely") and module_exists("pyproj"):
        return True

    if not messagebox.askyesno(
        "Sorting needs extra packages",
        "Sorting by Boundary KMZ needs shapely + pyproj.\n\nInstall them now?",
        parent=parent,
    ):
        return False

    req_path = get_scripts_dir() / "requirements.txt"
    run_pip_install(["install", "--user", "--upgrade", "pip"])
    ok, out = run_pip_install(["install", "--user", "-r", str(req_path)])
    if not ok:
        messagebox.showerror("Install failed", out or "(no output)", parent=parent)
        return False

    messagebox.showinfo("Installed", "Installed successfully.\n\nRestarting now...", parent=parent)
    restart_self()
    return True


def load_runtime_modules():
    global Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ExifTags
    global Polygon, Point, unary_union, shp_transform, STRtree, Transformer

    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw
    from PIL import ImageFont as _ImageFont
    from PIL import ImageOps as _ImageOps
    from PIL import ImageFilter as _ImageFilter
    from PIL import ExifTags as _ExifTags

    Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ExifTags = (
        _Image, _ImageDraw, _ImageFont, _ImageOps, _ImageFilter, _ExifTags
    )

    if module_exists("shapely") and module_exists("pyproj"):
        from shapely.geometry import Polygon as _Polygon, Point as _Point
        from shapely.ops import unary_union as _unary_union, transform as _shp_transform
        from shapely.strtree import STRtree as _STRtree
        from pyproj import Transformer as _Transformer

        Polygon, Point, unary_union, shp_transform, STRtree, Transformer = (
            _Polygon, _Point, _unary_union, _shp_transform, _STRtree, _Transformer
        )


# ---------------------------
# Helpers
# ---------------------------
_INVALID_WIN_CHARS = r'<>:"/\\|?*'
_invalid_re = re.compile(rf"[{re.escape(_INVALID_WIN_CHARS)}]")


def sanitize_folder_name(name: str, fallback: str = "Job") -> str:
    name = (name or "").strip()
    if not name:
        return fallback
    name = _invalid_re.sub("_", name)
    name = name.strip(" .")
    if not name:
        return fallback
    return name[:120]


def clean_display_name(name: str) -> str:
    if not name:
        return "Property"
    s = str(name).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dest_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = dest_dir / f"{base}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def is_image_filename(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def import_zip_to_originals(zip_path: Path, originals_dir: Path) -> Tuple[int, int]:
    copied = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if not is_image_filename(info.filename):
                skipped += 1
                continue
            base_name = Path(info.filename).name
            if not base_name:
                skipped += 1
                continue
            dst = unique_dest_path(originals_dir, base_name)
            try:
                with z.open(info) as fsrc:
                    with open(dst, "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
                copied += 1
            except Exception:
                skipped += 1
    return copied, skipped


# ✅ COPY-ONLY (no hardlinks)
def copy_only(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(str(src), str(dst))


# ---------------------------
# EXIF + GPS helpers
# ---------------------------
def get_exif_datetime(img: "Image.Image") -> Optional[datetime]:
    exif = img.getexif()
    for tag in (TAG_DATETIME_ORIGINAL, TAG_DATETIME):
        v = exif.get(tag)
        if v:
            try:
                return datetime.strptime(v, "%Y:%m:%d %H:%M:%S")
            except Exception:
                pass
    return None


def format_date(dt: datetime) -> str:
    return dt.strftime("%d %b %Y")  # 04 Jan 2026


def _rational_to_float(x) -> float:
    try:
        if isinstance(x, tuple) and len(x) == 2:
            return x[0] / x[1]
        return float(x)
    except Exception:
        return 0.0


def _dms_to_deg(dms) -> float:
    d = _rational_to_float(dms[0])
    m = _rational_to_float(dms[1])
    s = _rational_to_float(dms[2])
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_from_exif(img: "Image.Image") -> Optional[Tuple[float, float, float]]:
    exif = img.getexif()
    if not exif:
        return None

    gps_ifd = None
    try:
        gps_ifd = exif.get_ifd(TAG_GPSINFO)
    except Exception:
        gps_ifd = exif.get(TAG_GPSINFO)

    if not gps_ifd or not hasattr(gps_ifd, "items"):
        return None

    gps_decoded = {}
    for k, v in gps_ifd.items():
        name = ExifTags.GPSTAGS.get(k, k)
        gps_decoded[name] = v

    lat = gps_decoded.get("GPSLatitude")
    lat_ref = gps_decoded.get("GPSLatitudeRef")
    lon = gps_decoded.get("GPSLongitude")
    lon_ref = gps_decoded.get("GPSLongitudeRef")

    if not (lat and lat_ref and lon and lon_ref):
        return None

    lat_deg = _dms_to_deg(lat)
    lon_deg = _dms_to_deg(lon)

    if str(lat_ref).upper() == "S":
        lat_deg = -lat_deg
    if str(lon_ref).upper() == "W":
        lon_deg = -lon_deg

    alt = gps_decoded.get("GPSAltitude")
    alt_ref = gps_decoded.get("GPSAltitudeRef")
    alt_m = _rational_to_float(alt) if alt is not None else 0.0
    if alt_ref in (1, b"\x01"):
        alt_m = -alt_m

    return (lat_deg, lon_deg, alt_m)


# ---------------------------
# Stamp + thumbnail
# ---------------------------
def load_font(font_size: int):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def stamp_from_open_image(img0: "Image.Image", in_path: Path, out_path: Path) -> Optional["Image.Image"]:
    try:
        exif = img0.getexif()
        img = ImageOps.exif_transpose(img0)
        if exif is not None:
            exif[TAG_ORIENTATION] = 1
        exif_bytes = exif.tobytes() if exif is not None else b""

        dt = get_exif_datetime(img0) or datetime.fromtimestamp(in_path.stat().st_mtime)
        text = format_date(dt)

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        base = min(w, h)

        font_size = max(22, int(base * FONT_SIZE_RATIO))
        pad = max(8, int(base * PAD_RATIO))
        shadow_offset = max(1, int(base * SHADOW_OFFSET_RATIO))
        shadow_blur = max(1, int(base * SHADOW_BLUR_RATIO))

        font = load_font(font_size)

        meas = ImageDraw.Draw(img)
        bbox = meas.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = w - tw - pad
        y = h - th - pad

        shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_layer)
        sd.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, SHADOW_ALPHA))
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, shadow_layer)

        draw = ImageDraw.Draw(img_rgba)
        draw.text((x, y), text, font=font, fill=(255, 255, 0, 255))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_rgb = img_rgba.convert("RGB")
        out_rgb.save(str(out_path), quality=90, subsampling=2, exif=exif_bytes)
        return out_rgb
    except Exception:
        return None


def make_thumb_png_bytes_from_stamped(stamped_img_path: Path, size: int = THUMB_SIZE) -> Optional[bytes]:
    try:
        with Image.open(stamped_img_path) as im0:
            try:
                im0.draft("RGB", (size * 2, size * 2))
            except Exception:
                pass
            im = ImageOps.exif_transpose(im0)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            thumb = ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            buf = io.BytesIO()
            thumb.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return None


# ---------------------------
# Job folder handling
# ---------------------------
def ensure_originals_folder(job_dir: Path) -> Path:
    originals = job_dir / "Originals"
    originals.mkdir(exist_ok=True)

    for p in job_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            dest = originals / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))
    return originals


# ---------------------------
# Boundary KMZ parsing + matching (sorting mode)
# ---------------------------
def _parse_kml_coords(coord_text: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for token in coord_text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except Exception:
                pass
    return coords


def _kml_ns() -> Dict[str, str]:
    return {"k": "http://www.opengis.net/kml/2.2"}


def parse_boundary_kmz_properties(boundary_kmz: Path) -> Dict[str, Any]:
    """
    - If one top folder exists and it contains folders: treat it as "root"
    - Properties = root's direct child folders
    - Union ALL polygons under each property folder (any depth)
    """
    if Polygon is None:
        raise RuntimeError("Sorting mode needs shapely + pyproj.")

    ns = _kml_ns()
    with zipfile.ZipFile(boundary_kmz, "r") as z:
        kml_name = "doc.kml" if "doc.kml" in z.namelist() else z.namelist()[0]
        kml_text = z.read(kml_name).decode("utf-8", errors="replace")

    import xml.etree.ElementTree as ET
    root = ET.fromstring(kml_text)
    doc = root.find("k:Document", ns)
    if doc is None:
        raise RuntimeError("Boundary KMZ: Document node not found.")

    top_folders = doc.findall("k:Folder", ns)

    boundary_root = None
    if len(top_folders) == 1:
        children = top_folders[0].findall("k:Folder", ns)
        if children:
            boundary_root = top_folders[0]

    property_folders = boundary_root.findall("k:Folder", ns) if boundary_root is not None else top_folders
    if not property_folders:
        raise RuntimeError("Boundary KMZ: No property folders found.")

    def collect_polygons(folder_el) -> List[Any]:
        polys = []
        for pm in folder_el.findall(".//k:Placemark", ns):
            for poly in pm.findall(".//k:Polygon", ns):
                outer = poly.find(".//k:outerBoundaryIs/k:LinearRing/k:coordinates", ns)
                if outer is None or not outer.text:
                    continue
                outer_coords = _parse_kml_coords(outer.text)
                if len(outer_coords) < 4:
                    continue

                holes = []
                for inner in poly.findall(".//k:innerBoundaryIs/k:LinearRing/k:coordinates", ns):
                    if inner.text:
                        hole_coords = _parse_kml_coords(inner.text)
                        if len(hole_coords) >= 4:
                            holes.append(hole_coords)
                try:
                    g = Polygon(outer_coords, holes)
                    # ✅ Clean invalid/self-intersecting polygons
                    if not g.is_valid:
                        g = g.buffer(0)
                    if g.is_valid and not g.is_empty:
                        polys.append(g)
                except Exception:
                    pass
        return polys

    props: Dict[str, Any] = {}
    for f in property_folders:
        name_el = f.find("k:name", ns)
        raw_name = name_el.text if name_el is not None else "Property"
        prop_name = clean_display_name(raw_name)

        polys = collect_polygons(f)
        if polys:
            u = unary_union(polys)
            if not u.is_valid:
                u = u.buffer(0)
            props[prop_name] = u if (u is not None and not u.is_empty) else None
        else:
            props[prop_name] = None

    if not props:
        raise RuntimeError("Boundary KMZ: No properties parsed.")
    return props


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


class PropertyMatcher:
    """
    Buffers polygons in meters in a projected CRS and uses STRtree for fast matching.
    Uses .covers() (includes boundary) instead of .contains().
    """
    def __init__(self, props_wgs84: Dict[str, Any], buffer_m: float):
        if Transformer is None:
            raise RuntimeError("Sorting mode needs shapely + pyproj.")

        valid = [g for g in props_wgs84.values() if g is not None and not g.is_empty]
        if not valid:
            raise RuntimeError("Boundary KMZ: no usable polygon geometries found.")

        all_union = unary_union(valid)
        c = all_union.centroid
        epsg = utm_epsg_for(c.x, c.y)

        self.transformer = Transformer.from_crs(4326, epsg, always_xy=True)
        self.buffer_m = float(buffer_m)

        def proj(x, y, z=None):
            return self.transformer.transform(x, y)

        self.props_names: List[str] = []
        self.props_geom_buf: List[Any] = []

        for name, g in props_wgs84.items():
            if g is None or g.is_empty:
                continue
            gp = shp_transform(proj, g)
            if not gp.is_valid:
                gp = gp.buffer(0)
            gb = gp.buffer(self.buffer_m) if self.buffer_m > 0 else gp
            if not gb.is_valid:
                gb = gb.buffer(0)
            self.props_names.append(name)
            self.props_geom_buf.append(gb)

        self.tree = STRtree(self.props_geom_buf)

    def match(self, lat: float, lon: float) -> List[str]:
        x, y = self.transformer.transform(lon, lat)
        p = Point(x, y)

        hits = []
        try:
            res = self.tree.query(p)
        except Exception:
            res = []

        if not res:
            return hits

        first = res[0]
        # shapely2 may return indices (ints) or geometries
        if hasattr(first, "geom_type"):
            # geometry objects
            for g in res:
                try:
                    if g.covers(p):  # ✅ includes boundary points
                        # find index by identity fallback (small list so ok)
                        idx = self.props_geom_buf.index(g)
                        hits.append(self.props_names[idx])
                except Exception:
                    pass
        else:
            # indices
            for idx in res:
                try:
                    i = int(idx)
                    g = self.props_geom_buf[i]
                    if g.covers(p):  # ✅ includes boundary points
                        hits.append(self.props_names[i])
                except Exception:
                    pass

        return sorted(set(hits))


# ---------------------------
# Per-photo processing (parallel)
# ---------------------------
def process_one_photo(
    orig_path: Path,
    stamped_path: Path,
    cancel_event: threading.Event,
) -> Tuple[str, Optional[Tuple[float, float, float]], bool, Optional[str]]:
    try:
        if cancel_event.is_set():
            return orig_path.name, None, False, "Cancelled"

        orig_mtime = orig_path.stat().st_mtime
        stamped_exists = stamped_path.exists()
        stamped_changed = False

        with Image.open(orig_path) as img0:
            gps = get_gps_from_exif(img0)
            do_stamp = (not stamped_exists) or (stamped_path.stat().st_mtime < orig_mtime)

            if do_stamp:
                stamped_img_rgb = stamp_from_open_image(img0, orig_path, stamped_path)
                if stamped_img_rgb is None:
                    return orig_path.name, gps, False, "Stamp failed"
                stamped_changed = True

            return orig_path.name, gps, stamped_changed, None

    except Exception as e:
        return orig_path.name, None, False, str(e)


# ---------------------------
# KMZ creation (thumb icons embedded, big balloon)
# ---------------------------
def build_geoset_kmz_grouped(
    kmz_path: Path,
    doc_name: str,
    groups: Dict[str, List[str]],  # folder -> filenames
    photo_gps: Dict[str, Tuple[float, float, float]],
    stamped_files_resolver,        # callable(fname)->Path
    show_labels: bool = True,
) -> Tuple[int, int]:
    label_scale = "1.0" if show_labels else "0.0"

    style_block = """
    <Style id="balloonStyle">
      <BalloonStyle>
        <text><![CDATA[
          <div style="font-family:Segoe UI, Arial, sans-serif; width:760px;">
            <div style="font-weight:700; font-size:16px; margin:0 0 8px 0;">$[name]</div>
            <div style="border:1px solid #ddd; background:#000; padding:6px;">
              <img src="$[description]" style="width:100%; height:auto; display:block;"/>
            </div>
          </div>
        ]]></text>
      </BalloonStyle>
    </Style>
""".strip()

    def placemark_xml(stem: str, lon: float, lat: float, alt: float, photo_href: str, icon_href: str) -> str:
        return f"""
    <Placemark>
      <name>{stem}</name>
      <styleUrl>#balloonStyle</styleUrl>
      <Style>
        <IconStyle>
          <scale>1.0</scale>
          <Icon><href>{icon_href}</href></Icon>
        </IconStyle>
        <LabelStyle><scale>{label_scale}</scale></LabelStyle>
      </Style>
      <description>{photo_href}</description>
      <Point><coordinates>{lon},{lat},{alt}</coordinates></Point>
    </Placemark>
""".strip()

    folder_kml_parts: List[str] = []
    placemark_count = 0

    all_filenames: List[str] = []
    for _, files in groups.items():
        for f in files:
            if f not in all_filenames:
                all_filenames.append(f)

    for folder_name, files in groups.items():
        pm_parts: List[str] = []
        for fname in files:
            gps = photo_gps.get(fname)
            if not gps:
                continue
            lat, lon, alt = gps
            stem = Path(fname).stem

            photo_href = f"files/{fname}"
            icon_href = f"thumbs/{stem}.png"

            pm_parts.append(placemark_xml(stem, lon, lat, alt, photo_href, icon_href))
            placemark_count += 1

        folder_kml_parts.append(
            f"""
    <Folder>
      <name>{folder_name}</name>
      {''.join(pm_parts)}
    </Folder>
""".strip()
        )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{doc_name}</name>
    {style_block}
    {''.join(folder_kml_parts)}
  </Document>
</kml>
"""

    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)

        for fname in all_filenames:
            img_path = stamped_files_resolver(fname)
            if img_path and Path(img_path).exists():
                z.write(str(img_path), arcname=f"files/{fname}")

                thumb_bytes = make_thumb_png_bytes_from_stamped(Path(img_path), THUMB_SIZE)
                if thumb_bytes:
                    z.writestr(f"thumbs/{Path(fname).stem}.png", thumb_bytes)

    return placemark_count, len(all_filenames)


# ---------------------------
# GUI: Logs window
# ---------------------------
class LogWindow:
    def __init__(self, parent: tk.Tk, title: str = "Terminal / Logs"):
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("920x500")
        self.win.minsize(650, 320)

        frm = ttk.Frame(self.win, padding=10)
        frm.pack(fill="both", expand=True)

        self.text = tk.Text(frm, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(frm, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)
        self.text.configure(state="disabled")

    def append(self, line: str):
        self.text.configure(state="normal")
        self.text.insert("end", line.rstrip() + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")


# ---------------------------
# GUI: New Job Dialog
# ---------------------------
class NewJobDialog:
    def __init__(self, parent: tk.Tk, workflow_root: Path):
        self.workflow_root = workflow_root
        self.result_job_dir: Optional[Path] = None

        self.win = tk.Toplevel(parent)
        self.win.title("Create New Job")
        self.win.geometry("720x360")
        self.win.resizable(False, False)
        self.win.transient(parent)
        self.win.grab_set()

        self.job_name_var = tk.StringVar(value="")
        self.files: List[str] = []
        self.zips: List[str] = []

        frm = ttk.Frame(self.win, padding=14)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Create a new job folder and import photos", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        row1 = ttk.Frame(frm)
        row1.pack(fill="x", pady=(12, 8))
        ttk.Label(row1, text="Job name:").pack(side="left")
        ttk.Entry(row1, textvariable=self.job_name_var).pack(side="left", padx=(8, 8), fill="x", expand=True)

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=(4, 6))
        ttk.Label(row2, text="Add JPGs:").pack(side="left")
        ttk.Button(row2, text="Choose Photos…", command=self.choose_files).pack(side="left", padx=(8, 8))
        ttk.Button(row2, text="Clear", command=self.clear_files).pack(side="left")

        self.files_label = ttk.Label(frm, text="No photos selected.", wraplength=690)
        self.files_label.pack(anchor="w", pady=(0, 10))

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=(4, 6))
        ttk.Label(row3, text="Add ZIPs:").pack(side="left")
        ttk.Button(row3, text="Choose ZIPs…", command=self.choose_zips).pack(side="left", padx=(18, 8))
        ttk.Button(row3, text="Clear", command=self.clear_zips).pack(side="left")

        self.zips_label = ttk.Label(frm, text="No ZIPs selected.", wraplength=690)
        self.zips_label.pack(anchor="w", pady=(0, 8))

        ttk.Label(
            frm,
            text="ZIPs: Only .jpg/.jpeg are imported. Folder paths inside ZIP are ignored.",
            wraplength=690,
        ).pack(anchor="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(18, 0))

        ttk.Button(btns, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(btns, text="Create Job", command=self.create).pack(side="right", padx=(0, 10))

    def choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Select original photos",
            filetypes=[("JPEG photos", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if paths:
            self.files = list(paths)
            self.files_label.configure(text=f"{len(self.files)} photo(s) selected.")

    def clear_files(self):
        self.files = []
        self.files_label.configure(text="No photos selected.")

    def choose_zips(self):
        paths = filedialog.askopenfilenames(
            title="Select ZIP files",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
        )
        if paths:
            self.zips = list(paths)
            self.zips_label.configure(text=f"{len(self.zips)} ZIP(s) selected.")

    def clear_zips(self):
        self.zips = []
        self.zips_label.configure(text="No ZIPs selected.")

    def cancel(self):
        self.result_job_dir = None
        self.win.destroy()

    def create(self):
        raw = self.job_name_var.get().strip()
        if not raw:
            messagebox.showerror("Job name required", "Please enter a job name.")
            return

        if not self.files and not self.zips:
            if not messagebox.askyesno("No imports selected", "No photos or ZIPs selected. Create empty job anyway?"):
                return

        job_name = sanitize_folder_name(raw, fallback="Job")
        job_dir = self.workflow_root / job_name

        if job_dir.exists():
            i = 1
            while True:
                candidate = self.workflow_root / f"{job_name}_{i}"
                if not candidate.exists():
                    job_dir = candidate
                    break
                i += 1

        originals_dir = job_dir / "Originals"
        originals_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "Stamped").mkdir(exist_ok=True)

        copied = 0
        skipped = 0

        for f in self.files:
            src = Path(f)
            if not src.exists() or src.suffix.lower() not in IMAGE_EXTS:
                skipped += 1
                continue
            dst = unique_dest_path(originals_dir, src.name)
            try:
                shutil.copy2(str(src), str(dst))
                copied += 1
            except Exception:
                skipped += 1

        for z in self.zips:
            zp = Path(z)
            if not zp.exists() or zp.suffix.lower() != ".zip":
                skipped += 1
                continue
            try:
                c, s = import_zip_to_originals(zp, originals_dir)
                copied += c
                skipped += s
            except Exception as e:
                skipped += 1
                messagebox.showwarning("ZIP import failed", f"Failed to import:\n{zp}\n\n{e}")

        self.result_job_dir = job_dir
        messagebox.showinfo(
            "Job created",
            f"Created:\n{job_dir}\n\nImported files: {copied}\nSkipped: {skipped}",
        )
        self.win.destroy()


# ---------------------------
# Main App
# ---------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Photo Stamp + KMZ Builder")
        self.root.geometry("980x520")
        self.root.resizable(False, False)

        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self.log_window: Optional[LogWindow] = None
        self.log_buffer: List[str] = []

        self.workflow_root = get_workflow_root()

        self.selected_job_var = tk.StringVar(value="")
        self.jobs_list: List[str] = []

        self.sort_mode_var = tk.BooleanVar(value=False)
        self.boundary_kmz_var = tk.StringVar(value="")
        self.buffer_m_var = tk.IntVar(value=50)
        self.clear_sorted_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.refresh_jobs(select_if_single=True)
        self.root.after(100, self._poll)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=14)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Photo Processing → Stamp + Geo KMZ", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        r0 = ttk.Frame(frm)
        r0.pack(fill="x", pady=(8, 8))
        ttk.Label(r0, text="Workflow folder:").pack(side="left")
        ttk.Label(r0, text=str(self.workflow_root), wraplength=650).pack(side="left", padx=(8, 8), fill="x", expand=True)
        ttk.Button(r0, text="Open", command=self._open_workflow_root).pack(side="left")
        ttk.Button(r0, text="Create New Job…", command=self.create_new_job).pack(side="left", padx=(8, 0))

        r1 = ttk.Frame(frm)
        r1.pack(fill="x", pady=(0, 10))
        ttk.Label(r1, text="Selected job:").pack(side="left")

        self.jobs_combo = ttk.Combobox(r1, textvariable=self.selected_job_var, state="readonly", values=[])
        self.jobs_combo.pack(side="left", padx=(8, 8), fill="x", expand=True)
        self.jobs_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_job_selected())

        ttk.Button(r1, text="Refresh", command=self.refresh_jobs).pack(side="left")
        ttk.Button(r1, text="Open Job", command=self.open_selected_job).pack(side="left", padx=(8, 0))

        sep = ttk.Separator(frm)
        sep.pack(fill="x", pady=(4, 10))

        r2 = ttk.Frame(frm)
        r2.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(
            r2,
            text="Sort stamped photos into property folders using Boundary KMZ",
            variable=self.sort_mode_var,
            command=self._on_sort_toggle,
        ).pack(side="left")

        r3 = ttk.Frame(frm)
        r3.pack(fill="x", pady=(0, 6))
        ttk.Label(r3, text="Boundary KMZ:").pack(side="left")
        self.boundary_entry = ttk.Entry(r3, textvariable=self.boundary_kmz_var)
        self.boundary_entry.pack(side="left", padx=(8, 8), fill="x", expand=True)
        ttk.Button(r3, text="Browse…", command=self.choose_boundary_kmz).pack(side="left")

        r4 = ttk.Frame(frm)
        r4.pack(fill="x", pady=(0, 10))
        ttk.Label(r4, text="Close-to-boundary distance (m):").pack(side="left")
        self.buffer_spin = ttk.Spinbox(r4, from_=0, to=5000, textvariable=self.buffer_m_var, width=8)
        self.buffer_spin.pack(side="left", padx=(8, 16))
        ttk.Checkbutton(r4, text="Clear existing property folders before sorting", variable=self.clear_sorted_var).pack(side="left")

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self.status_var).pack(anchor="w")

        self.progress = ttk.Progressbar(frm, length=940, mode="determinate")
        self.progress.pack(anchor="w", pady=(10, 6))

        self.count_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.count_var).pack(anchor="w")

        btns = ttk.Frame(frm)
        btns.pack(anchor="e", pady=(14, 0), fill="x")

        ttk.Button(btns, text="Show Terminal / Logs", command=self.show_logs).pack(side="left")

        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="right", padx=(8, 0))

        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="right")

        self._on_sort_toggle()

    def _on_sort_toggle(self):
        enabled = bool(self.sort_mode_var.get())
        state = "normal" if enabled else "disabled"
        try:
            self.boundary_entry.configure(state=state)
            self.buffer_spin.configure(state=state)
        except Exception:
            pass

        if enabled:
            if (not module_exists("shapely")) or (not module_exists("pyproj")):
                ok = ensure_sort_deps_or_prompt(self.root)
                if not ok:
                    self.sort_mode_var.set(False)
                    self._on_sort_toggle()

    def choose_boundary_kmz(self):
        p = filedialog.askopenfilename(
            title="Select boundary KMZ (folders = properties, polygons inside)",
            filetypes=[("KMZ files", "*.kmz"), ("All files", "*.*")],
        )
        if p:
            self.boundary_kmz_var.set(p)

    def _open_workflow_root(self):
        self.workflow_root.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.workflow_root))
        except Exception:
            pass

    def refresh_jobs(self, select_if_single: bool = False):
        self.workflow_root.mkdir(parents=True, exist_ok=True)
        jobs = []
        for p in sorted(self.workflow_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name.lower() == SCRIPTS_DIR_NAME.lower():
                continue
            jobs.append(p.name)

        self.jobs_list = jobs
        self.jobs_combo["values"] = self.jobs_list

        current = self.selected_job_var.get().strip()
        if current in self.jobs_list:
            self._on_job_selected()
            return

        if select_if_single and len(self.jobs_list) == 1:
            self.selected_job_var.set(self.jobs_list[0])
        elif self.jobs_list:
            self.selected_job_var.set(self.jobs_list[0])
        else:
            self.selected_job_var.set("")
        self._on_job_selected()

    def _on_job_selected(self):
        job = self.selected_job_var.get().strip()
        if not job:
            self.status_var.set("Ready. No job selected.")
            return
        self.status_var.set(f"Ready. Selected job: {job}")

    def open_selected_job(self):
        job = self.selected_job_var.get().strip()
        if not job:
            messagebox.showinfo("No job selected", "Select a job first.")
            return
        job_dir = self.workflow_root / job
        if not job_dir.exists():
            messagebox.showerror("Missing job folder", f"Job folder not found:\n{job_dir}")
            return
        try:
            os.startfile(str(job_dir))
        except Exception:
            pass

    def create_new_job(self):
        self.workflow_root.mkdir(parents=True, exist_ok=True)
        dlg = NewJobDialog(self.root, self.workflow_root)
        self.root.wait_window(dlg.win)

        if dlg.result_job_dir is not None:
            self.refresh_jobs()
            self.selected_job_var.set(dlg.result_job_dir.name)
            self._on_job_selected()
            try:
                os.startfile(str(dlg.result_job_dir))
            except Exception:
                pass

    # ---- logs ----
    def show_logs(self):
        if self.log_window is None or not self._window_exists(self.log_window.win):
            self.log_window = LogWindow(self.root)
            for line in self.log_buffer:
                self.log_window.append(line)
        else:
            self.log_window.win.lift()
            self.log_window.win.focus_force()

    @staticmethod
    def _window_exists(win: tk.Toplevel) -> bool:
        try:
            return bool(win.winfo_exists())
        except Exception:
            return False

    def _log(self, line: str):
        self.log_buffer.append(line)
        if len(self.log_buffer) > 2500:
            self.log_buffer = self.log_buffer[-2500:]
        if self.log_window is not None and self._window_exists(self.log_window.win):
            self.log_window.append(line)

    # ---- run ----
    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.refresh_jobs()

        job_name = self.selected_job_var.get().strip()
        if not job_name:
            messagebox.showerror("Select a job", "Please select a job to process.")
            return

        job_dir = self.workflow_root / job_name
        if not job_dir.exists():
            messagebox.showerror("Missing job folder", f"Job folder not found:\n{job_dir}")
            return

        sorting = bool(self.sort_mode_var.get())
        boundary_kmz = self.boundary_kmz_var.get().strip()

        if sorting:
            if Polygon is None or Transformer is None:
                ok = ensure_sort_deps_or_prompt(self.root)
                if not ok:
                    return
                return
            if not boundary_kmz:
                self.choose_boundary_kmz()
                boundary_kmz = self.boundary_kmz_var.get().strip()
            if not boundary_kmz:
                sorting = False

        originals = job_dir / "Originals"
        if not originals.exists():
            self.status_var.set("No Originals folder found in this job.")
            return

        originals_imgs = [p for p in originals.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        total_steps = len(originals_imgs) + 1
        if len(originals_imgs) == 0:
            self.status_var.set("No photos found in Originals.")
            return

        self.cancel_event.clear()
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")

        self.progress["value"] = 0
        self.progress["maximum"] = max(1, total_steps)
        self.count_var.set(f"0 / {total_steps}")

        self._log("=== START ===")
        self._log(f"Selected job  = {job_dir}")
        self._log(f"Threads       = {MAX_WORKERS}")
        self._log(f"Sorting mode  = {sorting}")
        if sorting:
            self._log(f"Boundary KMZ  = {boundary_kmz}")
            self._log(f"Buffer (m)    = {self.buffer_m_var.get()}")

        self.worker_thread = threading.Thread(
            target=self._run_worker_single_job,
            args=(job_dir, total_steps, sorting, boundary_kmz),
            daemon=True,
        )
        self.worker_thread.start()

    def cancel(self):
        self.cancel_event.set()
        self.status_var.set("Cancelling... (finishing in-progress images)")
        self._log("Cancel requested by user.")

    def _clear_property_folders(self, stamped_dir: Path):
        if not stamped_dir.exists():
            return
        for p in stamped_dir.iterdir():
            if p.is_dir() and p.name != "_cache":
                try:
                    shutil.rmtree(str(p))
                except Exception:
                    pass

    def _run_worker_single_job(self, job_dir: Path, total_steps: int, sorting: bool, boundary_kmz: str):
        done = 0
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            self.q.put(("status", f"Preparing job: {job_dir.name}"))

            originals = ensure_originals_folder(job_dir)
            stamped_dir = job_dir / "Stamped"
            stamped_dir.mkdir(exist_ok=True)

            originals_imgs = [p for p in sorted(originals.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if not originals_imgs:
                self.q.put(("done", "No photos found in Originals."))
                return

            matcher: Optional[PropertyMatcher] = None
            props_wgs84: Dict[str, Any] = {}
            prop_fs_map: Dict[str, str] = {}

            groups: Dict[str, List[str]] = {}

            if sorting and boundary_kmz:
                self.q.put(("status", "Reading boundary KMZ…"))
                props_wgs84 = parse_boundary_kmz_properties(Path(boundary_kmz))

                # LOG properties and polygon availability
                for k, g in props_wgs84.items():
                    self.q.put(("log", f"  [BOUNDARY] {k}: {'OK' if (g is not None and not g.is_empty) else 'NO POLYGONS FOUND'}"))

                # Build matcher from valid polygons
                buffer_m = float(self.buffer_m_var.get())
                matcher = PropertyMatcher(props_wgs84, buffer_m)

                if self.clear_sorted_var.get():
                    self.q.put(("status", "Clearing old sorted folders…"))
                    self._clear_property_folders(stamped_dir)

                # Create property folders
                for prop_name in props_wgs84.keys():
                    fs_name = sanitize_folder_name(prop_name, fallback="Property")
                    prop_fs_map[prop_name] = fs_name
                    (stamped_dir / fs_name).mkdir(parents=True, exist_ok=True)
                    groups[prop_name] = []

                groups["_Unassigned"] = []
                groups["_NoGPS"] = []
            else:
                groups["Photos"] = []

            cache_dir = (stamped_dir / "_cache") if sorting else stamped_dir
            cache_dir.mkdir(exist_ok=True)

            photo_gps: Dict[str, Tuple[float, float, float]] = {}

            def assign_groups(fname: str, gps: Optional[Tuple[float, float, float]]) -> List[str]:
                if not gps:
                    groups.setdefault("_NoGPS", []).append(fname)
                    return ["_NoGPS"]
                if matcher is None:
                    groups.setdefault("Photos", []).append(fname)
                    return ["Photos"]

                lat, lon, _ = gps
                hits = matcher.match(lat, lon)
                if not hits:
                    groups.setdefault("_Unassigned", []).append(fname)
                    return ["_Unassigned"]

                for prop in hits:
                    groups.setdefault(prop, []).append(fname)
                return hits

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = []
                for orig_path in originals_imgs:
                    out_path = cache_dir / orig_path.name
                    futures.append(ex.submit(process_one_photo, orig_path, out_path, self.cancel_event))

                for fut in as_completed(futures):
                    if self.cancel_event.is_set():
                        break

                    fname, gps, stamped_changed, err = fut.result()

                    if gps:
                        photo_gps[fname] = gps

                    if err:
                        self.q.put(("log", f"  [WARN] {fname}: {err}"))
                    else:
                        self.q.put(("log", f"  {'[STAMP]' if stamped_changed else '[SKIP ]'} {fname}"))

                    assigned = assign_groups(fname, gps)

                    # ✅ COPY into folder(s) (no hardlinks)
                    if sorting:
                        src = cache_dir / fname
                        if src.exists():
                            for grp in assigned:
                                if grp in ("_NoGPS", "_Unassigned"):
                                    fs_folder = sanitize_folder_name(grp, fallback=grp)
                                    (stamped_dir / fs_folder).mkdir(parents=True, exist_ok=True)
                                    copy_only(src, stamped_dir / fs_folder / fname)
                                else:
                                    fs_folder = prop_fs_map.get(grp) or sanitize_folder_name(grp, fallback="Property")
                                    (stamped_dir / fs_folder).mkdir(parents=True, exist_ok=True)
                                    copy_only(src, stamped_dir / fs_folder / fname)

                    done += 1
                    self.q.put(("progress", done, total_steps))
                    self.q.put(("status", f"Processed: {job_dir.name} → {fname}"))

            if self.cancel_event.is_set():
                self.q.put(("done", "Cancelled."))
                return

            self.q.put(("status", f"Building geoset.kmz: {job_dir.name}"))
            kmz_out = job_dir / "geoset.kmz"

            def stamped_resolver(fname: str) -> Path:
                return (stamped_dir / "_cache" / fname) if sorting else (stamped_dir / fname)

            if sorting:
                ordered: Dict[str, List[str]] = {}
                for k in sorted([k for k in groups.keys() if not k.startswith("_")]):
                    ordered[k] = groups[k]
                for k in ["_Unassigned", "_NoGPS"]:
                    if k in groups:
                        ordered[k] = groups[k]
                groups_for_kmz = ordered
            else:
                groups_for_kmz = {"Photos": list(photo_gps.keys())}

            placemarks, photos_used = build_geoset_kmz_grouped(
                kmz_path=kmz_out,
                doc_name=job_dir.name,
                groups=groups_for_kmz,
                photo_gps=photo_gps,
                stamped_files_resolver=stamped_resolver,
                show_labels=True,
            )

            if sorting:
                self.q.put(("log", "  [MATCH SUMMARY]"))
                for k in groups_for_kmz.keys():
                    self.q.put(("log", f"    {k}: {len(groups_for_kmz[k])} photo(s)"))

            self.q.put(("log", f"  [KMZ ] {kmz_out.name}: {placemarks} placemarks, {photos_used} photos embedded"))
            done += 1
            self.q.put(("progress", done, total_steps))
            self.q.put(("done", "Done."))

        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]

                if kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "progress":
                    d, t = msg[1], msg[2]
                    self.progress["value"] = d
                    self.count_var.set(f"{d} / {t}")
                elif kind == "log":
                    self._log(msg[1])
                elif kind == "error":
                    self._finish_buttons()
                    self._log(f"[ERROR] {msg[1]}")
                    messagebox.showerror("Error", msg[1])
                    self.status_var.set("Error.")
                    self.count_var.set("")
                    self.progress["value"] = 0
                elif kind == "done":
                    self._finish_buttons()
                    self.status_var.set(msg[1])
                    self._log(f"=== {msg[1].upper()} ===")

        except queue.Empty:
            pass

        self.root.after(100, self._poll)

    def _finish_buttons(self):
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")


def main():
    ensure_workflow_layout_and_drop_scripts()

    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    ensure_requirements_on_first_launch(root)
    load_runtime_modules()

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
