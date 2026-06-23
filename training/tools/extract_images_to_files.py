#!/usr/bin/env python3
"""
Convert a MobileForge GRPO JSON with embedded base64 images into a JSON that stores
image file paths instead.

Why:
  A single JSON file with inline `data:image/...;base64,...` payloads can be
  hundreds of GB. `json.load()` expands that further in memory and can OOM
  before training starts. This script streams the input JSON item by item,
  writes each embedded image to disk, and replaces the image payload with a
  local file path.

Example:
  python tools/extract_images_to_files.py \
      --input data/mobileforge_grpo_20260426_151317.json \
      --output data/mobileforge_grpo_20260426_151317_image_paths.json \
      --image_dir data/mobileforge_grpo_20260426_151317_images

Requires:
  pip install ijson
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


DATA_URL_RE = re.compile(r"^data:image(?:/([a-zA-Z0-9.+-]+))?;base64,(.*)$", re.DOTALL)
EXT_MAP = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "webp": "webp",
    "gif": "gif",
}


def _normalize_decimal(value: Decimal) -> int | float:
    """Convert ijson Decimal values into JSON-serializable Python numbers."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _json_default(value: Any) -> Any:
    """Fallback serializer for values produced by streaming JSON parsers."""
    if isinstance(value, Decimal):
        return _normalize_decimal(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _parse_image_payload(value: str) -> tuple[bytes, str] | None:
    """Return (image_bytes, extension) for a base64 image string, else None."""
    if not isinstance(value, str):
        return None

    match = DATA_URL_RE.match(value)
    if match:
        fmt = (match.group(1) or "png").lower()
        b64 = match.group(2)
        ext = EXT_MAP.get(fmt, "png")
    else:
        # Plain file paths / URLs are left untouched.
        if os.path.exists(value) or value.startswith(("http://", "https://")):
            return None
        # Best-effort support for raw base64 strings.
        b64 = value
        ext = "png"

    try:
        return base64.b64decode(b64, validate=False), ext
    except Exception:
        return None


def _write_image(
    image_value: str,
    image_dir: Path,
    path_mode: str,
    output_dir: Path,
    dedup: bool,
    cache: dict[str, str],
) -> tuple[str, bool]:
    """Write an embedded image and return (path_to_store_in_json, converted)."""
    parsed = _parse_image_payload(image_value)
    if parsed is None:
        return image_value, False

    image_bytes, ext = parsed
    digest = hashlib.sha1(image_bytes).hexdigest()
    if dedup and digest in cache:
        return cache[digest], True

    shard = image_dir / digest[:2] / digest[2:4]
    shard.mkdir(parents=True, exist_ok=True)
    img_path = shard / f"{digest}.{ext}"
    if not img_path.exists():
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    if path_mode == "relative":
        stored_path = os.path.relpath(img_path, output_dir)
    else:
        stored_path = str(img_path.resolve())

    if dedup:
        cache[digest] = stored_path
    return stored_path, True


def _convert_obj(obj: Any, ctx: dict[str, Any]) -> Any:
    """Recursively replace image payloads in one sample."""
    if isinstance(obj, Decimal):
        return _normalize_decimal(obj)

    if isinstance(obj, list):
        return [_convert_obj(x, ctx) for x in obj]

    if not isinstance(obj, dict):
        return obj

    item_type = obj.get("type")

    if item_type == "image_url":
        new_obj = dict(obj)
        image_url = new_obj.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url", "")
            new_url, converted = _write_image(
                url,
                ctx["image_dir"],
                ctx["path_mode"],
                ctx["output_dir"],
                ctx["dedup"],
                ctx["cache"],
            )
            if converted:
                ctx["converted_images"] += 1
            new_image_url = dict(image_url)
            new_image_url["url"] = new_url
            new_obj["image_url"] = new_image_url
            return new_obj
        if isinstance(image_url, str):
            new_url, converted = _write_image(
                image_url,
                ctx["image_dir"],
                ctx["path_mode"],
                ctx["output_dir"],
                ctx["dedup"],
                ctx["cache"],
            )
            if converted:
                ctx["converted_images"] += 1
            new_obj["image_url"] = new_url
            return new_obj

    if item_type == "image":
        new_obj = dict(obj)
        image_value = new_obj.get("image", "")
        if isinstance(image_value, str):
            new_path, converted = _write_image(
                image_value,
                ctx["image_dir"],
                ctx["path_mode"],
                ctx["output_dir"],
                ctx["dedup"],
                ctx["cache"],
            )
            if converted:
                ctx["converted_images"] += 1
            new_obj["image"] = new_path
            return new_obj

    return {k: _convert_obj(v, ctx) for k, v in obj.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GRPO JSON file with embedded images")
    parser.add_argument("--output", required=True, help="Output GRPO JSON file with image paths")
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Directory to write images. Default: <output_stem>_images next to output JSON.",
    )
    parser.add_argument(
        "--path_mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="Store absolute image paths (default) or paths relative to the output JSON directory.",
    )
    parser.add_argument("--no_dedup", action="store_true", help="Disable SHA1 image deduplication")
    parser.add_argument("--progress_every", type=int, default=1000)
    args = parser.parse_args()

    try:
        import ijson  # type: ignore
    except ImportError:
        print("[ERROR] ijson is required. Install with: pip install ijson", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir) if args.image_dir else output_path.with_suffix("").parent / (
        output_path.with_suffix("").name + "_images"
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    ctx: dict[str, Any] = {
        "image_dir": image_dir,
        "output_dir": output_path.parent.resolve(),
        "path_mode": args.path_mode,
        "dedup": not args.no_dedup,
        "cache": {},
        "converted_images": 0,
    }

    print(f"[Convert] input     : {input_path}")
    print(f"[Convert] output    : {output_path}")
    print(f"[Convert] image_dir : {image_dir}")
    print(f"[Convert] path_mode : {args.path_mode}")
    print(f"[Convert] dedup     : {ctx['dedup']}")

    sample_count = 0
    with open(input_path, "rb") as fin, open(output_path, "w", encoding="utf-8") as fout:
        fout.write("[")
        first = True
        for sample in ijson.items(fin, "item"):
            sample_count += 1
            converted = _convert_obj(sample, ctx)
            if not first:
                fout.write(",")
            json.dump(converted, fout, ensure_ascii=False, separators=(",", ":"), default=_json_default)
            first = False

            if args.progress_every > 0 and sample_count % args.progress_every == 0:
                print(
                    f"[Convert] samples={sample_count:,} converted_images={ctx['converted_images']:,} "
                    f"unique_images={len(ctx['cache']):,}",
                    flush=True,
                )
        fout.write("]")

    print("[Convert] done")
    print(f"  samples          : {sample_count:,}")
    print(f"  converted images : {ctx['converted_images']:,}")
    print(f"  unique images    : {len(ctx['cache']):,}")
    print(f"  output json      : {output_path}")
    print(f"  image dir        : {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

