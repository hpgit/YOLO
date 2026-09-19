"""Download full official COCO 2017 with resumable parallel ranges and CRC checks."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
import zipfile
from pathlib import Path


def download(name, root, workers):
    directory = "annotations" if name.startswith("annotations") else "zips"
    url = f"https://s3.amazonaws.com/images.cocodataset.org/{directory}/{name}.zip"
    target = root / f"{name}.zip"
    parts = root / f".{name}.parts"
    parts.mkdir(exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    chunk = 4 * 1024 * 1024
    total = (size + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        path = parts / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Server did not honor range request")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Unexpected range response")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("Incomplete range")
                path.with_suffix(".tmp").write_bytes(data)
                path.with_suffix(".tmp").replace(path)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)

    started = last = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(
            concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(total)]), 1
        ):
            result.result()
            if time.monotonic() - last > 15 or done == total:
                print(
                    f"{name}: {done}/{total} parts ({100 * done / total:.1f}%), elapsed {time.monotonic() - started:.0f}s",
                    flush=True,
                )
                last = time.monotonic()
    temporary = target.with_suffix(".assembling")
    with temporary.open("wb") as output:
        for i in range(total):
            output.write((parts / str(i)).read_bytes())
    temporary.replace(target)
    print(f"{name}: extracting and checking ZIP CRC", flush=True)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root if directory == "annotations" else root / "images")
    for path in parts.iterdir():
        path.unlink()
    parts.rmdir()
    target.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/coco"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in ("annotations_trainval2017", "val2017", "train2017"):
        expected = {"val2017": 5000, "train2017": 118287}.get(name)
        if expected and len(list((args.root / "images" / name).glob("*.jpg"))) == expected:
            continue
        if not expected and all(
            (args.root / "annotations" / f"instances_{split}2017.json").exists() for split in ("train", "val")
        ):
            continue
        download(name, args.root, args.workers)
    counts = {}
    for split, expected in (("train2017", 118287), ("val2017", 5000)):
        annotation = json.loads((args.root / "annotations" / f"instances_{split}.json").read_text())
        images = args.root / "images" / split
        assert len(list(images.glob("*.jpg"))) == expected
        assert len(annotation["images"]) == expected
        assert all((images / item["file_name"]).is_file() for item in annotation["images"])
        counts[split] = {
            "images": expected,
            "annotations": len(annotation["annotations"]),
            "categories": len(annotation["categories"]),
        }
    (args.root / "verification.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
