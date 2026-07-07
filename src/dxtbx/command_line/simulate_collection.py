# LIBTBX_SET_DISPATCHER_NAME dev.dxtbx.simulate_collection
"""
Simulate a live serial (SSX) data collection from an example NXmx master file.

Given a real DLS-style NXmx master (e.g. ``escrima_1589.nxs``) whose ``entry/data/data``
is a virtual dataset (VDS) mapping ~1000-frame blocks to external ``*_NNNNNN.h5`` files,
this tool:

  1. duplicates the master's structure into a fresh output directory -- copying all the
     metadata verbatim and recreating the VDS + external links -- but with the underlying
     block files created *empty* (0 frames, growable), exactly as they exist up front at
     the start of a real collection; then
  2. fills those block files incrementally via SWMR, one frame at a time, closing each
     block file when it is complete -- mimicking the detector writer.

The point is to have a realistic, *growing* dataset to test ``dev.dxtbx.image_availability``
(and the ``xia2.ssx wait_for_images`` live-processing mode) against, without needing a live
beamline. No real image data is written -- frames are zeros (compressed to almost nothing),
so only frame *counts* grow; that is exactly what the availability tools inspect.

The example master and its data files are only ever opened read-only and are never modified.

Example::

    dev.dxtbx.simulate_collection /dls/.../escrima_1589.nxs -o /scratch/sim --frames 3000 --rate 50
    # ...then, in another terminal:
    dev.dxtbx.image_availability /scratch/sim/escrima_1589.nxs
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

import h5py
import numpy as np

import dxtbx.util


def _find_signal_vds(handle):
    """Return the NXmx signal dataset (which must be a VDS) from an open master file."""
    import nxmx

    nxmx_obj = nxmx.NXmx(handle)
    nxdata = nxmx_obj.entries[0].data[0]
    data = nxdata[nxdata.signal] if nxdata.signal else list(nxdata.values())[0]
    if not data.is_virtual:
        sys.exit(
            f"{handle.filename}: signal dataset {data.name!r} is not a virtual dataset; "
            "this tool only simulates VDS-backed (block) collections."
        )
    return data


def _block_layout(master):
    """Describe the VDS blocks of an open master file.

    Returns (total_frames, frame_shape, dtype, blocks) where blocks is a frame-ordered
    list of dicts: {start, stop, link_name, source_dset, source_filename}.
    """
    data = _find_signal_vds(master)
    total = int(data.shape[0])
    frame_shape = tuple(int(x) for x in data.shape[1:])
    blocks = []
    for vmap in data.virtual_sources():
        start, end = vmap.vspace.get_select_bounds()
        gstart, gend = int(start[0]), int(end[0])
        if vmap.file_name != ".":
            # VDS names the data file directly.
            source_filename = vmap.file_name
            source_dset = vmap.dset_name
            link_name = pathlib.PurePosixPath(vmap.dset_name).name
        else:
            # DLS/GDA layout: source is an in-master external link to the data file.
            link = master.get(vmap.dset_name, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                sys.exit(
                    f"VDS source {vmap.dset_name!r} is not an external link; "
                    "unsupported master layout."
                )
            source_filename = link.filename
            source_dset = link.path
            link_name = pathlib.PurePosixPath(vmap.dset_name).name
        blocks.append(
            {
                "start": gstart,
                "stop": gend + 1,
                "link_name": link_name,
                "source_dset": source_dset.lstrip("/"),
                "source_filename": source_filename,
            }
        )
    blocks.sort(key=lambda b: b["start"])
    return total, frame_shape, data.dtype, blocks


def _new_source_filename(original: str, stem: str) -> str:
    """Map an original block filename onto the output stem, preserving its index/suffix.

    e.g. ("escrima_1589_000007.h5", "sim") -> "sim_000007.h5".
    """
    base = pathlib.PurePosixPath(original).name
    match = re.search(r"(_\d+)(\.[^.]+)$", base)
    if match:
        return f"{stem}{match.group(1)}{match.group(2)}"
    return f"{stem}_{base}"


def _copy_structure(src, dst, skip_paths):
    """Recursively copy groups/datasets/soft-links from src group into dst group,
    skipping any object whose full path is in skip_paths (the VDS + its sources).

    The skip check uses the master-side path (group + key), computed before the link is
    resolved -- important because resolving an external link yields the *target* file's
    internal name, not the path in the master.
    """
    dst.attrs.update(src.attrs)
    base = src.name.rstrip("/")
    for key in src:
        if f"{base}/{key}" in skip_paths:
            continue
        link = src.get(key, getlink=True)
        if isinstance(link, h5py.SoftLink):
            dst[key] = h5py.SoftLink(link.path)
        elif isinstance(link, h5py.ExternalLink):
            # A non-data external link: preserve it verbatim.
            dst[key] = h5py.ExternalLink(link.filename, link.path)
        else:
            obj = src[key]
            if isinstance(obj, h5py.Group):
                _copy_structure(obj, dst.create_group(key), skip_paths)
            else:
                dst.copy(obj, key)


def _last_block_index(blocks, total_to_write):
    """Index of the last block that will receive data when writing total_to_write frames.

    Block files beyond this are never created: a real collection of total_to_write frames
    stops the detector there, so those blocks simply never exist on disk.
    """
    last = 0
    for i, b in enumerate(blocks):
        if b["start"] < total_to_write:
            last = i
        else:
            break
    return last


def build_structure(
    template,
    out_dir,
    stem,
    frame_shape,
    dtype,
    blocks,
    compression,
    files_ahead,
    frames_target,
):
    """Create the output master + empty growable block files. Returns list of
    (block, output_source_path) in frame order.

    files_ahead controls how many block files exist up front: 0 creates them all (as on a
    completed collection's directory), while N>0 creates only the first N -- modelling a
    live collection where the detector pre-creates a rolling window of files a few blocks
    ahead of the writer, so distant blocks are genuinely absent from disk. In either case
    no file is created beyond the last block that will actually receive data
    (frames_target), since a collection that stops early never writes those files.
    """
    out_master = out_dir / f"{stem}.nxs"
    last_block = _last_block_index(blocks, frames_target)
    outputs = []
    with h5py.File(template, "r") as src:
        vds = _find_signal_vds(src)
        vds_path = vds.name
        vds_group_path = vds.parent.name
        skip = {vds_path}
        for b in blocks:
            skip.add(f"{vds_group_path}/{b['link_name']}")

        with h5py.File(out_master, "w", libver="latest") as dst:
            _copy_structure(src, dst, skip)
            group = dst[vds_group_path]
            layout = h5py.VirtualLayout(
                shape=(int(vds.shape[0]),) + frame_shape, dtype=dtype
            )
            for i, b in enumerate(blocks):
                out_name = _new_source_filename(b["source_filename"], stem)
                out_path = out_dir / out_name
                # The master (and all its external links + VDS) is written up front, but a
                # link's target data file need only be created when it enters the window,
                # and never beyond the last block that will receive data.
                if i <= last_block and (files_ahead == 0 or i < files_ahead):
                    _ensure_block_file(
                        out_path,
                        b["source_dset"],
                        b["stop"] - b["start"],
                        frame_shape,
                        dtype,
                        compression,
                    )
                group[b["link_name"]] = h5py.ExternalLink(out_name, b["source_dset"])
                source = h5py.VirtualSource(
                    ".",
                    f"{vds_group_path}/{b['link_name']}",
                    shape=(b["stop"] - b["start"],) + frame_shape,
                )
                layout[b["start"] : b["stop"]] = source
                outputs.append((b, out_path))
            group.create_virtual_dataset(
                pathlib.PurePosixPath(vds_path).name, layout, fillvalue=0
            )
    return out_master, outputs


def _ensure_block_file(path, dset_name, block_len, frame_shape, dtype, compression):
    """Create a block data file at its full extent if it does not already exist.

    This mirrors how real detector writers lay out a block: the dataset is allocated
    at its full planned length up front (``shape[0] == block_len`` from creation),
    with one chunk per frame. No pixel data is written yet, so no per-frame chunks are
    allocated -- the frames become "available" only as :func:`write_frames` writes each
    chunk. (A growable/resized dataset would instead misrepresent an empty block as
    length zero, which is not what detectors produce.)
    """
    if path.exists():
        return
    with h5py.File(path, "w", libver="latest") as sf:
        sf.create_dataset(
            dset_name,
            shape=(block_len,) + frame_shape,
            chunks=(1,) + frame_shape,
            dtype=dtype,
            compression=compression,
            compression_opts=1 if compression == "gzip" else None,
        )


def write_frames(
    outputs,
    frame_shape,
    dtype,
    total_to_write,
    rate,
    files_ahead,
    compression,
    logger,
):
    """Fill the block files via SWMR, one frame at a time, up to total_to_write frames.

    The block datasets are preallocated at full extent (as a detector lays them out), so a
    frame becomes available when its chunk is written, not when the extent grows. Each
    frame therefore writes a zero payload into ``dset[i]`` and flushes; the availability
    tools count the allocated chunks. Use --compression none for the fastest writes.

    With files_ahead>0, block files are created lazily so that only that many files exist
    ahead of the writer at any time (matching a detector's rolling pre-creation).
    """
    zero = np.zeros(frame_shape, dtype=dtype)
    interval = 1.0 / rate if rate > 0 else 0.0
    # Never create block files beyond the last one that will receive data.
    last_block = _last_block_index((b for b, _ in outputs), total_to_write)
    written = 0
    next_t = time.monotonic()
    for k, (b, path) in enumerate(outputs):
        if written >= total_to_write:
            break
        if files_ahead:
            # Ensure the window [k, k+files_ahead) of block files exists on disk,
            # clamped to the last block that will actually be written.
            for j in range(k, min(k + files_ahead, last_block + 1)):
                bj, pj = outputs[j]
                _ensure_block_file(
                    pj,
                    bj["source_dset"],
                    bj["stop"] - bj["start"],
                    frame_shape,
                    dtype,
                    compression,
                )
        block_len = b["stop"] - b["start"]
        n_here = min(block_len, total_to_write - written)
        logger(
            f"Filling {path.name}: frames {b['start']}..{b['start'] + n_here - 1} "
            f"({n_here} of {block_len})"
        )
        with h5py.File(path, "r+", libver="latest") as f:
            f.swmr_mode = True
            dset = f[b["source_dset"]]
            for i in range(n_here):
                dset[i] = zero
                dset.flush()
                written += 1
                if interval:
                    next_t += interval
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.monotonic()
        # block file closed here -> visible to the 'file_close' availability method
    return written


def run(args=None):
    dxtbx.util.encode_output_as_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate an NXmx master's structure into empty growable block files, then "
            "fill them via SWMR to simulate a live collection for testing image "
            "availability. The example files are only read, never modified."
        )
    )
    parser.add_argument(
        "template", metavar="MASTER.nxs", help="Example NXmx master file to duplicate"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for the simulated master and block files.",
    )
    parser.add_argument(
        "--stem",
        help="Basename stem for output files (default: same as the template).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        help=(
            "Frames per second to write. Default: derived from the master's count_time "
            "(1/count_time). Override this if the collection rate is too slow (or fast). "
            "Note: writing full frames caps at ~10-15 fps; use --compression none for "
            "faster writes if the disk can keep up."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        metavar="N",
        help="Total frames to write (default: all frames the master declares). The VDS "
        "still declares the full length, so fewer frames simulates a partial collection.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "none"],
        help="Compression for the (all-zero) block datasets (default: gzip, keeps them "
        "tiny). Use 'none' to write uncompressed.",
    )
    parser.add_argument(
        "--files-ahead",
        type=int,
        default=0,
        metavar="N",
        help="How many empty block files exist ahead of the writer. 0 (default) creates "
        "all block files up front. N>0 models a live detector that pre-creates only a "
        "rolling window of N files, so distant blocks are genuinely missing from disk "
        "(exercises the availability tools' missing-source-file handling).",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create the empty duplicated structure and exit without writing frames.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output master file if it already exists.",
    )
    options = parser.parse_args(args)

    template = pathlib.Path(options.template)
    if not template.is_file():
        sys.exit(f"Template not found: {template}")
    out_dir = pathlib.Path(options.output)
    stem = options.stem or template.stem
    compression = None if options.compression == "none" else options.compression

    out_dir.mkdir(parents=True, exist_ok=True)
    out_master = out_dir / f"{stem}.nxs"
    if out_master.exists() and not options.force:
        sys.exit(
            f"Output master already exists (use --force to overwrite): {out_master}"
        )

    # Inspect the template (read-only) for layout and default rate.
    with h5py.File(template, "r") as src:
        total, frame_shape, dtype, blocks = _block_layout(src)
        rate = options.rate
        if rate is None:
            count_time = src.get("entry/instrument/detector/count_time")
            if count_time is not None and float(count_time[()]) > 0:
                rate = 1.0 / float(count_time[()])
            else:
                rate = 10.0

    frames_target = total if options.frames is None else min(options.frames, total)

    print(
        f"Template : {template}\n"
        f"Output   : {out_master}\n"
        f"Frames   : {total} declared, {len(blocks)} blocks, "
        f"frame {frame_shape} {dtype}\n"
        f"Writing  : {frames_target} frames at {rate:g} fps"
        + ("  [setup-only]" if options.setup_only else "")
    )

    out_master, outputs = build_structure(
        template,
        out_dir,
        stem,
        frame_shape,
        dtype,
        blocks,
        compression,
        options.files_ahead,
        frames_target,
    )
    n_created = sum(1 for _, p in outputs if p.exists())
    print(
        f"Created structure: {out_master} + {n_created} block files present"
        + (f" (rolling window of {options.files_ahead})" if options.files_ahead else "")
    )

    if options.setup_only:
        return

    try:
        written = write_frames(
            outputs,
            frame_shape,
            dtype,
            frames_target,
            rate,
            options.files_ahead,
            compression,
            logger=print,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; leaving partially-written collection in place.")
        return
    print(f"Done: wrote {written} frames into {out_dir}")


if __name__ == "__main__":
    run()
