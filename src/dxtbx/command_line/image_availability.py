# LIBTBX_SET_DISPATCHER_NAME dev.dxtbx.image_availability
"""
Live-monitor how much of an NXmx master file has actually been written to disk.

During serial (SSX) data collection at DLS the master ``.h5``/``.nxs`` file and its
underlying ``data_*.h5`` sources are created up front (the virtual dataset declares the
full planned number of frames) and then filled incrementally, typically via SWMR. File
presence therefore does not mean the image data is present.

This utility does *no* processing: it simply polls the availability tools in
``dxtbx.nexus`` and shows, live, how many frames of each given master are genuinely
written so far. Handy for watching a collection fill up, or for debugging the
``xia2.ssx wait_for_images`` live-processing mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import dxtbx.util
from dxtbx.nexus import get_frame_counts


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _bar(fraction: float, width: int = 30) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)


def _format_line(name: str, avail: int, total: int, rate: float | None) -> str:
    if total <= 0:
        return f"{name}  [{_bar(0)}] unreadable / no frames declared"
    fraction = avail / total
    body = f"[{_bar(fraction)}] {avail}/{total} ({100 * fraction:5.1f}%)"
    if avail >= total:
        extra = "complete"
    elif rate and rate > 0:
        eta = (total - avail) / rate
        extra = f"+{rate:6.1f}/s  ETA {_fmt_duration(eta)}"
    elif rate == 0:
        extra = "stalled"
    else:
        extra = "waiting…"
    return f"{name}  {body}  {extra}"


def run(args=None):
    dxtbx.util.encode_output_as_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Live-monitor how many image frames of NXmx master file(s) have been "
            "written to disk. Does no processing — only inspects data availability."
        )
    )
    parser.add_argument(
        "master", nargs="+", metavar="MASTER.h5", help="NXmx master file(s) to watch"
    )
    parser.add_argument(
        "--method",
        choices=["swmr", "file_close"],
        default="swmr",
        help=(
            "How to test source-file readiness: 'swmr' reads the current written extent "
            "(per-image); 'file_close' waits until the writer closes each source file "
            "(per-block). Default: swmr."
        ),
    )
    parser.add_argument(
        "-n",
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Polling interval in seconds (default: 2).",
    )
    parser.add_argument(
        "-1",
        "--once",
        action="store_true",
        help="Print a single snapshot and exit (no live updating).",
    )
    parser.add_argument(
        "--watch-complete",
        action="store_true",
        help=(
            "Keep polling even after every file is complete (default: exit once all "
            "files reach their full frame count)."
        ),
    )
    options = parser.parse_args(args)

    masters = options.master
    labels = [os.path.basename(m) for m in masters]
    width = max(len(label) for label in labels)
    labels = [label.ljust(width) for label in labels]

    # Redraw in place only for an interactive terminal doing live updates.
    live = not options.once and sys.stdout.isatty()

    previous: dict[str, tuple[float, int]] = {}

    def poll_once() -> bool:
        """Render one frame of output; return True when every file is complete."""
        now = time.monotonic()
        lines = []
        all_complete = True
        for master, label in zip(masters, labels):
            avail, total = get_frame_counts(master, options.method)
            rate: float | None = None
            if master in previous:
                prev_t, prev_n = previous[master]
                dt = now - prev_t
                if dt > 0:
                    rate = (avail - prev_n) / dt
            previous[master] = (now, avail)
            lines.append(_format_line(label, avail, total, rate))
            if total <= 0 or avail < total:
                all_complete = False
        if live:
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
        else:
            print("\n".join(lines))
        return all_complete

    if options.once:
        poll_once()
        return

    try:
        while True:
            complete = poll_once()
            if complete and not options.watch_complete:
                break
            time.sleep(options.interval)
            if live:
                # Move the cursor back up over the block we just printed.
                sys.stdout.write(f"\033[{len(masters)}F")
                sys.stdout.flush()
    except KeyboardInterrupt:
        # Leave the last rendered frame visible and exit quietly.
        print()


if __name__ == "__main__":
    run()
