"""Write a standalone <session>.trodesconf out of a .rec file's own header.

Why this exists
---------------
The post-Kilosort job on the cluster needs the Trodes configuration for two of its
steps: the trodes channel lookup (step 5) and the raw separation metric (step 6b,
which reads step 5's CSV). But only probe1.dat and the channel map are uploaded to
the cluster -- the 300+ GB .rec stays on Z -- so on RIS there is nothing to extract
the configuration from, and both steps fail. They fail late, after the GPU hours are
already spent, and they fail quietly enough that the session looks sorted.

A .rec begins with exactly that configuration as XML, so the fix is to carve the
header off locally and send the few hundred kilobytes up beside the data.

    python scripts/extract_trodesconf_from_rec.py 20260917_100043 --rat Shamir

The parse is deliberately the same shape as the one in
build_trodes_channel_lookup_session.py: find the last closing tag of the
configuration block and cut immediately after it. Anything else risks writing a
truncated XML file that parses as valid and yields a silently short channel list,
which is a worse outcome than no file at all -- so the count of SpikeNTrode entries
is checked before the file is written.
"""
from __future__ import print_function

import argparse
import os
import re
import sys

ZROOT = {
    "Shamir": r"Z:\Gil\Shamir_1",
    "Popper": r"Z:\Gil\Popper",
    "Begin": r"Z:\Gil\Begin_1",
}

HEADER_BYTES = 5_000_000
MIN_NTRODES = 300          # a 384-channel probe; a short parse means a bad cut


def extract(rec_path):
    with open(rec_path, "rb") as fh:
        header = fh.read(HEADER_BYTES)
    text = header.decode("utf-8", errors="replace")
    end = max(text.find("</SpikeConfiguration>"), text.rfind("</Configuration>"))
    if end == -1:
        raise SystemExit("no </Configuration> or </SpikeConfiguration> in the first "
                         "%d bytes of %s -- not a Trodes .rec header"
                         % (HEADER_BYTES, rec_path))
    return text[:text.find(">", end) + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--rat", default="Shamir", choices=sorted(ZROOT))
    ap.add_argument("--rec_root", default=None,
                    help="override the archive root (default: the rat's Z root)")
    ap.add_argument("--out", default=None,
                    help="where to write (default: outputs/<sid>.trodesconf)")
    args = ap.parse_args()

    sid = args.session_id
    root = args.rec_root or ZROOT[args.rat]
    rec = os.path.join(root, sid + ".rec", sid + ".rec")
    if not os.path.exists(rec):
        raise SystemExit("no such .rec: %s" % rec)

    conf = extract(rec)
    ntrodes = re.findall(r'<SpikeNTrode[^>]*id="(\d+)"', conf)
    print("%s: header carved, %d bytes, %d SpikeNTrode entries"
          % (sid, len(conf), len(ntrodes)))
    if len(ntrodes) < MIN_NTRODES:
        raise SystemExit("only %d SpikeNTrode entries (expected >= %d). Refusing to "
                         "write a file that would parse but be short -- a truncated "
                         "config is worse than none, because the lookup would succeed "
                         "on a subset and nothing downstream would notice."
                         % (len(ntrodes), MIN_NTRODES))

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs", sid + ".trodesconf")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(conf)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
