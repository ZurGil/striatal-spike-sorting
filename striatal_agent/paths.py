"""Where this project's session-level outputs live, so the same script runs here and
on the cluster.

Every script under scripts/ that writes a per-session artefact used to open it by
spelling the path out:

    OUT = r"D:\\Gil\\spike_sorting_agent\\outputs"

which is right on the Neuropixels PC and impossible anywhere else. That was fine while
the post-Kilosort half of the pipeline only ever ran on this machine. It stopped being
fine the moment the sort moved to RIS: the raw .dat is on cluster scratch for a few days
and then purged, and every step that READS THE RAW has to happen inside that window or
pay a ~250 GB re-upload. Those steps therefore have to be runnable on Linux, under a
different root, without editing them.

So: set SPIKE_AGENT_OUTPUTS and the scripts follow. Unset -- which is every interactive
use on this PC -- and the old path is used exactly as before, so nothing that worked
yesterday changes.

Only the scripts that must run cluster-side were converted (the agent pipeline batches,
bombcell, the corrected classification they feed, the Trodes channel lookup, and the two
raw_* metrics). The other ~70 are display-side, run only here, and were deliberately
left alone rather than touched en masse for a portability they do not need.
"""
import os

DEFAULT_OUTPUTS_DIR = r"D:\Gil\spike_sorting_agent\outputs"


def outputs_dir():
    """The directory for per-session outputs. SPIKE_AGENT_OUTPUTS wins when set."""
    d = os.environ.get("SPIKE_AGENT_OUTPUTS", "").strip()
    return d if d else DEFAULT_OUTPUTS_DIR


def out_path(*parts):
    """A path inside outputs_dir(), joined the way the running OS spells paths."""
    return os.path.join(outputs_dir(), *parts)
