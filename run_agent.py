"""
Single entry point: takes a session root path and auto-discovers everything else by the
naming convention in claude_code_kickoff_prompt.md.

Usage:
    python run_agent.py "D:\\Gil\\Shamir\\20260901_085606.rec"
    python run_agent.py "D:\\Gil\\Shamir\\20260901_085606.rec" --n-units 10
    python run_agent.py "D:\\Gil\\Shamir\\20260901_085606.rec" --units 285,342,295
"""
import argparse
from striatal_agent import run_pipeline, run_structural_screen, AgentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_root", help=r"Path to <session_id>.rec folder")
    parser.add_argument("--units", default="", help="Comma-separated cluster_ids to process (default: all)")
    parser.add_argument("--n-units", type=int, default=None, help="Process only the first N units (default: all)")
    parser.add_argument("--apply-tier1-merges", action="store_true",
                         help="Actually rewrite spike_clusters.npy for tier-1 merges (default: score/log only)")
    parser.add_argument("--screen-only", action="store_true",
                         help="Structural score + cross-unit ACG + footprint concentration only -- "
                              "skips the recovery sweep, collision check, and merge scoring")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    units = [int(x) for x in args.units.split(",") if x.strip()] if args.units else None

    if args.screen_only:
        run_structural_screen(args.session_root, units=units, cfg=AgentConfig(), seed=args.seed)
    else:
        run_pipeline(args.session_root, units=units, n_units=args.n_units,
                     apply_tier1_merges=args.apply_tier1_merges, cfg=AgentConfig(), seed=args.seed)


if __name__ == "__main__":
    main()
