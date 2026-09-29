"""
post_ks_correction -- the "three pillars" post-Kilosort correction module
(see striatal_spike_sorting_handoff.md for the full design).

Distinct from striatal_agent/: that package is pure post-hoc analysis feeding
the review tool and has never, in practice, modified a spike train (verified
2026-09-25 -- the one real agent_report.json run on 20260901_085606 landed
every unit in tier2_scored_logged with an empty child_cluster_id_map).

This package's actual purpose is to perform real corrective sorting (recover
missed spikes, resolve collision/residual splits) on the raw KS4 output,
upstream of the existing review tool -- not another diagnostic layer next to
it. See git tag `review-tool-stable` for the protected baseline this must
never silently break.
"""
