# Running Kilosort4 on WashU RIS Compute1

Goal, in Gil's words: **leave the Neuropixels PC free for Trodes**, use a free
compute allocation to sort **several sessions at once**, and possibly run the
review-tool pipeline there too.

Status 2026-11-04: local side built, allocation not yet requested. Nothing has
been connected to.

> **This replaces an earlier draft of this file that assumed the wrong
> architecture.** That draft assumed generic SSH nodes that could mount the
> archive share directly. They cannot. The real target is WashU RIS Compute1, an
> **LSF + Docker** cluster with **no route to the lab share**, so data has to be
> copied in and results copied back. The earlier "just mount Z" plan is dead.

## The architecture, as it actually is

```
Z:\Gil\Shamir_1          this PC             RIS storage1 / scratch1       RIS compute
(archive share)       (stage + map)            (2 TB allocation)          (LSF + Docker)
      |                     |                         |                         |
  probe1.dat  --------> robocopy ------------>  scratch1/<sid>/  ---------->  bsub -a docker()
  channelmap  --> make_probe_map.py --> map.json        |                    kilosort4 on CPU
      |                                                 |                         |
  results  <--------- robocopy <------------- storage1/kilosort4/<sid> <----------+
```

Four consequences worth stating plainly:

- **This PC is in the data path after all**, as the thing that copies. That is
  unavoidable without a RIS↔share route, but it is a copy, not a compute load,
  so it does not block Trodes the way sorting does.
- **The probe map must be generated here.** `make_probe_map.py` imports the
  Trodes python reader from `C:\Users\Adam\Documents\Trodes_2-4-2_Windows64\...`,
  which does not exist on a compute node. So the map is built on Windows and
  travels with the data. `scripts/stage_session_to_ris.ps1` does this in the
  right order and refuses to copy if the map step failed — this is the piece most
  likely to be forgotten, which is why it is code and not a checklist item.
- **Every job runs in a container.** RIS requires it. Hence
  `docker/Dockerfile.kilosort4`.
- **Results must come back to Z**, because the RIS inputs get purged to make
  room for other people.

## The number that decides everything: Kilosort4 on CPU

Measured today, same 120 s × 384 ch fixture (`hybrid_small/`), same thresholds
(`Th_universal` 9.0, `Th_learned` 8.0), on this box's Ryzen 7 3700X:

| | wall clock for 120 s of recording | × realtime |
|---|---|---|
| **GPU** (RTX 2060, from the 2026-09-30 run log) | **131.7 s** | 1.10 |
| **CPU**, 8 threads | ~1 h (run in progress at time of writing) | ~30 |

Be precise about what is measured here, because the conclusion rests on it. Two
things are **directly measured**: the GPU total of 131.7 s, and the CPU
spike-detection stage running at a steady **~31 s per batch over 60 batches**
≈ 30 min, against **65.1 s** for that same stage on the GPU — a **28×** ratio on
the stage that dominated the GPU run. The CPU *total* is **extrapolated** from
that ratio and will be replaced with the measured figure when the run finishes;
`scripts/benchmark_kilosort_cpu.py` writes it to `benchmark_summary.json`
alongside peak memory.

**What that means for a real session.** At ~30× realtime, a 4.5 h recording is
roughly **5–6 days** of CPU wall clock on 8 cores. The same sort takes about
**5 hours** on this PC's GPU. Treat the days figure as an order of magnitude,
not a schedule, until the node's own benchmark is in.

So the honest framing is **throughput, not speed**:

| approach | per session | concurrent | sessions/day | blocks Trodes? |
|---|---|---|---|---|
| local GPU | ~5 h | 1 | ~4–5 | **yes, completely** |
| RIS CPU, 8 cores | ~5–6 days | ~5 (storage-bound) | ~1 | no |

For **clearing the 32-session backlog** that is a good trade: about a month of
entirely unattended background sorting that costs nothing and never touches the
recording rig. For **keeping up with daily recording** it is roughly break-even
and tight.

Two things would change the picture, in order of leverage:

1. **A GPU queue.** 30× is the difference between 5 hours and 6 days. Even
   occasional or low-priority GPU access would dominate every other
   optimisation here. **This is now the most important question to ask** — it
   has replaced the share-mount question from the earlier draft.
2. **More cores.** Worth measuring on the node with
   `scripts/benchmark_kilosort_cpu.py --threads N` before requesting a wide
   allocation. KS4's CPU scaling is sub-linear, and a wide request queues longer
   for a gain that may not be there.

## The storage budget — and a question hiding in it

Measured per session:

| item | size |
|---|---|
| `probe1.dat` input | 252–370 GiB (mean ~325) |
| `timestamps.dat` | ~1.9 GiB |
| probe map JSON + channelmap | < 1 MiB |
| **Kilosort4 output** | **19.2 GiB** (733 files; `pc_features.npy` is 5.6 GiB of it) |

So ~345 GiB in flight per session, and **2 TB holds about 5 sessions at once**.
That — not cores — is what caps concurrency.

**But the 2 TB may be the wrong number to plan against.** RIS has two
filesystems: `storage1` (persistent, quota'd, where the allocation lives) and
`scratch1` (large, shared, purged). The *outputs* are only 19 GiB each, so 2 TB
of `storage1` would hold **100 sessions of results** comfortably. It is the
*inputs* that are enormous, and those belong on scratch. **If the 2 TB is
storage1 and scratch is separate, concurrency is set by scratch, not by the
allocation** — and could be much higher than 5.

Worth asking explicitly rather than assuming the pessimistic reading.

## Sequencing insight: sort and build review data in the same window

If the review-tool pipeline is going to run on RIS too, it has to run **while
the input is still staged**. Waveform extraction needs `probe1.dat`; the KS4
output alone is not enough. Purge the input, decide later you want waveforms,
and it is another 325 GiB copy.

So the unit of work per session is: **stage → sort → build review data → copy
results back → purge input.** Not "sort everything, then review everything."

## Ask Sarah / RIS

**To get started at all:**

1. **The allocation**: compute group for `bsub -G` (e.g. `compute-<PI>`), and the
   PI key id that appears in `/storage1/fs1/<PI>/Active` and `/scratch1/fs1/<PI>`.
2. **WashU Key access**, and membership of the `storage-<PI>` AD group — that is
   what gates the SMB mount of `storage1` from this PC.
3. **Confirmation that `compute1` is the right platform** and it is free to use
   until **July** (as Gil was told). RIS docs show compute1 pages migrating to a
   **Compute2** space, so check which cluster the allocation is actually on.
4. **VPN**: this PC is at `172.20.22.115`. RIS storage requires being on the
   WashU network or VPN. Which VPN, and can it run alongside the lab network
   without disturbing the acquisition link on `192.168.0.2`?

**Before the first real run:**

5. **Is there any GPU queue available to this allocation?** See above — this is
   the highest-leverage question of the lot. If yes: queue name, and whether
   `-R 'gpuhost' -gpu "num=1"` is the right incantation there.
6. **Is the 2 TB `storage1` or `scratch1`?** And what is the scratch quota and
   purge policy? This sets concurrency.
7. **Queue limits**: max walltime and max cores per job on the CPU queue. A
   5–6 day job needs a queue that permits it; if the cap is shorter, the sort has
   to be split or checkpointed. `bqueues -l general` answers this from a client
   host.
8. **How many jobs may we have running at once?**
9. **Docker registry**: where do we push the image so RIS can pull it, or should
   RIS build it from the Dockerfile?
10. **Kerberos**: what is the maximum renewable ticket lifetime? An expiring
    ticket breaks a running job's access to RIS storage, and a multi-day sort is
    well past the default. Writing results to scratch and copying them to
    storage in a separate short job is the workaround if the lifetime is short.
11. **Transfer method for ~325 GiB per session**: SMB mount from this PC, or
    Globus? Globus is likely much faster and is what RIS recommends for bulk
    data.

## What is built and ready

| file | what it does |
|---|---|
| `docker/Dockerfile.kilosort4` | CPU image, pins verified **at build time** so a job cannot discover a wrong version six hours in. No `ENTRYPOINT` and `/bin/sh` present, as LSF requires. GPU variant documented at the bottom. |
| `scripts/run_kilosort4_session.py` | The existing runner with paths and thresholds as arguments. **Refuses** to fall back to CPU if `--device cuda` was asked for, refuses a probe whose channel count disagrees, and writes `run_provenance.json` beside the results. |
| `scripts/ris_submit_kilosort.sh` | `bsub` template: `LSF_DOCKER_VOLUMES`, `span[hosts=1]`, walltime, `kinit` check. Allocation-specific values marked `PENDING`. |
| `scripts/stage_session_to_ris.ps1` | Generates the probe map, sanity-checks it, checks destination free space, then robocopies `probe1.dat` + `timestamps.dat` + channelmap + map JSON + DIO. Writes nothing to the archive. |
| `scripts/benchmark_kilosort_cpu.py` | The CPU benchmark above. **Re-run it on the node** — the number that matters is the node's, not this PC's. |
| `~/.ssh/id_ed25519_compute` + config | From the earlier draft. Still useful for `compute1-client-*`, though RIS logins are WashU-Key based, so the key may not be the auth path. |

## Two hazards the remote path must not walk into

Both produce output that looks completely normal and is wrong, so both are
enforced in code rather than documented and hoped for:

- **The KS4 thresholds are per session.** 20260916 used `Th_universal` 9.0 /
  `Th_learned` 8.0; 20260910 and 20260901 used 9.2 / 7.0. They are **required
  arguments** in `run_kilosort4_session.py` — no default, because a default is
  exactly how the wrong pair gets used silently.
- **There are two channel maps**, differing by a silent **820 µm depth shift**
  (map A through 2026-10-23, map B from 2026-10-26). The runner prints and
  records the probe's y-coordinate range, which is the only thing that
  distinguishes them by eye, and the staging script prints it before the copy.

## Before trusting any of it

**Re-sort a session that has already been sorted here, and compare the output.**
The point of this exercise is results that belong in the same analysis as the
existing ones. A remote setup that quietly produces different output is not a
faster pipeline, it is a second pipeline. `20260916_110311` is the obvious
candidate — it is the current working session and its KS4 output is on F.

## What is NOT known — do not assume

- **Nothing about the allocation has been verified**: no group, no paths, no
  queue, no GPU, no quota split between storage1 and scratch1. Every statement
  here about RIS is from public documentation plus Gil's relay, not from a
  login.
- **Whether a 5–6 day job is permitted** on the CPU queue.
- **Whether the review-tool pipeline runs unmodified on Linux.** It has only
  ever run on Windows here.
- **The exact RIS hostnames.** The docs use both `wustl.edu` and `washu.edu`;
  verify against the allocation.

Measurements about *this* PC, the archive, and the CPU/GPU benchmark were taken
on 2026-11-04 and are real.

Sources for the RIS platform details: the WashU RIS wiki pages on
[Docker and the RIS Compute1 Platform](https://washu.atlassian.net/wiki/spaces/RUD/pages/1705115761/Docker+and+the+RIS+Compute1+Platform),
[Storage1 Access Control](https://washu.atlassian.net/wiki/spaces/RUD/pages/1795948625/Storage1+Access+Control),
[Connecting to Storage from Linux](https://washu.atlassian.net/wiki/spaces/RUD/pages/1795260493/Connecting+to+Storage+from+Linux),
and the [Compute2 Cluster](https://washu.atlassian.net/wiki/spaces/RUD/pages/3307929753) page.

## A note on `make_probe_map.py`

The canonical copy Gil runs lives at `D:\Gil\make_probe_map.py`, **outside this
repo and tracked nowhere**. Since the remote path depends on it entirely, a
byte-identical snapshot is now committed at `scripts/make_probe_map.py`
(sha256 `d1bdd7ba6abc1c65...`, verified identical on 2026-11-04).

`stage_session_to_ris.ps1` still defaults to the `D:\Gil` copy so Gil's existing
workflow is unchanged. That means the two can drift — if the canonical one is
edited, re-copy it into the repo.
