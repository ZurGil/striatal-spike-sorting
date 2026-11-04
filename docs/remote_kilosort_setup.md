# Running Kilosort4 on compute1 / compute2

Goal: sort sessions on the lab compute nodes instead of on the Neuropixels PC.
This file is both the plan and the request list — the "Ask Sarah" section below
is written to be forwarded as-is.

Status as of 2026-11-04: **local side ready, waiting on access.** Nothing has
been connected to yet, because no hostname exists yet.

## Why move the sorting off this PC

Three reasons, in order of how much they actually matter:

1. **This PC cannot record and sort at the same time.** Kilosort and Trodes both
   want the GPU, and the failure mode is not an error — Kilosort **hangs
   silently** with Trodes open. So today, sorting blocks data collection for the
   whole run. Sorting elsewhere removes that interlock entirely, and that is the
   biggest single win.
2. **VRAM headroom.** The GPU here is an **RTX 2060, 6.4 GB** (compute
   capability 7.5). That is the low end for 384-channel Kilosort4; the drift and
   clustering stages are where it gets tight.
3. **Wall-clock.** A newer datacentre or current-gen card is several times
   faster on the same file. Worth having, but it is the least important of the
   three.

## What is already set up on this machine — DONE

| item | value |
|---|---|
| hostname | `KEPXNEUROPIXELS` |
| local user | `Adam` |
| lab-network IP | `172.20.22.115/24` (interface "Internet") |
| acquisition IP | `192.168.0.2/24` (interface "SpikeGadgets" — isolated, not for this) |
| SSH client | OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2 |
| key type | ed25519, **no passphrase** (see note) |
| private key | `C:\Users\Adam\.ssh\id_ed25519_compute`, ACL restricted to `Adam` only |
| public key | `C:\Users\Adam\.ssh\id_ed25519_compute.pub` |
| fingerprint | `SHA256:x6hiYyUqDtfn14nQy8cJdqiO6X9ukPRKIDti0BdrFno` |
| config | `C:\Users\Adam\.ssh\config` — `compute1` / `compute2` blocks, hostnames pending |

Deliberately **not** changed: the `ssh-agent` service is still Disabled. It is
not needed, because the key has no passphrase — one less system service altered.

**The passphrase choice is Gil's to revisit.** The key is unencrypted so that
long unattended jobs can be launched without anyone typing anything. The cost is
that the key file alone is enough to log in, so it must not be copied off this
machine or into a backup that leaves it. If you would rather it were encrypted,
say so and I will regenerate it with a passphrase held in the Windows ssh-agent
(which does persist across reboots, so unattended use still works).

### The public key to install

This is the line that goes into `~/.ssh/authorized_keys` on each node:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEYnS47aZP4O8C3b7qVAgAH6bAV0GAWuh6wPhsRs9WHM gilzur302@gmail.com neuropixels-pc kilosort-remote
```

`ssh-copy-id` does not ship with Windows OpenSSH. Once there is a hostname and a
password, this is the equivalent one-liner from PowerShell on this PC:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519_compute.pub | ssh USER@HOST "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

## Ask Sarah — to connect at all

1. **Hostname or IP of compute1 and compute2.**
2. **Is either reachable directly from `172.20.22.115`**, or is there a jump
   host / VPN in the way? (The config here has a commented `ProxyJump` block
   ready if so.)
3. **Account name** on the nodes, and whether we install our own public key or
   they install it. The key is in the section above.
4. **Does anything need allowlisting** for this PC to reach them on port 22?

## Ask Sarah — needed before the first real run

5. **Can the nodes mount `\\172.20.22.203\NeuroPixels`?** This is the most
   consequential question of the lot — see "the data problem" below. Read access
   is enough to start; write access would let results land beside the data.
6. **GPU model, VRAM, and how many per node.**
7. **Is there a scheduler** (Slurm / PBS), or do jobs just run on the node? If
   Slurm: partition name, max walltime, and how to request a GPU. A sort is a
   multi-hour single-GPU job.
8. **Container or environment story**: Docker, Apptainer/Singularity, or plain
   conda? And can we install into our own home directory without root? Plain
   conda in `$HOME` is the easiest path and needs no privileges — see the
   environment spec below.
9. **Local scratch**: path and free space per node. Budget **~400 GB per
   session** for the input alone, plus Kilosort4's own output.
10. **Home directory quota.**
11. **OS and NVIDIA driver version.** This picks the torch build; the driver has
    to be new enough for the CUDA runtime we install.
12. **Usage policy**: shared node or exclusive? Is a 6+ hour job acceptable, and
    should it be scheduled off-hours?

Items 1–4 are enough for me to connect and survey the rest myself.

## The data problem — and why question 5 decides it

What Kilosort4 actually needs is not the `.rec`. It needs the `trodesexport`
products, and **those are already on the archive share**. Measured on
`Z:\Gil\Shamir_1\20261021_111842.rec\20261021_111842.kilosort\`:

| file | size |
|---|---|
| `20261021_111842.probe1.dat` | **369.87 GiB** |
| `20261021_111842.timestamps.dat` | 1.93 GiB |
| `20261021_111842.channelmap_probe1.dat` | <1 MiB |

The share is `\\172.20.22.203\NeuroPixels`, mounted here as `Z:`, with **42.3 TB
free** and **32 archived sessions**. The share host is on the same /24 as this
PC, so the nodes plausibly reach it too.

That gives two very different plans:

- **If the nodes can mount the share** — nothing is copied from this PC, ever.
  The node reads `probe1.dat` straight from the archive (or, better, stages it
  to local scratch at server-to-server speed), sorts, and writes results back.
  This PC is not in the data path at all and can keep recording.
- **If they cannot** — every session has to be pushed from here. Measured
  throughput on this link is **~113–118 MiB/s**, so 270–370 GiB is **roughly
  7–9 hours per session**, during which this PC is busy. That is worse than
  sorting locally for all but the largest jobs, and it would need a different
  approach: export on the node from the `.rec`, or get the nodes onto the share.

So question 5 is not a detail — it decides whether this is worth doing. Worth
asking Sarah directly rather than discovering it later.

One caveat even in the good case: reading 370 GiB over SMB *during* sorting will
be slower than local disk, because Kilosort4 passes over the file more than
once. The right pattern is copy-to-scratch, sort, copy-results-back — hence
question 9.

## Environment to reproduce on the node

Exactly what works here today, read out of the `kilosort4` conda env on this PC:

| package | version |
|---|---|
| Python | 3.9.19 |
| `kilosort` | **4.0.24** |
| `torch` | 2.4.0 (CUDA build **11.8**) |
| `numpy` | 1.26.4 |
| `scipy` | 1.13.1 |

Pin `kilosort` to 4.0.24 rather than taking latest — every sorted session in
this project was produced by it, and a version change would make new results
incomparable to the existing ones without a deliberate re-run.

The torch CUDA build may need to move up from 11.8 depending on the node's
driver and GPU generation (a current-gen card may require a newer CUDA). That
is a legitimate change; the `kilosort` pin is not.

A no-root install in the home directory:

```bash
# on the node
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda create -y -n kilosort4 python=3.9.19
$HOME/miniconda3/bin/conda run -n kilosort4 pip install \
    "kilosort==4.0.24" "numpy==1.26.4" "scipy==1.13.1"
# torch LAST, from the index matching the node's CUDA -- confirm the version first
$HOME/miniconda3/bin/conda run -n kilosort4 pip install \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

Install torch last and from the explicit CUDA index; letting pip resolve it as a
`kilosort` dependency pulls a build that may not match the node's driver.

### Verifying the GPU before trusting a run

```bash
python -c "import torch,kilosort; p=torch.cuda.get_device_properties(0); \
print(kilosort.__version__, torch.__version__, torch.version.cuda, p.name, \
round(p.total_memory/1e9,1), 'GB')"
```

If `torch.cuda.is_available()` is False, stop — Kilosort4 will fall back in ways
that waste hours rather than failing cleanly.

## The sort itself

The runner used here is `scripts/run_kilosort4_20260916.py`. Settings that are
not Kilosort4 defaults, and must be carried across deliberately rather than
re-guessed:

| setting | value | note |
|---|---|---|
| `n_chan_bin` | 384 | |
| `fs` | 30000.0 | |
| `Th_universal` | 9.0 | **session-specific** — 20260910/20260901 used 9.2 |
| `Th_learned` | 8.0 | **session-specific** — those sessions used 7.0 |
| `do_CAR` | True | KS4 default |
| `invert_sign` | False | KS4 default |
| `data_dtype` | int16 | |

The thresholds differ per session on purpose. Do not standardise them on the
node without checking which session is being sorted and what it was run with
before.

Each session also needs its **probe map JSON**, generated per session. Note
there are two channel maps in play in this project, differing by a silent 820 µm
depth shift (map A through 2026-10-23, map B from 2026-10-26). The remote setup
must carry the *same* map the session was recorded with; getting this wrong
produces plausible-looking output with wrong unit depths.

## Order of work once access exists

1. `ssh compute1` with the key — confirm login and no password prompt.
2. Survey the node: GPU, driver, scratch, quota, scheduler, whether the share is
   mountable. This answers questions 5–12 without needing Sarah again.
3. Install the environment above in `$HOME`; verify with the GPU one-liner.
4. **Smoke test on a small file before any real session.** A full sort is hours;
   a short recording proves the environment end to end in minutes. There are
   small test fixtures in `hybrid_small/` that can serve.
5. Sort one session that has already been sorted here, and compare the output to
   the local result. This is the step that proves the remote setup is
   trustworthy, and it is worth the GPU time.
6. Only then move the real queue across.

Step 5 matters more than it looks: the point of this exercise is results that can
sit in the same analysis as the existing ones. A remote setup that produces
*different* output is not a faster pipeline, it is a second pipeline.

## What is NOT known yet — do not assume these

- **Nothing about compute1/compute2 has been verified.** No hostname, no
  account, no GPU model, no scheduler, no mount. Every statement in this file
  about the nodes is a question, not a measurement.
- **Whether the nodes can see the archive share.** The share host is on the same
  subnet as this PC, which makes it plausible and nothing more.
- **Whether the lab's policy allows a multi-hour GPU job**, or expects it
  scheduled.

Measurements in this file about *this* PC, the archive, and the local
environment were taken on 2026-11-04 and are real.
