# Questions for Unmukt about RIS — with the jargon explained

Written so the terms make sense before the questions do. Nothing here needs a
computing background.

## What we are trying to do, in two sentences

Spike sorting (Kilosort) currently runs on the Neuropixels recording PC, and it
cannot run at the same time as Trodes, so every sort costs us a recording slot.
We want to run it on WashU's shared computers instead: about **30 sessions
backlogged now**, then roughly **one new session per day**.

## The money question, already answered

**It is free.** RIS's own page says Compute1 "will remain operational at no cost
until that date", where the date is its **end-of-life, 1 July 2027**. So there is
no per-hour charge to negotiate, for CPU or GPU. Availability is the only
currency.

Two deadlines come with that, and the first one is soon:

- **1 January 2027** — Compute1 "General Access allocations stop accepting new
  users". If that is the kind of allocation we need, we have to be signed up
  before then. That is about **eight weeks away**.
- **1 July 2027** — everything must be off Compute1. The long-term home is
  **Compute2**, which is *not* free in the same way: it bills in "Standard
  Billing Units" against a yearly faculty subsidy (reported as ~$2,800
  equivalent, plus 5 TB of free storage).

So the backlog should run on Compute1 while it costs nothing, and we should find
out early what Compute2 will cost for the steady one-session-a-day load.

## The terms, in plain words

| term | what it means for us |
|---|---|
| **Compute1 / Compute2** | Two generations of WashU's shared computer cluster, not two machines. Compute1 is the old one, free, switched off July 2027. Compute2 is the new one, billed. |
| **node** | One physical computer in the cluster. A job runs on one node. |
| **core** | One processing unit inside a node. Asking for "8 cores" means 8 of them work on our sort together. More cores = faster, but not proportionally. |
| **CPU** | The ordinary processor. Flexible, and for Kilosort **slow**. |
| **GPU** | The graphics card. Kilosort is written for it and is roughly **30× faster** on one. Ours needs only a small GPU — 3.3 GB of its memory. |
| **job** | One unit of work we hand to the cluster: "sort this session." |
| **queue** | The waiting line for jobs. Different queues have different limits. |
| **LSF / `bsub`** | The cluster's booking system, and the command that submits a job to it. We don't log in and run things by hand; we submit and it schedules. |
| **walltime** | The maximum time a job is allowed to run before being killed. Important for us: a CPU sort may need **5–6 days**, and some queues cap jobs well below that. |
| **Docker / container / image** | RIS requires every job to run inside a pre-packaged software bundle (the "image"), so our exact Kilosort version travels with the job. Already built: `docker/Dockerfile.kilosort4`. |
| **`storage1`** | The lab's **permanent** space. Quota'd — this is what the "2 TB" probably refers to. Backed up. |
| **`scratch1`** | **Temporary working space.** Large, shared, and **automatically deleted** after a while. Meant exactly for big files you need during a computation and not afterwards. |
| **quota** | The size cap on our space. |
| **Kerberos ticket** | A login pass that expires after some hours. If it expires while a job is running, the job can **lose access to storage and die**. Matters because our CPU jobs run for days. |
| **Globus** | A tool for moving large amounts of data, faster and more reliably than dragging files. Possibly how we send 325 GB per session. |

## Why `scratch1` vs `storage1` is the question I kept asking badly

Each session needs **two very different files**:

| | size | needed |
|---|---|---|
| `probe1.dat` — the recording Kilosort reads | **~325 GB** | only *during* the sort |
| Kilosort's results | **~19 GB** | **forever** |

So the big file is temporary and the small file is permanent. That matters
because:

- If the 325 GB has to sit inside our 2 TB permanent space, we can only fit
  **about 5 sessions at a time**, and that caps how many we can sort at once.
- If the 325 GB can sit on **`scratch1`** instead, then our 2 TB only ever holds
  results at 19 GB each — enough for **~100 sessions** — and the limit
  disappears.

That's the whole question: **where does the big temporary file live?**

## The questions, in the order they matter

**1. Can we get an allocation on Compute1 before 1 January 2027?**
That is the cutoff for new General Access users, and Compute1 is the free one.
*(If we miss it, we may be straight onto the billed Compute2.)*

**2. Where should the big 325 GB file go while a session is being sorted — our
2 TB, or `scratch1`?** And if `scratch1`: how much space, and how long before
files get deleted?
*Why it matters: this decides whether we can sort 5 sessions at a time or many
more. It is the single most useful answer.*

**3. How many jobs can we run at the same time?**
*Gil was told "several CPUs, probably a few of them" — we want the actual number.
With 5 at once the backlog takes about a month; with 10, about two weeks.*

**4. How many CPU cores can one job ask for?**
*We measured Kilosort on 8 cores. More would help, though less than
proportionally — worth knowing the ceiling before we test.*

**5. Is there any GPU we could use, and how busy is it?**
*This is the big one for speed: ~30× faster. We need a **small** GPU — 3.3 GB of
graphics memory, about 4.5 hours per session, which is roughly 20% of one GPU
running continuously. Since Compute1 is free, this is a question about
availability and queue time, not cost. Also: which queue, and does it need the
`gpuhost` flag we saw in the docs?*

**6. How long is a single job allowed to run?**
*A CPU sort may need 5–6 days. If the queue cap is shorter, we have to split the
work or use a GPU.*

**7. What is the best way to send ~325 GB per session from a Windows PC?**
*Can we map RIS storage as a network drive (`\\storage1.ris.wustl.edu\...`) and
copy to it, or should we use Globus? And do we need the VPN — this PC is on the
lab network at 172.20.22.115, and it must keep talking to the recording hardware
while copying.*

**8. Do we build the Docker image ourselves and push it somewhere, or does RIS
build it from our file?**
*We have the file written already; we just need to know where it goes.*

**9. Will a multi-day job lose its storage access when the Kerberos ticket
expires?**
*If so, the fix is to have jobs write to `scratch1` and copy results to permanent
storage in a separate short job — but better to know in advance than to lose a
five-day sort at hour 100.*

**10. From your experience — is any of this a bad fit for RIS?**
*Unmukt may simply know that e.g. 5-day CPU jobs get killed in practice, or that
the GPU queue is hopeless, or that Globus is the only sane transfer route. Worth
asking directly rather than discovering it.*

## What we have already built, so he knows what's not needed

- The Docker image definition, with the exact Kilosort version pinned.
- The job submission script (needs the allocation's group and queue names filled in).
- The staging script that copies a session over and generates the probe map first
  — the map has to be built on the Windows PC, because it needs the Trodes
  software, which does not exist on the cluster.
- A benchmark we can re-run on the cluster to measure its own speed rather than
  guessing from ours.

## Numbers he may ask for

| | |
|---|---|
| sessions backlogged | ~30 |
| ongoing rate | ~1 session/day |
| recording length | 3.0–4.8 h, mean 4.13 h |
| channels / rate | 384 at 30 kHz, int16 |
| input file per session | ~325 GB (`probe1.dat`) |
| results per session | ~19 GB |
| measured CPU speed | ~30× slower than real time on 8 cores → **~5 days/session** |
| measured GPU speed | 1.1× real time on an RTX 2060 → **~4.5 h/session** |
| GPU memory needed | **3.3 GB peak** |
| software | Kilosort 4.0.24, PyTorch 2.4.0, Python 3.9 |
