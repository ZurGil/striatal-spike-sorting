# Adding a FLIR camera to the rig, synced to the neural data

Design decisions and the reasoning, so the implementation on the camera PC does
not have to re-derive them.

## 1. Separate computer — yes

Run the camera on its own machine. Reasons, strongest first:

1. **Trodes acquisition is latency-sensitive and its losses are permanent.** A
   dropped packet is neural data that does not exist any more. Video can be
   re-recorded tomorrow; a session cannot.
2. **Compression is CPU-heavy.** This box is a Ryzen 7 3700X (8 cores) already
   feeding Trodes at ~23 MB/s continuously.
3. **Disk contention.** Trodes writes continuously to `F:`, which also holds the
   exports.
4. **This PC is already oversubscribed in principle.** The whole reason we are
   moving Kilosort to RIS is that Trodes and Kilosort cannot share this machine
   (Kilosort hangs *silently* with Trodes open). Adding a third demanding job to
   the same box argues against itself.

In fairness: a single compressed 640×480 stream is only ~0.1 MB/s written, so it
*could* share the machine. The case for separating is risk, not throughput — the
neural data is irreplaceable and the camera is cheap to isolate.

**Spec for the camera PC** — modest:

| | |
|---|---|
| CPU | any modern 6+ core (it only has to encode) |
| RAM | 16 GB |
| disk | SSD, ~100 GB free per recording day |
| interface | **USB3 on a dedicated controller card** (USB3 FLIR), or a **GigE NIC with jumbo frames** (GigE FLIR) |
| OS | Windows 10/11 or Linux — Spinnaker supports both |
| nice to have | any NVIDIA GPU, so NVENC does the encoding and the CPU does nothing |

## 2. Sync — into Trodes, not into Bpod

**Do not sync the camera to Bpod.** Sync it to Trodes, and Bpod alignment comes
for free. The reasoning is from our own measurements:

- Bpod's state word is **already recorded in Trodes** at 30 kHz on Din1–Din6
  (`docs/bpod_mcu_sync_wiring.md`).
- So if camera frames are *also* in Trodes, then camera ↔ Bpod alignment is
  automatic, through Trodes as the common clock. Nothing more is needed.
- Syncing to Bpod directly would be **worse**: we measured the Bpod clock
  drifting **2000–3000 ppm** against Trodes. Hanging the camera off Bpod would
  inherit that drift. Bpod also cannot timestamp events at the rate a camera
  produces frames.

### The design

```
  FLIR camera ──(USB3/GigE: pixels)──> camera PC ──> video file + frame-metadata CSV
       │
       └──(GPIO: "exposure active", one pulse per frame)──> Trodes Controller Din7
                                                             │
  Bpod ──(6-bit state word)──────────────────────> Trodes Din1..Din6
                                                             │
                                               all on the same 30 kHz clock
```

Every frame gets a pulse, and every pulse is timestamped on the **same clock as
the neural data**. That gives frame-exact alignment with no drift, and dropped
frames show up as missing pulses rather than as a silent shift.

**Din7 is genuinely free.** The `.trodesconf` declares all eight DIO channels as
inputs (`Controller_Din1`…`Din8`, bits 0–7), and we measured Din7 constant LOW
and Din8 constant HIGH with **zero transitions over 300 s** — nothing is wired to
either.

### A correction to an earlier note of mine

`docs/bpod_mcu_sync_wiring.md` says "`trodesexport` writes only `Din1..Din6` — a
signal moved onto `Din7` or `Din8` will not be exported." That conclusion was
drawn from exported sessions that contained six `dio_*.dat` files. The likely
real explanation is different and more benign: **Trodes DIO export writes state-
change events, and Din7/Din8 had zero transitions**, so there were no events to
write. A channel carrying an actual pulse train should export normally as
`dio_Controller_Din7.dat`.

That is a hypothesis, not a measurement — **verify it on the first test
recording.** And it does not gate anything either way, because
`scripts/check_bpod_dio_in_rec.py` already decodes **all eight bits** straight out
of the `.rec` (the DIO byte is byte 1 of every packet), so the strobe is readable
even if export does ignore it. **No change to `sync_pipeline/` is needed for any
of this.**

### Prerequisite: which physical pin is Din7?

This is the one real blocker. We established the channel → bit → function mapping
with certainty, but **not which physical pin of the 20-pin connector carries
which `Din`** — nothing in the recording carries that. The pin-walk procedure in
`docs/bpod_mcu_sync_wiring.md` measures it in one short recording, and it is
worth doing while that connector is already open.

### Electrical warning

FLIR GPIO is **not** guaranteed to drive a Trodes input directly:

- FLIR's opto-isolated output line typically **requires an external pull-up
  resistor** to produce a logic swing at all. This catches people constantly —
  the line looks dead without it.
- FLIR GPIO is **3.3 V** logic on most models; the Trodes input threshold is
  undocumented for us ("electrical details … not observable here").

So check the camera's datasheet for its GPIO output type, and confirm the swing
on a scope before trusting the pulses. Use the camera's **non-isolated 3.3 V
output line** if it has one, since that drives directly.

## 3. Triggering: let it free-run, start it second

Keep this simple. Do **not** try to make Bpod start and stop the camera per
trial — a missed trigger loses a trial, and per-trial files are harder to work
with than one continuous file.

Instead:

1. Start the **Trodes recording** first.
2. Then start the **camera**, free-running at a fixed frame rate, for the whole
   session.
3. Stop the camera, then Trodes.

That ordering buys a clean, checkable invariant: **the first strobe pulse Trodes
sees is video frame 1**, and the pulse count should equal the frame count. Trim
and clip per trial afterwards using the Bpod state word that is already in the
same file.

**Also save per-frame metadata.** Spinnaker can attach chunk data to each frame
(`FrameID`, camera `Timestamp`). Write it to a CSV beside the video. Then camera-
side frame drops are detectable independently of the Trodes pulse count, and any
disagreement between the two is a diagnosis rather than a mystery.

## 4. How big is a session?

Measured session length in this project averages **4.13 h** (range 3.0–4.8).
Mono 8-bit:

| resolution | fps | raw | **H.264** | 30 sessions (H.264) |
|---|---|---|---|---|
| 640×480 | 30 | 137 GB | **1.4 GB** | 41 GB |
| 640×480 | 60 | 274 GB | **2.7 GB** | 82 GB |
| 1280×1024 | 30 | 585 GB | **5.8 GB** | 175 GB |
| **1280×1024** | **60** | 1169 GB | **11.7 GB** | **351 GB** |
| 1440×1080 | 60 | 1387 GB | **13.9 GB** | 416 GB |

(H.264 estimated at 0.08 bits/pixel/frame, a reasonable quality for monochrome
behavioural video. Verify against a real 5-minute clip.)

**Raw is not an option** — 137 GB for the *smallest* configuration. Compress on
the fly.

**Recommended default: 1280×1024 at 60 fps mono, H.264 → ~12 GB/session.** `F:`
has 1.6 TB free and `Z:` has 42 TB, so 12 GB is nothing, and the extra resolution
matters because there is a **DeepLabCut environment on this PC** — pose tracking
wants pixels. Drop to 30 fps (~6 GB) if 60 is not needed for the behaviour being
scored.

Use a good-quality encode (H.264 CRF ≈ 17–20). Aggressive compression introduces
artifacts that degrade pose estimation, and the storage saved is irrelevant here.

## 5. Software on the camera PC

FLIR cameras use the **Spinnaker SDK**; `PySpin` is its Python binding.

- **First**, install Spinnaker and open the camera in **SpinView** (ships with
  it). Confirm the camera streams, find the working resolution/fps, and configure
  the GPIO line for "exposure active". This proves the hardware before any code.
- **Then**, for production recording, a **PySpin script piping frames to
  `ffmpeg`** is the most controllable option: no GUI dependency, scriptable
  start/stop, and the frame metadata CSV comes out of the same loop.
- Alternatives worth knowing: **Bonsai** (`bonsai-rx.org`, visual dataflow, has
  Spinnaker nodes, very common in rodent behaviour labs — good if a GUI is
  preferred) and **campy** (purpose-built for streaming machine-vision cameras to
  compressed video in behaviour rigs, with NVENC support).

Points that decide whether a 4-hour recording survives:

- **Buffer handling.** Set the stream buffer mode to oldest-first with enough
  buffers; the default can silently drop under momentary disk stalls.
- **The encoder must keep up** with the capture rate, or buffers back up and
  frames drop. Measure sustained throughput over at least 10 minutes before
  trusting a 4-hour run.
- **Check free disk before starting**, the same lesson as `trodesexport`: nothing
  warns you at hour three.

## 6. Who does what

| step | who |
|---|---|
| Design, sync architecture, file-size budget | done, this document |
| Identify which 20-pin pin is Din7 (pin walk) | **Gil** — hardware, while the connector is open |
| Wire camera GPIO → Din7, confirm the swing on a scope | **Gil** |
| Install Spinnaker, configure GPIO, write the recording script | **camera-PC agent** — see the prompt below |
| Verify strobe in the `.rec`, pulse count vs frame count, alignment to Bpod states | **this agent** — tooling already exists |

## 7. Prompt for the agent on the camera PC

> I need to set up continuous video recording from a FLIR machine-vision camera
> on this computer, to run alongside (but on separate hardware from) a
> Neuropixels/Trodes electrophysiology rig and a Bpod behavioural task on another
> machine. Please work through this and tell me what you find.
>
> **Goal:** record one continuous video per behavioural session — sessions run
> **3–5 hours** — with a hardware sync pulse sent out per frame, and per-frame
> metadata saved so dropped frames are detectable.
>
> **The sync design is already decided, please do not redesign it:** the camera's
> GPIO "exposure active" output will be wired to a digital input on the
> electrophysiology acquisition system, which samples it at 30 kHz. That gives
> frame-exact alignment to the neural data. The camera is therefore a free-running
> source — it does **not** need to be triggered by, or synced in software to, the
> behavioural system. Your job is the camera side only.
>
> **Please do:**
> 1. Identify the camera: exact FLIR model, interface (USB3 or GigE), sensor
>    resolution, max frame rate at full resolution in mono 8-bit, and **what GPIO
>    lines it has and their electrical type**. This last point matters: FLIR
>    opto-isolated output lines usually need an external pull-up resistor to
>    produce a usable logic swing, and most FLIR GPIO is 3.3 V. Report whether
>    there is a **non-isolated 3.3 V output line** available, since that drives a
>    logic input directly.
> 2. Install the **Spinnaker SDK** and its Python bindings (**PySpin**). Confirm
>    the camera streams in **SpinView** first, before writing any code.
> 3. Configure a GPIO line as **"exposure active"** (one pulse per frame
>    exposure) and report which line number and how it is configured, so the
>    wiring can be done against it.
> 4. Write a recording script (Python + PySpin, piping to `ffmpeg`) that:
>    - records **mono 8-bit at 1280×1024, 60 fps** as a target — report the
>      actual sustainable numbers if that is not achievable;
>    - encodes to **H.264 at CRF 17–20** (use NVENC if there is an NVIDIA GPU,
>      otherwise libx264), because the video will be used for **DeepLabCut pose
>      tracking** and compression artifacts hurt that;
>    - writes a **CSV of per-frame metadata** — frame ID and camera timestamp per
>      frame — beside the video, via Spinnaker chunk data;
>    - **checks free disk space before starting** and refuses to start if there
>      is not enough for the expected session length;
>    - sets the stream buffer handling explicitly (enough buffers, oldest-first)
>      rather than relying on defaults, which can drop frames silently under a
>      momentary disk stall;
>    - logs the total frames written at the end, which is the number that gets
>      cross-checked against the sync pulse count.
> 5. **Run a sustained test of at least 20 minutes** at the chosen settings and
>    report: frames captured, frames dropped (Spinnaker reports this), average
>    and peak write throughput, output file size, and CPU/GPU load. Extrapolate to
>    a 5-hour session. A configuration that works for 60 seconds and drops frames
>    at minute 30 is the main failure mode here, so please do not skip the long
>    test.
> 6. Report the **expected file size per 4-hour session** at the settings you
>    settle on.
>
> **Please do not:** attempt to connect to or configure the Trodes or Bpod
> machines, change anything about the behavioural task, or add software triggering
> between the camera and the behaviour system. The hardware strobe is the sync
> mechanism.
>
> **Report back:** the camera model and GPIO line to wire, its electrical type and
> whether it needs a pull-up, the final recording settings, measured dropped-frame
> rate over the long test, per-session file size, and the command to start a
> recording.

## 8. Still unknown

- ~~Which FLIR model this is.~~ **Identified 2026-11-04: `BFS-U3-13Y3M-C`** —
  Blackfly S USB3.1, ON Semi PYTHON 1300 1/2" CMOS, global shutter, **1280×1024**,
  **170 fps** max in Mono8, 4.8 µm pixels, 6 µs–30 s exposure, bus-powered (3 W),
  6-pin GPIO with 1 opto-isolated input, 1 opto-isolated output, 1 non-isolated
  bidirectional line, 1 non-isolated input, and a **3.3 V/120 mA auxiliary
  output**. The recommended 1280×1024 @ 60 fps is the sensor's native resolution
  at ~35% of its maximum frame rate, and 78.6 MB/s is comfortable on USB 3.1 — so
  the recommendation stands unchanged.

  **Mono, and FLIR mono models normally ship without an IR-cut filter**, so it
  should be NIR-sensitive (~20–30% QE near 850 nm) and **850 nm illumination is
  the right approach** — invisible to the rat, bright to the camera. Verify with
  the one-second test: point a TV remote at the lens in SpinView and look for the
  LED flashing.

  **GPIO mating connector is JST NSHR-06V-S** (camera side is BM06B-NSHSS-TBT) —
  check whether FLIR shipped a pre-wired GPIO cable, because without one there is
  no way to get the pulse out.

  **Correction to §2's electrical warning:** the opto-isolated *output* is the
  line that needs an external pull-up; the **non-isolated bidirectional** line is
  the easier one to drive a logic input with. Pin 6's 3.3 V/120 mA is what a
  pull-up would use. Line numbering differs between cased and board-level
  variants, so confirm against this model's own installation guide rather than a
  generic Blackfly S pinout.
- **Which physical pin is Din7**, as above. **No longer a blocker**: with a camera
  emitting a known 60 fps strobe, `scripts/find_camera_strobe_in_rec.py` reads all
  eight DIO bits and names whichever line the pulses landed on, so a wrong guess
  costs one 60 s recording instead of a datasheet hunt. See §9.
- **The Trodes digital input's logic threshold and whether it has a pull-up** —
  needed to know whether the camera's GPIO can drive it directly.
- **Whether `trodesexport` exports a Din7 that actually has transitions** — the
  hypothesis in §2, to be checked on the first test recording.

## 9. Settled camera-side parameters, and the strobe geometry to expect (2026-11-05)

The camera PC has these measured and locked. They are recorded here so verifying a
recording is a lookup rather than a re-derivation.

| | |
|---|---|
| resolution / format | 1280×1024, Mono8 |
| frame rate, nominal | 60 fps |
| frame rate, **measured** | **59.9944 fps** (−94 ppm; ordinary crystal tolerance) |
| exposure | **15.0 ms fixed**, `ExposureAuto` **off** |
| gain | **12.9 dB fixed**, `GainAuto` **off** |
| GPIO | **Line1**, source `ExposureActive` |
| sustained test | 1 h, **215,971 frames, zero dropped, zero FrameID gaps** |

**Why 15 ms and not 10.** I first asked for ≤10 ms to guarantee a wide gap between
pulses, then walked it back. The pulse-merging hazard is real but belongs to
*auto*-exposure, which was already off; at a fixed 15 ms the low gap is **50 samples at
30 kHz**, which is ample. Measured cost of going to 10 ms was **gain 12.9 → 16.4 dB**
(the 3.5 dB the one-third light loss predicts). Gain is sensor noise and this video feeds
**DeepLabCut**, so the cleaner image wins. Rule for later: buy brightness with IR
illumination first, exposure second, gain last.

Note the sustained hour-long test was run at 15 ms, so **the shipped configuration is the
validated one** — the exposure decision needed no re-test.

### What `find_camera_strobe_in_rec.py` should see

At 30 kHz, with `--fps 60 --exposure-ms 15`:

| quantity | expected |
|---|---|
| period | **500 samples** (500.05 at the measured 59.9944 fps) |
| pulse width | **450 samples** = 15.0 ms |
| low gap between pulses | **50 samples** = 1.667 ms |
| `frac high` | **≈ 0.90** |
| pulses in 60 s | **≈ 3600** |

Three failure signatures worth recognising:

- **width ≈ 50 samples (1.667 ms) instead of 450** — the line is **inverted**. Pulse
  *count* is unaffected, since there is still one rising edge per frame, so this is a
  polarity fix and not lost data.
- **width matching neither 450 nor 50** — probably not `ExposureActive`. `ExposureEnd`
  and `FrameTriggerWait` have different widths.
- **pulse count below frame count** — pulses were missed, not frames. The classic cause
  is exposure too close to the frame period, which is exactly what fixing exposure at
  15 ms with auto off is there to prevent.

Pulse count minus frame count of **0 or +1 is correct**: the last exposure can fire as the
recording script stops, before that frame reaches the container. Anything beyond that is
a finding.

### Note on the pin-ID test

Because the pin is unknown, the first test deliberately does not assume it. Wire a best
guess, record ~60 s with the camera running, and `find_camera_strobe_in_rec.py` names the
line the pulses arrived on. The **Trodes-first invariant does not apply to this test** —
it only matters for "first pulse = video frame 1" in real sessions — but the start order
should be recorded rather than assumed. The script warns if the strobe appears on
**Din1–Din6**, which would mean the camera is on a Bpod pin and that recording's
behavioural decode is corrupted.
