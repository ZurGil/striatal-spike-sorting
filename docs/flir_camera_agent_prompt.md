# Prompt for the camera-PC agent

Everything below the line is written to be handed to an agent running on the
camera computer. Design rationale for these choices is in
`docs/flir_camera_sync_design.md`; the prompt is self-contained so the agent does
not need that file.

---

## Task: set up synchronised video recording for a rodent electrophysiology rig

You are working on a **dedicated camera computer**. It sits beside — but is
deliberately separate from — a Neuropixels/Trodes electrophysiology rig and a
Bpod behavioural task, both on another machine. Do not try to reach those
machines.

Please walk me (the user) through this from the beginning. I am a neuroscientist,
not a software engineer: explain what you are doing and why, tell me which
physical things to plug where, and stop to check with me when something needs a
hardware change or a purchase.

### What we are building

One continuous video per behavioural session, with a hardware pulse sent out per
frame so the video can be aligned to neural data afterwards.

- Sessions run **3–5 hours**. A typical one is 4.1 h.
- Target settings: **1280×1024, mono 8-bit, 60 fps, H.264** → expect **~12 GB per
  session**. Raw would be ~1.2 TB per session, so compression is not optional.
- There will be roughly **one session per day**, ongoing.

### The camera

**Teledyne FLIR Blackfly S, model `BFS-U3-13Y3M-C`.** Confirmed specs:

| | |
|---|---|
| sensor | ON Semi PYTHON 1300 (P1300), 1/2" CMOS, **global shutter** |
| resolution | 1280 × 1024 (1.3 MP), 4.8 µm pixels |
| max frame rate | **170 fps** at full resolution in Mono8 |
| interface | **USB 3.1**, 5 V bus-powered, 3 W max |
| exposure range | 6 µs – 30 s |
| I/O | 6-pin GPIO: 1 opto-isolated input, 1 opto-isolated output, 1 non-isolated bidirectional line, 1 non-isolated input, plus a **3.3 V / 120 mA auxiliary output on pin 3** (pin 6 = GND) |

Our 60 fps target is well inside the camera's 170 fps capability, and
1280×1024 mono8 at 60 fps is 78.6 MB/s, comfortable for USB 3.1.

### The sync design — already decided, please do not redesign it

The camera's GPIO will emit **one pulse per frame exposure** (`ExposureActive`).
That pulse is wired into a digital input on the electrophysiology acquisition
system, which samples it at **30 kHz**. Every frame therefore gets a timestamp on
the same clock as the neural data, giving frame-exact alignment with no drift and
making dropped frames visible as missing pulses.

Consequences for your work:

- The camera is a **free-running source**. It does **not** need to be triggered
  by, or software-synced to, the behavioural system. Do not add software
  triggering or network time sync.
- Your job is the **camera side only**: get frames to disk reliably, and get the
  `ExposureActive` pulse out of a GPIO pin.
- **Recording order at session time** will be: start the ephys recording first,
  then start the camera. That makes "first pulse = video frame 1" a checkable
  invariant. Your script should therefore be startable on command and run until
  stopped.

### Step 1 — Physical connection, and what we may need to buy

Please check and tell me what is missing:

1. **USB 3** — the camera needs a USB 3 port, ideally on a **dedicated host
   controller**. A shared controller with other devices is the usual cause of
   dropped frames. Check what controllers this machine has and whether an add-in
   USB3 card would be wise. The camera ships with a locking USB3 cable; confirm we
   have it and that it is USB3 (not a USB2 cable that fits).
2. **A C-mount lens.** The camera body has no lens. For a behaviour box, tell me
   what focal length suits the working distance — I will measure the distance from
   where the camera will sit to the floor of the box and tell you. The lens must
   **transmit near-infrared** (see step 2).
3. **The GPIO mating connector.** This is the part people forget. The camera's
   GPIO is a **JST BM06B-NSHSS-TBT**; the mating connector is a
   **JST NSHR-06V-S** with crimp contacts. FLIR sometimes ships a pre-wired GPIO
   cable — check the box first. If not, we need that connector or a pre-made FLIR
   GPIO cable, or there is no way to get the sync pulse out.
4. **Disk space** — ~12 GB per session, one session a day. Check free space and
   tell me how many days of recording fit.

### Step 2 — Infrared illumination (please advise me on this)

The camera is **monochrome**, and FLIR mono models are normally supplied
**without an IR-cut filter**, so it should be sensitive into the near infrared
(roughly 20–30% quantum efficiency around 850 nm). This matters because rodents
cannot see 850 nm, so we can illuminate the box brightly for the camera while it
stays dark for the animal — the standard approach for behaviour video.

**Please verify this empirically rather than trusting the datasheet**, with the
simplest possible test: open the camera in SpinView, point a **TV remote
control** at the lens and press a button. If you see the remote's LED flash in
the image, the camera sees IR and there is no IR-cut filter.

Then advise me on:

- **850 nm vs 940 nm IR LEDs.** My understanding is 850 nm is the normal choice
  because sensor sensitivity at 940 nm is much lower, at the cost of a faint red
  glow visible if you look straight at the LED. Confirm or correct that.
- **How much light we need.** At 60 fps the exposure can be at most ~16 ms, and
  less in practice. Work out roughly what illumination is needed for a decent
  image at low gain, and tell me what to buy — IR LED arrays or illuminator
  panels, how many, and where to place them for even lighting without hot spots
  or specular glare off the box walls.
- **Two cautions I have been warned about:** a lens focuses *differently* in IR
  than in visible light, so we must **focus under the actual IR illumination we
  will use**, not under room light. And not every C-mount lens transmits NIR
  well. Flag anything relevant when recommending a lens.

### Step 3 — Software

1. Install the **Spinnaker SDK** (Teledyne FLIR) and its Python bindings,
   **PySpin**. Note that PySpin's version must match the installed Spinnaker
   version — they are not independent.
2. **Open the camera in SpinView first** and confirm live streaming before
   writing any code. Report the camera's serial number, firmware version, and
   whether a firmware update is available.
3. Find and report the achievable settings: confirm 1280×1024 Mono8 at 60 fps
   streams without dropped frames, and note the maximum stable frame rate.

### Step 4 — Configure the sync pulse

Configure a GPIO line as an **output** with its source set to **`ExposureActive`**
(one pulse per exposure).

Important, and please verify against the installation guide for this exact model
rather than assuming — Blackfly S line assignments vary between cased and
board-level variants:

- The **opto-isolated output** is electrically an open-drain and **requires an
  external pull-up resistor** to produce any usable voltage swing. Many people
  conclude the line is dead when it is simply missing its pull-up.
- The **non-isolated bidirectional line** is the more convenient choice for
  driving a logic input directly, and may also want a pull-up. **Pin 3 of the
  GPIO connector provides 3.3 V at up to 120 mA** (dual-function supply/GPIO);
  **pin 6 is GND**. Note that pulling up from the camera's own rail defeats the
  point of using the opto-isolated output — see the design doc.

Please report back:

- **Which line number** you configured, and **which physical pin and wire colour**
  it is on;
- whether it is **isolated or non-isolated**, and whether it **needs a pull-up**
  and of what value;
- the **measured voltage swing** — I have a scope, so tell me what to look for and
  I will confirm the pulse is really there, its height, and its width.

This matters because the receiving input's logic threshold is not documented for
us, so we must know the camera's actual output levels rather than assume they are
compatible.

### Step 5 — The recording script

Write a script (Python + PySpin, piping frames to `ffmpeg`) that:

- records **Mono8 1280×1024 at 60 fps**, encoding to **H.264 at CRF 17–20** — use
  **NVENC** if this machine has an NVIDIA GPU, otherwise `libx264`. Quality
  matters: this video will be used for **DeepLabCut pose tracking**, and
  compression artifacts degrade pose estimation. Do not optimise for file size.
- writes a **CSV of per-frame metadata** beside the video — at minimum frame ID
  and camera timestamp per frame, via Spinnaker **chunk data**. This lets
  camera-side frame drops be detected independently of the sync pulses.
- **sets stream buffer handling explicitly** — enough buffers, `OldestFirst` — and
  does not rely on defaults, which can silently drop frames during a momentary
  disk stall.
- **checks free disk space before starting** and refuses to start if there is not
  enough for the expected session length. Nothing warns you at hour three.
- **logs the total number of frames written** at the end. That number gets
  cross-checked against the sync pulse count on the ephys side, so it must be
  accurate and easy to find.
- names files by date/time so a session's video, CSV and log stay together.
- can be **started and stopped cleanly** (a keypress or Ctrl-C) and still finalise
  a playable video file — a 4-hour recording that is unplayable because the
  container was never closed is a lost session.

### Step 6 — The test that actually matters

**Run a sustained recording of at least 20 minutes** at the final settings, and
preferably an hour. Then report:

- frames captured and **frames dropped** (Spinnaker reports this — please find and
  report the incomplete/dropped frame counters explicitly);
- average and peak **write throughput**, and whether the encoder kept up;
- **CPU and GPU load**, and whether there is headroom;
- output **file size**, extrapolated to a 5-hour session;
- whether the video **plays back correctly** end to end, and whether the frame
  count in the file matches the frame count logged.

Please do not skip this or shorten it. The failure mode that matters here is a
configuration that works perfectly for sixty seconds and begins dropping frames
at minute thirty — a 4-hour recording is a long time for a buffer problem to
develop, and a session cannot be repeated.

### Step 7 — Report back

Tell me, in plain language:

1. What hardware is still missing or needs buying (lens, GPIO connector, IR
   illumination, USB3 card).
2. Whether the IR test passed, and your recommendation for IR illumination.
3. The GPIO line, physical pin, wire colour, isolated or not, pull-up needed,
   and the expected voltage swing.
4. Final recording settings and the measured dropped-frame rate over the long
   test.
5. Expected file size per session.
6. The exact command I run to start a recording, and how I stop it.

### Please do not

- Connect to, or change anything on, the Trodes or Bpod machines.
- Modify the behavioural task.
- Add software triggering or network time synchronisation between the camera and
  the behaviour system — the hardware pulse is the sync mechanism.
- Record raw/uncompressed video to disk as a default.

### Coordinating with the other agent

There is another Claude agent working on the electrophysiology side, which owns
the sync verification: reading the frame pulses out of the ephys recording,
checking the pulse count against your frame count, and confirming alignment to
the behavioural events. If both sessions are connected to the same account via
**Remote Control**, the two of you can message each other directly.

When your long test is done, the useful handoff is: **frame count, recording
start and stop times, the GPIO line and pulse width you configured, and the path
to the frame-metadata CSV.** That is what the other side needs to verify the sync
end to end.
