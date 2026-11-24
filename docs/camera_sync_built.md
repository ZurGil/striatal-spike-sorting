# Camera-to-neural sync, as built

**Status: complete and tested, 2026-11-23. First session with video: 2026-11-24.**

This is the as-built reference. `flir_camera_sync_design.md` is the *design* phase and
describes a route we abandoned (Din7 on the rear header, an external resistor divider,
an inverted pulse) — read this one for what is actually on the rig.

Everything below is measured on this rig unless it says otherwise. Where a prediction was
wrong, the wrong prediction is kept and marked, because the wrong ones were all plausible.

---

## 0. The whole thing in one paragraph

The camera free-runs at 60 Hz. Each time it opens its shutter it pulls a wire, and that wire
goes into a digital input on the Trodes MCU, which stamps the edge on the **same 30 kHz clock
that stamps the spikes**. So frame *i* of the video happened at a Trodes sample number we
read directly out of the recording. There is no clock fitting, no drift correction and no
camera timestamp in the analysis path — two wires and a count. The only thing that can go
wrong is losing track of *which* frame is frame *i*, and §5 is how that is checked.

---

## 1. The cable

### Parts

| part | detail |
|---|---|
| camera-side connector | FLIR's own cased 6-pin Hirose GPIO cable, shipped with the `-C` model |
| MCU-side plug | **Adam Tech MDP-007**, 7-position circular plug, male pins, solder cups — <https://www.digikey.com/en/products/detail/adam-tech/MDP-007/10222602> (DigiKey 2057-MDP-007-ND) |
| MCU-side socket | **Aux2** — the **right-hand** 7-pin mini-DIN on the MCU **front** panel, engraved **2** |
| external components | **none** |

The MDP-007 is **unshielded**: its metal shroud is *not* a ground. Ground is pin 3.

### What connects to what

Only two of the camera's six wires are used.

| camera wire | camera function | goes to | mini-DIN pin |
|---|---|---|---|
| **WHITE** | Line1 = OPTOOUT, opto-isolated, **open drain** | the input | **1** |
| **BLUE** | Opto GND, isolated return | ground | **3** |
| BROWN | camera *power* ground, **not** isolated | **nothing** | — |
| BLACK | Line0, opto input | nothing | — |
| GREEN | Line3 | nothing | — |
| RED | Line2 / 3.3 V | nothing | — |

```
  FLIR BFS-U3-13Y3M-C                        MCU front panel, Aux2
  6-pin Hirose GPIO cable                    7-pin mini-DIN (engraved "2")

        WHITE  (Line1, open drain) ───────────────▶  pin 1   input, 3.3 V idle
        BLUE   (Opto GND)          ───────────────▶  pin 3   ground
        BROWN  (camera PWR GND)    ──╳  leave cut back and insulated
        BLACK / GREEN / RED        ──╳  unused

                                            pin 4   +5 V  ── DO NOT CONNECT
```

### Where the wires actually went — Gil's record of the plug he built

Marked on the MDP-007 datasheet outline, **in the orientation he was looking at while
soldering**. The plug face has three holes across the top, two in the middle row either side
of the rectangular key, and two along the bottom.

```
                    key notch
                   ___     ___
                  /   \   /   \
             .---'     '-'     '---.
            /                       \
           |    (o)    (o)    (o)    |     top row    — all empty
           |                         |
          |  (o)   [   key   ]  (B)  |     middle row — RIGHT = BLUE, Opto GND, 0 V
           |                         |
           |      (o)       (W)      |     bottom row — RIGHT = WHITE, Line 1, ~3.3 V
            \                       /
             '---.             .---'
                  '-----------'

        (B) blue = opto ground      (W) white = Line 1, the signal
        (o) empty — one of these is +5 V
```

**This is a record, not an authority.** The two holes were chosen by *metering* — 0 V and
~3.3 V — and that is what made them correct; the drawing only remembers where they turned out
to be. The datasheet outline does not state whether it is drawn from the mating face or the
solder side, so the left–right sense of this picture is Gil's orientation at the bench and
nothing more.

**It is, however, validated.** The plug built this way produced 15,191 clean pulses at
59.9951 Hz with zero drops, so these positions are known-good for a second build. Meter them
anyway: it costs a minute, and the failure mode is a damaged MCU.

**BROWN must stay unconnected.** The whole point of using the opto-isolated output is that
the camera's ground and the rig's ground never meet; brown is the camera's non-isolated power
ground and connecting it bonds them, which is the ground loop we paid for the opto to avoid.

### Why there are no resistors

Line1 is an open-drain output and needs a pull-up — but the MCU already has one. Inside the
MCU, pin 1 runs through 200 Ω and then a **10 kΩ pull-up to 3.3 V** (confirmed by
SpikeGadgets with a board image, and by Daria that this revision's resistors "are placed the
same"). The line rests at 3.3 V and the camera's phototransistor pulls it to ~0.2 V while
exposing, sinking about 0.32 mA. Nothing else is needed.

### Which hole is pin 1 — and why the drawing can't tell you

**We never established the pin positions from the datasheet, and no drawing in this
repository asserts them.** Two reasons:

1. Mini-DIN footprints **mirror** between a PCB layout view and a front-panel view, and the
   7-pin field is left-right symmetric — so a mirrored solder still *looks* correct.
2. We didn't need to. Each contact is one metal post straight through the insulator, so **the
   solder cup behind a pin IS that pin.** Identify it electrically and the numbering, the
   datasheet and the mirror all stop mattering.

The procedure (Gil's, and it is the right one):

1. Push stiff wires into all seven solder cups, friction-fit, splayed apart so they can't touch.
2. Fit the **metal shroud** — see the warning below, the carrier does not make contact without it.
3. Plug it into Aux2, power on, and meter the **protruding wire ends** against the **Bpod
   header ground pin**.
4. Read them off:

| reading | that pin is | what to do |
|---|---|---|
| **~3.3 V** | **pin 1, the input** | camera WHITE here |
| **0 V** | **pin 3, ground** | camera BLUE here |
| **~5 V** | pin 4 | leave empty |

**Never probe the socket contacts directly.** They are recessed and closely spaced, and
slipping between the 5 V and ground contacts shorts the MCU supply.

### Four hazards, all of which have already bitten or nearly bitten

- **The shroud is required for contact, not for looks.** Measured 2026-11-20: without the
  shroud the carrier doesn't seat deep enough and *every* pin read open to ground — including
  the ground pin, which must read ~0 Ω powered or not. The probing was sound (front pin to its
  own cup read 1.3 Ω); the seating was the fault. Fit the shroud, then measure. **Backshell
  and strain relief go on last**, after soldering, so the cups stay reachable.
- **A bridge from pin 4 to pin 1 puts 5 V on a 3.3 V input.** The MCU manual says in bold
  that **>5 V damages the system**. After soldering, power off and continuity-check every pair
  of tails for solder bridges before plugging anything in.
- **The MCU chassis is not ground.** It is painted or anodized; Gil has tried repeatedly and
  it never works. The only ground reference for metering is the **Bpod header GND pin**.
- **Gil's meter (Mastech MS8217) shows open circuit as `0.L`**, which reads like a number but
  means **OL, over limit**. Always check the unit in the corner — a sweep that looked like
  "20+ volts" was 20+ **MΩ**.

### The divider board

A 2.2 k / 4.7 k resistor divider on perfboard was built for the earlier plan and **is not in
the signal path.** Do not reconnect it, in whole or in part:

- whole board in → the node sits at **3.41 V**, above SpikeGadgets' 3.3 V maximum
- red wire removed but the 4.7 k left to ground → the high level becomes
  3.3 × 4700/14900 = **1.04 V**, below the input threshold, and nothing registers at all

A partial teardown is worse than none. The board is correct, proven, and finished with. The
camera wires were **unsoldered** from it rather than cut, so it stays whole and reusable.

---

## 2. The MCU side

**The pulse arrives on Din8 — bit 7 of the DIO byte.** The Bpod state word uses bits 0–5
(Din1–Din6) and its codes run to 40, so the camera cannot collide with it.

### Din8 is recorded even though it is not exported

`trodesexport --dio` writes only the Din channels that appear in the **display
configuration**, not all the channels the hardware acquires. All eight digital inputs sit in
the `HardwareConfiguration` and go into **every packet** regardless, so the frame times are in
the `.rec` whatever the display config says.

Proven across 40 archived sessions: 39 list six `DispChannel` entries and export six Din
files; exactly one (Popper `20261110_114556`) lists eight and exports eight. Six listed, six
exported; eight listed, eight exported — nothing else distinguishes them.

Practical consequences:

- `scripts/camera_frame_times.py` reads the **raw packet bytes**, so it needs no export and
  works on a recording that is still open. Din8 being absent from the DIO folder costs nothing.
- **Still worth doing:** add Din7 and Din8 to the DIO display in Trodes and **save the
  workspace**, so the convenience `.dat` files appear too.
- Do **not** load the Nov 10 `temp.trodesconf` to get them. Its 384 `hwChan` assignments are
  byte-for-byte identical to current recordings, so the channel map isn't the hazard — but
  the channel map is the only field that was compared, and there's no reason to import
  unexamined fields into a working acquisition setup for two display entries.

### Packet layout — re-derive it, never assume it

The script measures the geometry from each recording's own embedded XML. For the current rig
it comes out as: DIO at byte **1**, timestamp at byte **84**, packet **856 bytes**
(= 84 + 4 + 2 × 384), **384 channels at 30 kHz**. The header length varies with the XML, so it
is never hardcoded.

Two self-checks must pass before any DIO byte is believed, because a wrong offset otherwise
gives a confident wrong answer:

1. the `0x55` sentinel on every packet, and
2. a timestamp that increments by exactly 1.

---

## 3. The camera side

Owned by the **"FLIR camera setup"** Claude session, on the camera PC.

| | |
|---|---|
| camera | Teledyne FLIR Blackfly S **BFS-U3-13Y3M-C**, s/n **19061068** |
| SDK | Spinnaker **4.4.0.246** |
| mode | **free-running**, `TriggerMode` **Off** — nothing triggers it, it just runs |
| format | 1280 × 1024, Mono8 |
| rate | 60 Hz nominal; **59.9951 Hz measured** (−82 ppm) |
| GPIO | `LineSelector` **Line1**, `LineMode` **Output**, `LineSource` **ExposureActive**, `LineInverter` **0** |
| chunk data | `ChunkModeActive` on, with **`ChunkFrameID`** and `ChunkTimestamp` enabled |
| exposure / gain | **pending Gil's choice** — see §8 |

### What it writes

`record.py` (PySpin → ffmpeg, libx264) writes two files per session into
`D:\camera_recording\sessions` :

- `<session>.mp4` — GOP pinned with `-g 60 -keyint_min 60 -sc_threshold 0`, so any trial can
  be seeked or clipped without re-encoding
- `<session>_frames.csv` — **one row per grab attempt**, not per frame written, with columns
  `grab_index, status, frame_id, timestamp_ticks`

About **1.8 GB per hour**; 1.98 TB free on that drive, so space is not a constraint.

**It refuses to start if either file already exists.** Until 2026-11-23 ffmpeg ran with `-y`
and the CSV opened in `"w"` mode, so relaunching with a reused session name silently
destroyed the earlier video *and* its frame log, with no warning and no trace. Found and
fixed by the camera session the day before first use.

### The session name is typed by hand — and why that no longer matters

`record.py --session <name>` takes the name as an argument. If it is omitted it falls back to
the camera PC's clock and prints a loud warning that it won't match Trodes. So the video and
the recording are only named consistently if whoever starts the camera reads the session
string off Trodes and retypes it correctly.

That is a step that gets done right for a month and then wrong once, and a video paired with
the wrong recording does not *look* wrong. So the pairing does not depend on the name — see
§5.3. **Type it correctly anyway** (matching names make everything downstream easier to read),
but a typo is now cosmetic rather than dangerous.

---

## 4. How the sync works

### Count, don't fit

Trodes stamps each pulse edge on the same 30 kHz clock as the spikes, so

> **pulse *i* ↔ frame *i***

and that is the entire model. No regression, no drift term, no camera clock in the analysis.

This is worth contrasting with the **Bpod** sync in the same recordings, which needs a global
affine fit (residual ~116 ms) plus local interpolation, because the Bpod free-runs on its own
oscillator and drifts 2300–2911 ppm — 34–43 s over a 4.13 h session. The camera also
free-runs and also drifts (−82 ppm, ~0.3 s per hour), but its drift is invisible to us
because we never use its clock: we use the pulse the camera *sent*, stamped by Trodes.

**The corollary is a rule:** frame times come from counting pulses, **never** from *i*/60.
At 60 Hz the two diverge by ~0.3 s per hour.

### Which edge is the exposure — measured, and my reasoning was backwards

**The HIGH phase is the exposure. The rising edge opens the shutter.**

I predicted the opposite, from circuit reasoning: Line1 is an open-drain opto output, so
asserting ExposureActive should pull the MCU input *low*, making the LOW phase the exposure.
The design doc says so, and it is wrong.

How it was settled: the camera reported `ExposureTime` = 10002 µs, and on the test recording
the two phases measured

| phase | measured | vs 10002 µs |
|---|---|---|
| **high** | **9.9584 ms** | **−43 µs** ✅ |
| low | 6.7095 ms | −3292 µs ❌ |

43 µs is about what an optocoupler's turn-on/turn-off asymmetry costs. The low phase is out
by 3.3 ms, which is nothing.

**The lesson, recorded in the script's own comments so it can't be re-derived wrongly:** do
not infer polarity from the circuit. Check it against the reported exposure time. That is
why `ExposureTime` is the thing to compare against, and why the default is now `--polarity high`.

### Idle level is not diagnostic

The MCU's pull-up holds the input **high** when the opto is idle — and high is *also* the
exposure-active level. So a stopped camera and an exposing camera read **identically** at any
single instant. Only the **absence of transitions** distinguishes them.

This killed an earlier version of the start/stop guard, which tested the line's level on the
first packet. The guard now measures the **quiet lead-in** before the first edge and the
**quiet tail** after the last, and requires at least two frame periods of silence at each end.

### What the train looks like when it is healthy

Measured on `F:\Gil\test.rec`, 253.2 s:

| | |
|---|---|
| complete pulses | **15,191** |
| frame interval | median 16.6667 ms, mean 16.6680 ms → **59.9951 Hz** |
| jitter | **7.0 µs** std (below one 33.3 µs sample period) |
| min / max interval | 16.6333 / 16.7000 ms |
| intervals > 1.5× median | **0** — no dropped frames |
| exposure (high) | 9.9584 ms, std 14 µs |
| high fraction | 0.5975 |

---

## 5. Verification — the part that took three attempts

The mapping is only as good as the claim "pulse *i* is frame *i*". Three versions of the
check, two of them wrong, because the wrong ones look right:

### 5.1 Wrong: pulses == frames written

The pulse is produced by the **exposure**, not by the write. A frame that is grabbed and then
lost still exposed, and still put a pulse on the wire. So a session with dropped frames fails
this check while being perfectly fine, and — worse — a session can pass it while being
misaligned.

### 5.2 Better but still wrong: `M = N + D`, with `grab_index` giving the position

`grab_index` counts only what the **host was handed.** A frame lost in transport consumes no
`grab_index` at all, so it is reported as no drop whatsoever, while still having exposed and
still having pulsed.

### 5.3 Correct: key on ChunkFrameID

`ChunkFrameID` is assigned **by the camera at exposure**, so it indexes exactly the events the
pulses index. The whole verification is one comparison:

```
pulses  ==  f_last - f_first + 1
```

If that holds, no exposure happened outside the observed span, so

```
pulse i  ↔  ChunkFrameID  f_first + i - 1
```

**exactly** — regardless of how many frames were dropped or where they fell. No counting
argument, no position argument, no inference.

The camera session agreed to this form and reports `f_first` and `f_last` in every session
summary.

### The same number also proves the pairing

An hour at 60 Hz is ~216,000 exposures. Two different sessions agree on that count only if
they were started and stopped on the very same frame. So the ChunkFrameID span is a
**fingerprint**: `--frames-csv` accepts a *directory*, reads the span of every
`*_frames.csv` in it, and selects the one matching this recording's pulse count — whatever
the file is called. This is what makes the hand-typed session name harmless.

Tested against `test.rec` (15,191 pulses) with three candidates in a folder: the true match
at span 15,191 was selected; a near-miss at 15,190 (one frame lost at the start) and a decoy
at 5,000 were both rejected. With the correct file removed it refuses and prints the span
table so the mismatch is visible.

### The dangerous case

**A frame dropped at the very start shifts every frame by one** — 16.7 ms — and it passes
every naive quality check, is invisible on the neural side, and leaves no trace in the video.
It is the specific failure §5.3 exists to catch.

### The tool refuses rather than guesses

`scripts/camera_frame_times.py` writes nothing when the train is not clean. It refuses on: a
failed sentinel or timestamp check, a lead-in or tail shorter than two frame periods, a
ChunkFrameID span that disagrees with the pulse count, or logged grabs whose `frame_id` falls
outside the train. A plausible-looking table of wrong timestamps is the worst possible
output, so it is not an output it can produce.

---

## 6. Run order for a session

The order is not cosmetic — it is what makes the count valid, and it replaces a separate
calibration recording.

### Setup, once

| | |
|---|---|
| **S1** | Camera plugged into Aux2 (right-hand front mini-DIN, engraved 2) |
| **S2** | Din7 + Din8 added to the Trodes DIO display, **workspace saved** |
| **S3** | Trodes **closed** before any Kilosort run — it holds the GPU and Kilosort hangs silently |
| **S4** | IR illumination on, focus set **under the IR light** (NIR focus shift is real) |
| **S5** | Exposure and gain set to the agreed values, both Auto **OFF** |

### Every session

1. Start the Trodes acquisition program (needs the Neuropixels probe connected — Gil connects
   it last, as it is the delicate step; no DIO appears on screen until this runs).
2. Confirm Din8 is **flat, no transitions**, with the camera not yet started.
3. **Hit record in Trodes.** Note the session name it uses.
4. **Now** launch `record.py --session <that exact session name>`. It starts acquiring the
   moment it launches; nothing waits on Trodes, so this ordering is in your hands.
5. Run the behavioural session.
6. **Stop the camera first** (Ctrl+C, or let `--duration` elapse).
7. **Then** stop the Trodes recording.
8. Copy `<session>.mp4` and `<session>_frames.csv` together — never one without the other.
9. Run the check (§7).

**Trodes record ON → camera ON … camera OFF → Trodes record OFF.**

Why it matters: done this way, every real recording contains its own idle measurement in the
quiet lead-in and tail, and a clean run **proves** the camera started after and stopped before
the neural recording — which is precisely the condition that makes pulse *i* = frame *i* true.
Done the other way, the first pulse in the file is not the first frame of the video, and the
count is unrecoverable without ChunkFrameID.

---

## 7. The tool

```
C:/Users/Adam/anaconda3/envs/phy2_ky/python.exe -I scripts/camera_frame_times.py \
    <path to the .rec> \
    --frames-csv D:\camera_recording\sessions
```

`phy2_ky` is the only environment with pyarrow. Reads raw packets, so it runs on an open
recording and needs no export. Useful flags: `--line 8`, `--polarity high` (the default),
`--frames N`, `--out <dir>`, `--no-figure`.

Output, one row per frame:

| column | |
|---|---|
| `pulse_index` | 1-based, in the order the pulses arrived |
| `trodes_sample_exposure_start` | **the number to use** — same clock as the spikes |
| `trodes_sample_exposure_end` | |
| `t_exposure_start_s`, `t_exposure_end_s`, `t_exposure_mid_s` | seconds from the recording start |
| `frame_id`, `grab_index`, `status` | present when a CSV was joined |

Plus a three-panel figure: 150 ms of the raw trace, the frame-interval histogram, and the
pulse-width histogram.

---

## 8. Open

- **Exposure and gain.** Current `ExposureTime` 10002 µs / gain 6.98 dB; the documented and
  1-hour-validated configuration is 15000 µs / 12.9 dB. Numbers can't settle this — the
  camera session has sent Gil a side-by-side image to choose from. **Constraint from the
  neural side: anything up to 15 ms is safe** (15 ms still leaves a 1.67 ms gap = 50 samples
  at 30 kHz). Past ~15.5 ms edge detection degrades; past 16.67 ms the camera silently drops
  its frame rate. So image quality decides it, not sync.
- **Din7/Din8 in the Trodes DIO display**, then save the workspace (§2).
- **No network route from the camera PC to the archive.** Camera PC is `10.22.68.87/20`; the
  share is `172.20.22.203`; this PC is `172.20.22.115/24` plus `192.168.0.2/24` and **cannot
  bridge**. The camera PC has a free second NIC (Intel I219-LM), so the fix is a cable and a
  `172.20.22.x` address. Parked by Gil — flash drive for now.
- **Per-trial clips.** Better plan: run DeepLabCut **once** over the continuous video and
  slice the **pose table** per trial. Physical clips only on demand. A cut-list splitter was
  offered; not built.

---

## 9. Dead ends — recorded so they are not resurrected

| idea | why it is dead |
|---|---|
| **Din7 on the rear header** | Din1–6 are the only live digital inputs. Walked one identical signal along the top row: Din4/5/6 saw the pulses, **Din7 read flat 0 and Din8 flat 1**. Only SpikeGadgets can enable 7–8. Aux2 made the request unnecessary. |
| `Din7` appearing on the Trodes screen | Proves **nothing**. The workspace declares `Controller_Din1..Din8` with `input="1"` regardless of what the controller physically has. This cost most of a day on 2026-11-06. |
| External resistor divider | The MCU already has the pull-up. See §1. **Never reconnect it, not even partly.** |
| Expecting an inverted pulse | Predicted ~1.667 ms high / 15 ms low. **Wrong** — high is the exposure. §4. |
| Testing the line's idle *level* | The pull-up makes idle and exposing read the same. Only the absence of transitions distinguishes them. §4. |
| Bonsai + Bpod | Bpod drifts 2300–2911 ppm (34–43 s/session) and Bonsai adds a *third* unmodelled clock (the PC). The hardware pulse removes the camera clock from the analysis entirely. |
| A 30 s calibration recording with the camera idle | Made redundant by the run order: every real session contains its own idle lead-in and tail. §6. |

---

## 10. Where things live

| | |
|---|---|
| tool | `scripts/camera_frame_times.py` |
| this document | `docs/camera_sync_built.md` |
| design phase (superseded) | `docs/flir_camera_sync_design.md` |
| log entry on the export gate | `post_ks_correction/RESEARCH_LOG.md` §5bf |
| video + frame CSV | `D:\camera_recording\sessions\` on the camera PC |
| recordings | `F:\Gil\<rat>\` during work, archived to `Z:\Gil\<rat>\` |
| camera PC owner | the **"FLIR camera setup"** Claude session |

Commits: `3b5a1b0` (export gate), `233d765` (the tool), `5763e6c` (ChunkFrameID keying),
`8dec017` (fingerprint pairing).
