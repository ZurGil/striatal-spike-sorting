# Bpod → Trodes MCU digital sync: what each line carries

Written for someone working on the physical connection. Everything in the
"measured" sections was read out of real recordings on 2026-10-29
(`20261029_110025`, `20261029_123356`) and cross-checked against the Bpod session
file `Shamir01_Dual2AFC_nat_Oct29_2026_Session2.mat`. The section on what is *not*
known is as important as the rest — do not fill it in by guessing.

Regenerate any of this with:

```
python scripts/check_bpod_dio_in_rec.py <session-id> --seconds 150 --from-end \
    [--bpod <SessionData.mat>]
python scripts/compare_dio_word_to_bpod.py <session-id> <SessionData.mat> \
    --start-s 5 --seconds 230
```

Both read only, work on a recording that is still running, and need no export.

## The eight digital input channels — MEASURED

The controller presents 8 digital inputs. The `.rec` XML header declares them as
one 1-byte device:

```xml
<Device available="1" name="Controller_DIO" numBytes="1" packetOrderPreference="10">
  <Channel bit="0" dataType="digital" id="Controller_Din1" input="1" startByte="0"/>
  ... through ...
  <Channel bit="7" dataType="digital" id="Controller_Din8" input="1" startByte="0"/>
</Device>
```

| channel | bit | value | what it carries | state in every window checked |
|---|---|---|---|---|
| `Controller_Din1` | 0 | 1 | state word bit 0 | **active** — 40 transitions / 150 s |
| `Controller_Din2` | 1 | 2 | state word bit 1 | **active** — 30 transitions / 150 s |
| `Controller_Din3` | 2 | 4 | state word bit 2 | **active** — 10 transitions / 150 s |
| `Controller_Din4` | 3 | 8 | state word bit 3 | **active** — 22 transitions / 150 s |
| `Controller_Din5` | 4 | 16 | state word bit 4 | **active** — 2 transitions / 150 s |
| `Controller_Din6` | 5 | 32 | state word bit 5 | **active** — 16 transitions / 150 s |
| `Controller_Din7` | 6 | 64 | nothing | constant **LOW**, 0 transitions, only value 0 |
| `Controller_Din8` | 7 | 128 | nothing | constant **HIGH**, 0 transitions, only value 1 |

Transition counts vary with how many trials and which trial types occur in a
window — `Din5` (value 16) moves rarely because only the `rewarded_Lin_start` /
`rewarded_Rin_start` family of states sets it. A low count is not a fault; **zero
over a window containing several trials is.**

## The encoding — MEASURED

- **Six lines, one parallel word, active high.** `Din{j}` is bit `(j-1)`, so the
  value is `Din1·1 + Din2·2 + Din3·4 + Din4·8 + Din5·16 + Din6·32`.
- **The word is the Bpod state number.** For `Dual2AFC_nat` these run 1–40;
  `OriginalStateNamesByNumber` in the Bpod file is the authoritative name list.
- **Code 0 is not a state** — it is all six lines low, which is what sits between
  trials.
- **Code 1 = `PreITI_start` = the first state of every trial.** This is the marker
  `sync.py` aligns the two clocks on, so bit 0 (`Din1`) is the single most
  load-bearing line: if it fails high, code 1 and code 0 become the same value and
  there are no trial starts at all.
- **Code 40 = `ITI` = the last state of every trial.**
- **6 bits span 0–63 and the protocol only uses 1–40**, so there is headroom; a
  seventh line is not needed for this task.

### There is no strobe, and no glitch at the sampling resolution

Measured over 238 word changes in 300 s: changes of 1, 2, 3, 4, 5 and even **6
bits at once** all land inside a single 33.3 µs sample. No intermediate value ever
persisted for 1 or 2 samples. The shortest dwell in the whole window was 3
samples, and those are real states, not skew (below). So the word can be read at
any sample without a clock or latch line.

### Whatever reads this word must sample at ≥ 30 kHz

The protocol has eleven **pass-through states that last 0.10 ms** —
`broke_fixation`, `early_withdrawal`, `stimulus_delivery`, `start_Lin`,
`start_Lin2`, `start_Rin`, `start_Rin2`, `timeOut_IncorrectChoice`,
`timeOut_SkippedFeedback`, `skipped_feedback`, `missed_choice`. At 30 kHz that is
exactly 3 samples, confirmed in both a September and an October session. Anything
sampling slower will silently drop them, and `start_Lin`/`start_Rin` are what
distinguish a left from a right choice.

## Where the word ends up in the data — MEASURED

- **In the `.rec`, it is byte 1 of every 856-byte packet.** A packet is: `0x55`
  sentinel (byte 0), then each available `<Device>`'s bytes in
  `packetOrderPreference` order (`Controller_DIO` 1 B at byte 1, `Multiplexed`
  8 B, `neuropixels_LFP` 66 B, `SysClock` 8 B), then a uint32 timestamp at byte
  84, then 384 int16 samples from byte 88.
- **`trodesexport` was observed writing only `Din1..Din6`** — the `.DIO` folder
  of an exported session contains six `dio_Controller_Din{1..6}.dat` files and
  nothing for `Din7`/`Din8`. **Revised 2026-11-04:** the original reading of
  that ("a signal on Din7/Din8 would never be exported") is probably wrong. The
  `.trodesconf` declares all eight channels as inputs, and Trodes DIO export
  writes *state-change events* — Din7 and Din8 had **zero transitions**, so there
  was nothing to write. A channel carrying real pulses should export normally.
  That is a hypothesis; verify it on a recording where Din7 actually moves.
  Either way it gates nothing: `scripts/check_bpod_dio_in_rec.py` decodes all
  **eight** bits straight from the `.rec`, so an extra line is readable without
  touching `sync_pipeline/`. See `docs/flir_camera_sync_design.md`.
- `sync_pipeline/dio_decoder.py` reads those six files and rebuilds the word by
  zero-order hold, treating "before a channel's first sample" as low.

## Timing agreement with Bpod — MEASURED

Pairing trials on the wire against Bpod's own `OriginalStateData`, trials 519–524
of `Session2` matched **state-for-state**, including a 30-state and a 20-state
trial, with every state onset sitting **0.13–0.30 ms** from Bpod's own timestamp
for it.

The Bpod clock drifts against the Trodes clock by roughly **2000–3000 ppm**
(2300 ppm on 20260916, 2167 and 2911 ppm on 20261029). That is normal for this rig
and is what `sync.py` fits; it is not a wiring symptom.

## What is NOT known — do not guess these

- **Which physical pin of the 20-pin connector is which `Din` channel.** Nothing
  in the recording carries it. The `.rec` header names channels and bit positions
  only. Any pin number in a document that cites this file is an invention unless
  it came from the controller's own pinout or from the walk below.
- **Which Bpod output drives which `Din`.** The data shows that six lines are
  being driven and what each bit means to the decoder; it cannot say which BNC,
  port or header pin on the Bpod side feeds a given bit.
- **Electrical details**: logic levels, pull-up or pull-down behaviour, source
  impedance, termination. Not observable here.
- **The idle level of an unconnected input.** `Din7` sits constantly LOW and
  `Din8` constantly HIGH, both with nothing wired to them, so there is no single
  idle level to infer from. An earlier note of mine claimed stuck-high implies
  disconnected, citing `Din8`; `Din7` contradicts that and the claim is withdrawn.

## How to establish the physical pin map with certainty

A pin walk, which turns the unknowns above into measurements:

1. Start a Trodes recording (any scratch session; 30–60 s is plenty, ~1 GB).
2. Drive **one** line at a time from the Bpod side — toggle it a few times, pause
   a few seconds, move to the next, and keep a note of the order.
3. Run `check_bpod_dio_in_rec.py <sid> --seconds 60 --from-end`. It reports
   transitions per channel, so the line that moved identifies itself, and the
   pauses separate the steps.

That produces a `physical pin → Din channel → state-word bit` table that is
measured rather than assumed. Until then, only the middle column of that table is
established.

## The failure seen on 2026-10-29, as a reference symptom

`Din1` chattered for about 15 s around Bpod trial 364 and then latched **HIGH**
permanently at 3453.9193 s into `20261029_110025` (11:57:58.919), while the other
five lines kept working. The symptom set, worth recognising again:

- the affected line shows **0 transitions** over a window with several trials;
- every decoded value is **odd** (`state | 1`);
- **41 appears**, and there is no state 41;
- `0` and `1` both read as `1`, so **trial starts vanish**;
- the 0.10 ms states that differ from their neighbour only in bit 0 disappear.

None of that is recoverable in software — a jammed bit destroys the information.
The chatter before the latch is the one diagnostic hint: a logic output that dies
usually dies at once, whereas a contact that is losing grip chatters first.

## The Digital I/O pinout, from the SpikeGadgets MCU manual (added 2026-11-05)

Gil found the MCU manual (SpikeGadgets, Rev 2d-0723, p.13). It answers the pin question
that the pin-walk procedure above was written to measure, and it adds a prerequisite
nobody knew about.

### The connector

Figure 2 of the manual gives the Digital I/O header as **10 columns × 2 rows**. Column
labels, left to right:

```
   5V    8    7    6    5    4    3    2    1   GND
   []   []   []   []   []   []   []   []   []   []     <-- INPUTS   (top row)
   []   []   []   []   []   []   []   []   []   []     <-- OUTPUTS  (bottom row)
```

So **`Din7` is the top row, third column from the left** — immediately right of the `8`
column, immediately left of the `6` column. The numbering runs **descending** left to
right, which is the opposite of what most people assume, and the power/ground pins sit at
the two ends.

Two miswires to avoid, in order of seriousness:

- **One column to the RIGHT of Din7 is Din6 — a live Bpod state-word line.** Landing the
  camera there would corrupt the behavioural decode of every recording made that way.
  The safe direction to err is **left**, onto Din8, which carries nothing.
- **The bottom row is OUTPUTS.** Column 3 of the bottom row is *Dout7*, not Din7.
  Connecting the camera's output to the MCU's output means two drivers on one node.

`scripts/find_camera_strobe_in_rec.py` detects the first of these explicitly and warns,
because it is the one failure worse than nothing happening.

### Prerequisite: Din7 and Din8 are DISABLED by default

The manual, verbatim: *"By default, 6 digital inputs (Din 1 - 6) are enabled for
receiving signals from other hardware devices. **Digital inputs 7 & 8 as well as digital
outputs are turned off by default.**"*

**This is not something the workspace file can fix.** I checked both our
`20260901_084658_shamir.trodesconf` and Trodes' own
`Resources/Images/DeviceProfiles.xml`: both declare all eight channels with
`dataType="digital"` and nothing else, and **no enable/disable attribute exists anywhere
in the schema**. The DIO byte always carries 8 bits; bits 6 and 7 simply report whatever
a disabled input buffer produces. That is consistent with what we measured — `Din7`
constant 0 and `Din8` constant 1, with **zero** transitions over both a 300 s and a 99 s
recording. An *enabled* but unconnected input would be more likely to show at least
occasional noise.

So **Din7 must be enabled at the MCU before the strobe test**, through Trodes' hardware
settings rather than the workspace file. (I cannot determine the exact menu path from the
filesystem; it is an MCU-level setting.)

**Why this ordering matters:** if the test recording is made with Din7 still disabled, a
null result is **ambiguous** — "wrong pin" and "right pin, input disabled" look
identical. Enabling first makes the test interpretable.

This also retro-explains the open question in §2 above. The reason `trodesexport` never
produced a Din7 file is now over-determined: the channel had no transitions *and* the
input was off. It still does not tell us whether export works once there are real
transitions, so that hypothesis stays open.

### Electrical limits, from the same page

- Digital inputs are **3.3 V TTL** with a **5 V maximum**. The manual highlights:
  *"Inputs greater than 5V will damage the system."*
- **Pull up to 3.3 V, never to 5 V or higher.** The camera's GPIO **pin 3** supplies
  3.3 V / 120 mA (dual-function supply/GPIO pin); **pin 6 is GND**. (Corrected
  2026-11-06 — I had these swapped; the camera side verified it against FLIR's
  docs and a live PySpin query.) But see the isolation note below before using the
  camera's rail as a pull-up source at all.
- **Grounds must be tied.** The `GND` pin is the far-right column. The manual's own
  Ground Point note asks for a shared ground across experimental hardware, and an
  opto-isolated camera output specifically has no return path without it.
### The isolation trap (added 2026-11-06)

The camera side chose the **opto-isolated** output deliberately, to keep a ground loop
out of the ephys rig. That isolation **only survives if the pull-up is powered from the
MCU side.**

The opto output is a phototransistor: collector on **pin 2 (Line1)**, emitter on
**pin 1 (OPTO_GND)**. Both are galvanically isolated from the camera's own **pin 6 (GND)**
and **pin 3 (+3.3 V)**.

```
CORRECT -- isolation preserved
    MCU 5V --- 2.2k --+-- 4.7k --- MCU GND
                      |
                      +-- pin 2 (Line1)        pin 1 (OPTO_GND) --- MCU GND
                      |
                      +-- Din7                 pins 3 and 6: NOT CONNECTED

WRONG -- isolation silently defeated
    camera pin 3 (3.3V) --- Rpullup --- pin 2,  with pin 1 --- MCU GND
```

In the wrong version current leaves the camera's power domain, crosses the
phototransistor and lands in MCU ground, **bonding the two grounds through the very part
that was supposed to isolate them**. Nothing misbehaves visibly, so it would never be
noticed. My earlier advice to pull up from the camera's 3.3 V was right for the
*non-isolated* line and wrong for this one.

**Divider rather than a plain pull-up**, because a 5 V pull-up sits exactly at the
documented maximum and a nominal 5 V rail can read 5.25 V. With 2.2 k / 4.7 k the node
idles at 5 × 4.7/6.9 = **3.4 V** (inside 3.3 V TTL) and the opto pulls it to ~0.2 V,
sinking ~2 mA. RC is ~80 ns against a 1.667 ms pulse, so edges do not matter.

**Verify before relying on it:** that the header's 5 V pin really is a supply (5 V pin to
GND pin ≈ 5 V). The figure implies it; we have not measured it.

**Identifying unlabelled wires**, since the cable in use is not a factory one:
- **GND (pin 6) vs OPTO_GND (pin 1)**, both reading 0 V — camera powered **off**, meter in
  continuity mode, one probe on the chassis or USB shell. **Pin 6 reads ~0 Ω; pin 1 reads
  open**, because it is isolated by design. This doubles as a check that the isolation is
  actually intact.
- **The two wires reading ~3 V** — a supply sources current, an input idling high through
  an internal pull-up does not. Load with **1 kΩ to GND**: the real 3.3 V rail barely
  moves (3.3 mA of a 120 mA budget); an internal pull-up of tens of kΩ collapses to a few
  hundred mV. Safe for both cases. Better still, toggle the line in PySpin and watch which
  wire moves — a supply pin does not respond to a Line command.

- The camera side selected **Line1 = `ExposureActive`**. On a cased Blackfly S, Line1 is
  the **opto-isolated output**, which is **open-drain**: it needs that pull-up, and the
  resulting logic is typically **inverted** (exposure active ⇒ output pulled low). So the
  expected strobe may well arrive as 1.667 ms high / 15 ms low rather than the other way
  round. That costs nothing — there is still exactly one rising edge per frame — and
  `find_camera_strobe_in_rec.py` reports it rather than being confused by it. The
  non-isolated bidirectional line is the alternative if a direct 3.3 V push-pull drive is
  preferred.
