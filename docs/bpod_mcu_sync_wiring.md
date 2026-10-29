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
- **`trodesexport` writes only `Din1..Din6`** — the `.DIO` folder of an exported
  session contains six `dio_Controller_Din{1..6}.dat` files and nothing for
  `Din7`/`Din8`. **A signal moved onto `Din7` or `Din8` will not be exported and
  the sync pipeline will never see it.** Plan rewiring around that.
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
