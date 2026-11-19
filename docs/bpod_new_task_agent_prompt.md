# Prompt for the agent that writes the new Bpod task

Copy everything below the line into the new agent's first message. It is written to
be self-contained: the agent has none of our conversation history.

Collected from: `post_ks_correction/RESEARCH_LOG.md` §5bd.74 and §5bd.75 (the two
proposed changes and the measurements behind them), the GUI/state dump of
`Z:\Gil\Shamir_1\bpod\Shamir01_Dual2AFC_nat_Sep25_2026_Session1.mat`,
`docs/bpod_mcu_sync_wiring.md` (the Trodes sync contract) and
`D:\Gil\sync_pipeline\TASK_TIMELINE.md` (the state-by-state trace).

---

# Write a new Bpod protocol for the rat striatum confidence experiment

You are writing MATLAB code for a Bpod (Bpod_Gen2) behavioural rig in a rodent
electrophysiology lab. The rig runs a working protocol today, every day, on a rat
with a Neuropixels probe in its head. Recording time with this animal is
irreplaceable.

## 0. THE RULE THAT OUTRANKS EVERY OTHER INSTRUCTION

**You may not modify, delete, rename, move, or overwrite a single existing file.**

Everything you produce is a NEW file in a NEW folder. After you are finished, the
protocol that runs today must run *exactly* as it does now — same code, same
settings, same behaviour, byte for byte. The worst possible outcome of this task,
far worse than producing nothing, is that the existing task stops working or
silently changes. A rat trained for months and a probe already in the brain are
behind that file.

**Your very first action, before reading any file, before planning, before
anything else: write this to your memory directory, verbatim.** Create
`bpod_protocol_additive_only.md`:

```markdown
---
name: bpod_protocol_additive_only
description: "NEVER modify, delete or rename anything in the existing Bpod protocol or on the Bpod PC - the new task is new files only"
metadata:
  type: feedback
---

On the Bpod behaviour computer I add files; I never change them. The existing
protocol `Dual2AFC_nat` and everything else already on that machine must run
exactly as before after my work - unchanged, byte for byte. The new task is a
NEW protocol folder with a NEW name, created by copying, never by editing in
place. No edits, no deletions, no renames, no moves, no "small cleanups", no
reformatting, no fixing a bug I notice in the old protocol, not even in a file I
am sure nobody uses.

**Why:** a rat trained for months is running this task daily with a Neuropixels
probe already implanted. Breaking or silently altering the running task destroys
irreplaceable data and months of training. Gil, 2026-11-19: "ruin our current
function task is the worst can happen."

**How to apply:** before starting, record a checksum of every file under the
existing protocol folder; after finishing, re-check and show that the list is
identical. If a change to shared/common Bpod code seems unavoidable, STOP and ask
Gil - do not do it. Copy the function into the new protocol folder under a new
name instead.
```

Then add a line to your `MEMORY.md` index pointing at it. Do this first. Re-read
it before every file write.

Practical form of the rule:

- New protocol folder: `<BpodUser>\Protocols\<NewName>\<NewName>.m` plus its own
  copies of every helper it needs. Bpod requires the main `.m` to have the same
  name as its folder.
- Helper functions live inside the new folder, even when they are copies of
  functions that already exist elsewhere. Do not add to, or edit, anything in
  `Bpod_Gen2\` or any shared `Functions\` directory.
- If a helper is shared and used unchanged, call it where it is — do not touch it.
- Never write into the existing protocol folder, its `Session Settings`, or any
  `Data\` folder.
- Before you start: `Get-FileHash -Algorithm SHA256` (recursively) every file under
  the existing protocol folder and save the list. After you finish: run it again and
  show both lists are identical. That proof is part of the deliverable.

## 1. What you are building

A new Bpod protocol, derived from the existing `Dual2AFC_nat`, that adds three
experimental manipulations. Each is behind its own GUI switch, and **with every new
switch at its default the new protocol must behave exactly like the current one** —
same states, same timings, same stimulus, same saved fields. That property is what
makes it safe to run on the rat, and it is how the first test session will be
validated: run the new protocol with defaults, and the resulting data file should be
indistinguishable from a normal session.

Suggested name: `Dual2AFC_nat_conf` (confidence). Avoid digits in the name that
could read as a date; see §7.

## 2. The files you need, and where they are

On the Bpod computer (paths recovered from the session file's saved settings — the
protocol folder is `C:\Users\KepecsLab\Documents\BpodUser\Protocols\Dual2AFC_nat\`):

1. **The whole existing protocol folder**, every `.m` and `.mat` in it, including
   subfolders — the main protocol file, the state-matrix builder, the stimulus
   generation, the custom-data init/update functions, the plotting functions. You
   can skip the `naturalsounds165\` wav library except the two files actually used
   (`frogs2.wav`, `Passer_Montanus.wav`).
2. **The Bpod settings folder** (`BpodUser\Settings\*.mat`) — the hardware
   configuration. This is where you find out which state machine, which modules
   (HiFi? PsychToolbox? PulsePal? olfactometer? NIDAQ?) and, critically, how the
   6-bit state-number sync word in §3 is produced.
3. **One `Session Settings` .mat** for this protocol and subject — the parameter set
   actually loaded at the rig.
4. **One recent data file**, e.g.
   `Z:\Gil\Shamir_1\bpod\Shamir01_Dual2AFC_nat_Sep25_2026_Session1.mat` — ground
   truth for what the protocol saves.
5. **The Bpod_Gen2 version** (folder + git revision if available). Not to modify —
   to know which API you are writing against.

Everything in §5 below was measured from file 4 and can be trusted without
re-deriving it. Everything about *code structure* must come from files 1–3: do not
guess what a function does, read it.

## 3. The rig, and what else depends on this protocol

The Bpod is not alone. Every state transition is broadcast as a **6-bit parallel
word on six digital lines into the SpikeGadgets/Trodes acquisition controller**
(`Din1..Din6` = bits 0..5), and the value of that word **is the Bpod state number**.
This is how the behaviour is aligned to the neural recording, measured to 0.1–0.3 ms.
A separate Python pipeline decodes it. Therefore:

- **State 1 must stay the first state of every trial** (`PreITI_start` today). The
  aligner finds trial starts by looking for code 1. Code 0 = all lines low = between
  trials.
- **The protocol must not exceed 63 states** (6 bits). It uses 40 today, so there is
  room for 23 new ones — not unlimited.
- **No state may be shorter than 0.1 ms.** The recording samples at 30 kHz; the
  existing protocol's pass-through states are 0.1 ms = 3 samples, which is the floor.
- **Find out how that word is emitted** before adding any state — it is either
  Bpod's own sync-channel feature (automatic for new states) or explicit output
  actions written per state in the state matrix (then every new state needs its
  own). Report which it is. Guessing here breaks the neural alignment silently.

The Python side also reads the saved file directly, so these are a contract:

- State names already in use must keep their current names and meanings. In
  particular `start_Lin` / `start_Rin` define "the rat made a choice",
  `stimulus_delivery_min` / `stimulus_delivery` define the cue, `water_L` / `water_R`
  the reward, `wait_Sin`, `stay_Cin`, `ITI`, `PreITI_start`. New states get new
  names; nothing existing is renamed or repurposed.
- `SessionData.Custom` fields keep their names and meanings: `ChoiceLeft`,
  `ChoiceCorrect`, `Rewarded`, `Feedback`, `FeedbackTime`, `FixBroke`,
  `EarlyWithdrawal`, `FixDur`, `MT`, `ST`, `CatchTrial`, `DV`, `AuditoryOmega`,
  `RewardMagnitude`, `TrialNumber`, `StimDelay`, `FeedbackDelay`, `MinSampleAud`,
  `ResolutionTime`. **Add new fields; never rename, drop or re-use one.** A new
  per-trial field must be one entry per trial, in trial order.
- The data file name is produced by Bpod as
  `<Subject>_<Protocol>_<Mon><DD>_<YYYY>_Session<N>.mat`; our matcher parses the
  date and the session number only, so a new protocol name is safe.

## 4. What the experiment is trying to learn

One rat, a single straight Neuropixels shank through dorsal striatum into the
ventral striatum / NAc border (units from 2.76 to 6.42 mm below the brain surface).
The question is how **decision confidence** is represented along that dorsoventral
axis, in a circuit implicated in OCD.

What the current data already show (three sessions, hundreds of units):

- The rat's **waiting time at the side port** is its confidence report (the task has
  catch trials where no feedback ever comes, so how long it waits is a read-out of
  how sure it is).
- About 10 % of units track stimulus strength `|DV|` at the cue, about 8 % track
  P(correct) at the choice port; the cue signal is biased dorsally (14.4 % dorsal vs
  6.0 % ventral), the post-choice error signal is everywhere.
- **Post-error adjustment is essentially absent**, and the task gives it no room to
  appear: sampling time is pinned, feedback delay is short, error trials are never
  repeated.

Three things the current design cannot do, which the new protocol is for:

1. **Separate confidence from a stimulus echo.** A cell that fires more on easy
   trials at the choice port might be encoding "I am probably right" or might just be
   echoing the loud/easy stimulus it heard a moment ago. Nothing in the current task
   separates these.
2. **Separate confidence from expected value.** Waiting time tracks both.
3. **Measure a commitment time.** The rat never chooses how long to listen, so there
   is no per-trial decision time and no drift-diffusion bound to fit.

## 5. The current task, as measured (trust these numbers)

40 states, in order (state number = DIO code):

```
 1 PreITI_start            11 wait_Sin                21 rewarded_Rin_grace     31 DrinkingR
 2 preITI                  12 start_Lin               22 unrewarded_Lin_start   32 DrinkingGraceR
 3 wait_Cin                13 start_Lin2              23 unrewarded_Lin         33 DrinkingGraceL
 4 broke_fixation          14 start_Rin               24 unrewarded_Lin_grace   34 timeOut_BrokeFixation
 5 stay_Cin                15 start_Rin2              25 unrewarded_Rin_start   35 timeOut_EarlyWithdrawal
 6 stimulus_delivery_trigger 16 rewarded_Lin_start    26 unrewarded_Rin         36 timeOut_IncorrectChoice
 7 No_Stim                 17 rewarded_Lin            27 unrewarded_Rin_grace   37 timeOut_SkippedFeedback
 8 stimulus_delivery_min   18 rewarded_Lin_grace      28 water_L                38 skipped_feedback
 9 early_withdrawal        19 rewarded_Rin_start      29 water_R                39 missed_choice
10 stimulus_delivery       20 rewarded_Rin            30 DrinkingL              40 ITI
```

The trial, in time: `PreITI_start` (0.05 s) -> `preITI` (0.5 s) -> `wait_Cin`
(center LED on, waits up to `CenterWaitMax` 20 s for the poke) -> `stay_Cin`
(fixation hold, `StimDelay` drawn 0.2–0.4 s; leaving early = `broke_fixation` -> 3 s
timeout) -> `stimulus_delivery_trigger` -> `stimulus_delivery_min` (`MinSampleAud`
0.35 s; leaving early = `early_withdrawal` -> 3 s timeout) -> `stimulus_delivery`
-> `wait_Sin` (`ChoiceDeadLine` 3 s to poke a side port) -> `start_Lin` /
`start_Rin` -> rewarded/unrewarded branch -> the **feedback delay**, which is the
waiting period the rat can abandon -> `water_L`/`water_R` -> drinking -> `ITI` (1 s).

The stimulus is **not a click train**. It is a linear morph of two fixed natural
sounds, `x(t) = w*frogs2 + (1-w)*Passer_Montanus`, played at 192 kHz, with
`w = Custom.AuditoryOmega` and `DV = 2w - 1`. **The mixing ratio is constant for the
whole 350 ms** — two trials with the same `w` are the same sound sample for sample.
Difficulty levels `Aud_Levels.AudFracHigh = [5 30 45 55 70 95]` %.

Key GUI parameters as they run today:

```
MinSampleAud 0.35   MinSampleAudMin 0.35   MinSampleAudMax 0.35   (so: pinned)
AuditoryStimulusTime 0.35   AuditoryStimulusType 3 (natural)      StimDelay 0.2-0.4
ChoiceDeadLine 3    CenterWaitMax 20   ITI 1   PreITI 0.5   DrinkingTime 0.3
FeedbackDelaySelection 3   FeedbackDelayMin 0.5  Max 8  Tau 1.5  Grace 0.3
PercentCatch 0.15   CatchError 1   ErrorRepeat 0   StartEasyTrials 30
RewardAmount 25     RewardMin 6   RewardMax 50   MaxSessionTime 360
Ports_LMRC 1238     PortLEDs 1     TrialSelection 3
BlockTable: 8 blocks, BlockLenL/R all 5000, RewL/RewR all 1, CategoryBoundary 50
            (i.e. the block machinery exists and is currently switched off)
```

And the behaviour it produces (16, 17, 18 September):

```
session            starts  completed    early withdrawal   accuracy
20260916_110311       953   478 (50 %)    399 (42 %)        0.851
20260917_100043      1106   605 (55 %)    457 (41 %)        0.893
20260918_100248      1115   593 (53 %)    478 (43 %)        0.885
```

Sample time is exactly 0.3501 s on **every** completed trial, movement time ~0.25 s
median, waiting time ~6.5 s median.

## 6. The three changes to implement

Every one is off by default. Each needs: a GUI parameter (in the existing GUI
panel/tab structure, so it appears in the normal parameter window), the state-machine
logic, and per-trial fields saved in `Custom` so the analysis can find what happened.

### A. Free sampling — the cue lasts as long as the rat stays in the center port

Today `MinSampleAud` = `AuditoryStimulusTime` = 0.35 s, and `stimulus_delivery` has a
duration of 0.000 s on all 605 trials of one session — so the free-response scaffold
may already be wired in the state machine and simply terminating immediately.
**Check this in `stateMatrix.m` first and report it**: if it is already there, this
change may be mostly a parameter and sound-length question, not new states.

What is wanted: after a minimum, the sound keeps playing while the rat holds the
center port, and ends when it withdraws; the withdrawal time is the trial's
sampling time (a commitment time, and a second confidence read-out).

- `FreeSampling` (0/1, default **0**).
- `FreeSampleMax` (default 2.0 s) — a ceiling, after which the state ends itself.
- The minimum stays `MinSampleAud`, which Gil will lower to 0.05–0.10 s. Make sure
  nothing in the code assumes min == max or a 0.35 s sound.
- The **sound must be rendered long enough** to cover `FreeSampleMax` (loop or
  generate the full length) and must be **stopped on center-port-out**, not left
  playing into the choice. Say explicitly in your write-up how playback is started
  and stopped, and what the worst-case latency is.
- Save per trial: the actual sampling duration and how it ended (withdrew / hit the
  ceiling / broke the minimum). `Custom.ST` already exists — keep its meaning and add
  new fields rather than redefining it.
- Known risk, from the data: the rat already withdraws early on 42 % of trials, so
  with the minimum where it is this change would floor at the minimum and measure
  nothing. That is a parameter problem for Gil, not a code problem, but the code must
  not stand in the way of a minimum as low as 0.05 s.

### B. Post-decision evidence — a short burst *after* the choice is committed

This is the control for the echo-vs-confidence ambiguity. After the rat has poked a
side port and can no longer change its answer, play a short extra burst of the same
kind of sound, on a minority of trials, pointing one of three ways.

- `PostDecisionEvidence` (0/1, default **0**).
- `PercentPDE` (default 0.30) — fraction of completed trials that get a burst.
- `PDEDuration` (default 0.15 s, range 0.10–0.20).
- Three conditions, randomly assigned among PDE trials: **congruent** (the morph
  favours the side the rat chose), **neutral** (50/50), **incongruent** (favours the
  other side). `PDEStrength` (default e.g. 0.70 fraction, i.e. the same scale as
  `AudFracHigh`).
- **The three must be physically matched** — identical duration, identical overall
  loudness/RMS, same two source sounds, differing only in the mixing ratio, so that
  the only thing that differs between conditions is what the sound *means* relative
  to the rat's own choice. This matters more than anything else in the feature; if
  the three conditions differ in energy the experiment is dead.
- Timing: the burst starts as soon as the side-port entry is registered
  (`start_Lin` / `start_Rin`), before/at the start of the waiting period, and must
  not alter the length or structure of the waiting period itself. The waiting time is
  the measurement; the burst must not truncate or extend the states around it.
- Save per trial: `PDETrial` (0/1), `PDECongruence` (-1 incongruent / 0 neutral /
  +1 congruent), the burst's own mixing ratio, its onset time, and its duration.
- The prediction being tested: a stimulus echo must be **flat** across the three
  conditions; a real confidence signal must move monotonically with congruence, and
  must move the waiting time on the same trials.

### C. Reward blocks — separating confidence from expected value

The protocol already carries block machinery (`BlockTable` with `BlockNumberL/R`,
`BlockLenL/R`, `RewL/RewR`, `NoiseL/NoiseR`, `CategoryBoundary`, plus
`Custom.RewardMagnitude`, `RichLeft`, `RichRight`, `BlockTrialL/R`), currently
switched off (block length 5000, reward multiplier 1 on both sides).

This is therefore mostly **verification, not new code**: read the block code path,
confirm it still works in the new protocol, confirm `Custom.RewardMagnitude` and the
block fields are saved per trial, and provide a ready-made block preset (e.g.
alternating 60–80 trial blocks with left/right reward multipliers such as 1.0/2.0)
that Gil can load. If the path is broken or incomplete, say so — do not fix the old
protocol.

### Also available, decide and report rather than silently implement

`ErrorRepeat` (exists, currently 0) and the feedback-delay distribution
(`FeedbackDelaySelection/Min/Max/Tau`) are existing GUI parameters that would give
post-error adjustment room to appear. They need no code. Mention them in your
write-up as settings changes. An explicit "re-check / confirm" port is a larger
state-machine change and is **out of scope** unless Gil asks.

## 7. Acceptance criteria — check every one and show the evidence

1. Every file under the existing protocol folder has the same SHA-256 as before you
   started. Show the before/after comparison.
2. The new protocol is a self-contained folder; nothing outside it was created or
   changed except, if Gil asks for it, a new `Session Settings` file for the new
   protocol.
3. With all three new switches at their defaults, the new protocol produces the same
   40 states in the same order with the same numbers, the same timings, and the same
   `Custom` fields as the current one.
4. Total state count ≤ 63, `PreITI_start` is state 1, no state shorter than 0.1 ms.
5. Every new state (if any) emits the correct sync word — state what mechanism does
   this and why it is automatic or how you handled it.
6. No existing state name, and no existing `Custom` field name, changed meaning.
7. New `Custom` fields are one entry per trial, in trial order, written for every
   trial (NaN where not applicable), never only on the trials where the feature fired.
8. The protocol name contains no `Xxx##_####` date-like pattern (the data-file
   matcher reads the first such token as the session date).
9. The code runs in MATLAB without error in Bpod's emulator mode, if available. **Do
   not run it on the rig and do not run it on a rat** — that is Gil's call, with a
   test session, with no animal first.

## 8. How to work

1. Write the memory file (§0). Then checksum the existing protocol folder.
2. **Inventory and read before writing.** List every file in the protocol folder,
   read the main protocol file and the state-matrix builder end to end, and write a
   short map of what calls what. Do not start from an assumption about how a standard
   Dual2AFC is written — read this one.
3. Answer the open questions in §9 from the source, and report the answers.
4. Then propose the design (states added, parameters added, fields saved) and get
   Gil's agreement before writing the bulk of the code.
5. Write the new folder. Keep the diff against the original visible: a
   `CHANGES_FROM_Dual2AFC_nat.md` inside the new folder, listing every function that
   differs from its original and why, so a reader can audit the copy.
6. Deliver: the new folder, the CHANGES document, the checksum proof, a short test
   plan for the first emulator and first no-animal rig session, and the list of GUI
   values Gil should set for each of the three experiments.

If anything is ambiguous, ask. If something seems to require touching an existing
file, stop and ask — there is always an additive alternative, and if there is not,
that is a decision for Gil, not for you.

## 9. Questions to answer from the source before you design anything

1. Is the free-response scaffold already in the state machine? `stimulus_delivery`
   has a duration of exactly 0.000 s on every completed trial — is that a state that
   already waits for center-port-out and is simply configured to end immediately?
2. How is the sound played — HiFi module, PsychToolbox, PulsePal? (`Custom` carries
   both `PsychtoolboxStartup` and `PulsePalParamStimulus`, and the settings carry a
   `HiFiModule` struct, so more than one path may exist.) How is a waveform loaded,
   triggered, and **stopped**?
3. How is the 6-bit state-number sync word produced — Bpod's own sync configuration,
   or explicit per-state output actions in the state matrix?
4. Where are per-trial parameters drawn (the look-ahead buffer: some `Custom` arrays
   are longer than `nTrials`), and where would a new per-trial condition such as the
   PDE type naturally be drawn and saved?
5. Does the existing block code path actually work, end to end, with a non-trivial
   `BlockTable`?
