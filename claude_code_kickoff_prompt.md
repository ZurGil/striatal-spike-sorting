I want to build a post-Kilosort4 curation agent for striatal Neuropixels recordings,
following the attached design spec (striatal_spike_sorting_agent_spec.md).

## Data layout (per-session)

Each session lives in a folder like:
D:\Gil\Shamir\<session_id>.rec\

Contents:
- <session_id>.rec                          — original raw Trodes recording
- <session_id>.kilosort\                    — Trodes-exported inputs to Kilosort:
    - <session_id>.channelmap_probe1.dat    — Trodes-native channel map (likely not
                                               needed directly if the JSON below is what
                                               KS4 was actually given — verify channel
                                               ordering matches before ignoring it)
    - <session_id>.probe1.dat               — exported per-probe raw trace, fed to KS4
    - <session_id>.timestamps.dat           — sample timestamps
    - <session_id>_custom_neuropixels_map.json — KS4 channel map (chanMap, xc, yc,
                                               kcoords, n_chan=384) — sample attached
    - kilosort4\                            — full standard KS4 output folder
- <session_id>.timestampoffset              — not needed for this pipeline
- <session_id>.DIO\                         — behavioral digital I/O, not needed here

## What I want first (in this order)

1. Confirm you can read <session_id>.probe1.dat and the channel map JSON correctly —
   report back the sampling rate, channel count, and probe geometry you infer, so I can
   sanity-check it before anything else runs.
2. Confirm you can read the existing kilosort4\ output (spike_times.npy, spike_clusters.npy,
   templates.npy, etc.) and summarize what's there (n_units, spike counts) as a smoke test.
3. Implement Section 1 of the spec (the validation gate) first, on this one session, before
   building anything else — I want quantitative evidence of whether spike recovery is even
   needed before we invest in the rest.
4. Only after that: scaffold the full pipeline per the spec, with a single entry point that
   takes the session root path and auto-discovers everything else by the naming convention
   above, writing kilosort_original\ and agent_processed\ inside the session's .kilosort\
   folder (non-destructive, per spec Section 9).

## Open questions for you to investigate/ask me

- What's already installed here (SpikeInterface? kilosort pip package? phy? CUDA/GPU
  driver)? Check and tell me what's missing before assuming.
- Best way to read Trodes .rec / .probe1.dat format — via SpikeInterface's
  spikegadgets extractor, or does this data need a different reader? Investigate and
  propose an approach rather than guessing.
- Confirm whether Kilosort4's detect_spikes / template deconvolution step can be re-run
  standalone (with an adjusted Th_learned) without redoing full clustering — this matters
  for spec Section 2.7's threshold-sweep approach.

Let's go step by step — don't scaffold the whole thing at once. Start with step 1 above and
show me what you find before proceeding.
