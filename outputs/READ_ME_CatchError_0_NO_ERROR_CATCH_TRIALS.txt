================================================================================
  READ THIS BEFORE USING ANY POPPER SESSION FOR A CONFIDENCE ANALYSIS.
  CatchError = 0 ON EVERY POPPER SESSION RECORDED SO FAR.
================================================================================
written 2026-11-16, from scripts/audit_catch_trials.py over every Bpod file of
both rats. Per-session evidence: outputs/catch_trial_audit.csv in the
spike_sorting_agent repo.

WHAT IS FINE
------------
Catch trials on CORRECT choices exist and work, in every Popper session:
PercentCatch = 0.15, 11-16 % of trials carry Custom.CatchTrial = 1, the reward is
withheld on 100 % of them, and the rat waits a median of 5.0-7.1 s against
1.0-1.5 s when it is rewarded. Nothing is wrong with those.

WHAT IS MISSING
---------------
CatchError is the separate switch for whether an ERROR trial is also a catch
trial. On Popper it is 0, on Shamir it is 1, on every session of each.

  CatchError = 1 (Shamir)  an error is never announced. The rat waits until it
                           gives up, and that wait is its own decision. Median
                           wait on error trials: 4.8-6.9 s. The state
                           timeOut_IncorrectChoice is entered ZERO times.
  CatchError = 0 (Popper)  the error is announced at the end of the feedback
                           delay, which WE drew. Median "wait" on error trials:
                           1.1-1.5 s -- our delay, not the animal's decision.
                           timeOut_IncorrectChoice is entered on most error
                           trials (79 of 92 completed errors on 20261112_082028).

WHY IT MATTERS
--------------
A time-investment / confidence analysis needs the waiting-time curves for correct
AND error trials together; the two curves crossing as a function of evidence
strength IS the measurement. Popper gives only the correct-trial curve, so that
analysis cannot be done on these sessions no matter how good the ephys is.

WHAT THESE SESSIONS ARE STILL GOOD FOR
--------------------------------------
Everything that does not depend on waiting time on error trials. The ephys, the
sorting and the Bpod alignment are all sound and were checked, not assumed:
20261112_082028 syncs with 100 % of 16,410 Bpod events matched, a worst-case
time-conversion error of 29 ms, and 1085 of 1085 trials valid; 61 of its 120 good
units are reward-modulated, with unit 280 going 1.2 -> 11.2 Hz at the valve.

AFFECTED (every Popper session that exists as of 2026-11-16)
    20261026, 20261027, 20261028, 20261030, 20261109, 20261110,
    20261111, 20261112, 20261113, 20261116

TO FIX IT FOR FUTURE SESSIONS: set CatchError = 1 in the protocol's GUI settings
before the next recording. Nothing can recover it for the sessions above.
