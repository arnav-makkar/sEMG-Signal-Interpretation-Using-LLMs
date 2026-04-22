"""
Clinical EMG Rules — injected directly into prompts.
Source: IP (1).pdf — EMG Signal Interpretation Using LLMs
"""

ACTIVATION_RULES = """
[A1] Activation Level: RMS relative to session mean. Low: <30%. Moderate: 30-70%. High: 70-100%. Very High: >100%.
[A2] Gesture Patterns: Flexion = S1-S4 (palmar). Extension = S5-S8. Deviations = placement error or compensation.
[A3] Co-contraction: Agonist + antagonist RMS both >50% peak simultaneously. Normal for power grip; abnormal for fine motor.
[A4] Consistency: CV of RMS. <20% consistent. 20-40% moderate. >40% difficulty maintaining gesture.
[A5] Dominant Sensor: Highest average RMS. Report dominant-to-weakest ratio.
[A6] Temporal Profile: Classify as Sustained, Ramp-up, Ramp-down, Intermittent, or Compensatory shift.
[A7] Force Estimation: RMS has linear-to-quadratic relationship with force. Valid only for relative within-session comparisons.
[A8] Motor Unit Recruitment: Higher WAMP/ZC = fast-twitch recruitment (increasing effort or fatigue).
[A9] Onset Detection: Threshold = baseline mean + k*SD (k in [1,3]) with minimum duration 20-100 ms.
[A10] Envelope Smoothing: Rectify + low-pass. Use <=25 ms for timing accuracy; up to 250 ms for amplitude trends.
[A11] False Onset Guard: Require consecutive windows above threshold. Single-window crossings = noise spikes.
[A12] Duty Cycle: % time above activation threshold per sensor. Sustained vs intermittent classification.
[A13] Normalization: Report as %MVC or %baseline. Without normalization, cross-session/subject comparisons are unreliable.
""".strip()

QUALITY_RULES = """
[Q1] Baseline Noise: Resting baseline noise RMS below 0.005 mV (5 uV) is good. Above 0.010 mV = POOR QUALITY.
[Q2] SNR Threshold: >15 dB reliable. 10-15 dB MARGINAL. <10 dB UNRELIABLE.
[Q3] Powerline Contamination: PowerlineRatio >0.10 = significant PLI. Recommend notch/adaptive cancellation.
[Q4] Clipping/Saturation: ClippingRatio >0.01 or >0.5% samples at extreme percentiles = amplifier saturation. Reduce gain.
[Q5] Spectral Shape: Broadband 10-500 Hz expected, peaking 50-150 Hz. Spectral entropy <3.0 bits = single-frequency artifact.
[Q6] Cross-talk: Opposite-side sensors correlated >0.8 during unilateral gesture = crosstalk. Review electrode spacing.
[Q7] Motion Artifact: DC offset shift >0.5 mV between segments + low-freq dominance (<5 Hz) = motion artifact.
[Q8] Dropout/Flatline: DropoutPct >0.5% or flatline >200 ms = possible disconnection. Mark segment unusable.
[Q9] LF Artifact: LF_artifact_ratio (1-10 Hz / 20-450 Hz) >0.30 = motion/cable artifact.
[Q10] Anomalous Distribution: Kurtosis >10 = spike artifacts. Unusually high ZC/SSC relative to RMS = noise-dominated.
""".strip()

DRIFT_RULES = """
[D1] Drift Rate: DriftSlope >0.001 mV/s = baseline wander. Mild: 0.001-0.005. Moderate: 0.005-0.01. Severe: >0.01 mV/s.
[D2] DC Offset Trend: Monotonic change >0.05 mV over 5+ consecutive segments = systematic drift.
[D3] Noise Floor Drift: BaselineNoiseRMS increase >50% from first to last 5 segments = interface degradation.
[D4] Impedance Drift: Simultaneous noise increase + SNR decrease = rising impedance. Common after 15-20 min.
[D5] Sudden Shift: DCOffset change >0.1 mV between consecutive segments = electrode displacement. Flag for exclusion.
[D6] Perspiration: Gradual noise decrease in first 5-10 min is normal (skin hydration). Concern if it suddenly increases.
[D7] Warm-Up Stabilization: Electrode-skin impedance changes during first 10-15 min. Use minute 15+ as stable baseline.
[D8] Amplitude-Frequency Dissociation: RMS change >20% but MDF change <5% = sweat/impedance drift, not fatigue.
[D9] Step Change Detection: Step >3xMAD = electrode displacement or cable tug. Annotate timing.
[D10] Spatial Reconfiguration: Activation map cosine similarity <0.7 for 3+ segments = spatial pattern change.
""".strip()

FATIGUE_RULES = """
[F1] MDF Decline (Gold Standard): 10-20% decline from baseline = moderate fatigue. >20% = severe.
[F2] MNF Parallel: MNF should decline with MDF. MNF-only decline may be noise/artifact.
[F3] RMS Increase: Sustained contraction RMS increases 25-50% from motor unit recruitment. RMS up + MDF down = strong fatigue.
[F4] L/H Ratio: Increase >30% from baseline = spectral compression consistent with fatigue.
[F5] Entropy Decrease: SampleEntropy decline >15% = fatigue-related motor unit synchronization.
[F6] ZC Decrease: >10% decrease supports fatigue (slowed conduction velocity).
[F7] Onset Detection: Onset = 3+ consecutive MDF-declining segments. Report time in seconds.
[F8] Fatigue Index: FI = (MDF_baseline - MDF_final) / MDF_baseline x 100%. Mild: 5-10%. Moderate: 10-20%. Severe: >20%.
[F9] Multi-sensor Consistency: Multiple sensors fatiguing simultaneously strengthens finding.
[F10] Confound Check: If SNR drops concurrently, electrode degradation is more likely than fatigue.
[F11] MDF Slope Rate: Track Hz/min. Sustained -0.1 to -0.5 Hz/min = fatigue. Outside this range = possible artifact.
[F12] Spatial Compensation: If concentration index decreases while total RMS is maintained, load is redistributing.
[F13] Prerequisite Gate: Before concluding fatigue, confirm: (a) quality GOOD/MARGINAL, (b) drift None/Mild, (c) D8 excludes sweat-only amplitude change.
""".strip()

PHYSIOLOGICAL_CONTEXT = """
[P1] EMG Range: Meaningful 10-500 Hz. <10 Hz = motion artifact. >500 Hz = noise.
[P2] Motor Units: 100-1000 fibers each. Henneman size principle. Fast-twitch = higher freq EMG.
[P3] Conduction Velocity: Normal 3-6 m/s. Fatigue reduces it, shifting spectrum lower.
[P4] Typical Amplitude: 0-6 mV p-p. Forearm moderate grip: 0.1-1.0 mV RMS. <0.01 mV = not over muscle.
[P5] Session Phases: 0-5 min warmup, 5-10 stable, 10-15 early fatigue, 15-20 pronounced. Impedance stabilizes ~10-15 min.
[P6] Inter-Subject Variability: 3-5x amplitude variation. Within-subject trends far more reliable.
[P7] Amplitude Caution: sEMG amplitude cannot infer absolute force, compare between muscles, or determine specific activation in isolation. Valid for relative changes within the same muscle/session/electrode setup.
""".strip()

SYSTEM_PROMPT = """You are an expert clinical EMG analysis assistant. You interpret extracted EMG signal features to provide structured clinical assessments for physicians.
Base analysis on provided features and rules only. Never fabricate data.
Do not invent new thresholds. If data is insufficient, say so and lower confidence.
Do not diagnose disease. Role: summarization, explanation, flagging issues.
Return STRICT JSON matching the output schema. No markdown."""
