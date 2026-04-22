# Muscle Activation Analysis Report

## Overview
Participants analyzed: 9

## RAG System
Backend: ChromaDB with sentence-transformers (all-MiniLM-L6-v2)
Papers indexed: Konrad 2005, Nazmi 2016, Phinyomark 2012, Vigotsky 2018, Yang 2024 (EMGBench)
Rules: A1-A13 (activation) + P1-P7 (physiological context) embedded directly in prompt

## Summary Table
participant feature_source overall_activation temporal_profile consistency dominant_sensors  dominant_weakest_ratio  concentration_index  pattern_stability  co_contraction_ratio  flexor_extensor_balance  confidence  parse_success
         P1     raw_signal               high        Ramp-down  consistent           ['S4']                    7.27               0.2006             0.9983                0.3691                   2.7329        0.95           True
         P2     raw_signal          very_high        Sustained  consistent           ['S2']                    8.03               0.1846             0.9979                0.2324                   4.3190        0.95           True
         P3     raw_signal               high          Ramp-up  consistent           ['S4']                    5.66               0.1851             0.9942                0.4613                   2.2287        0.95           True
         P4     raw_signal          very_high          Ramp-up    moderate     ['S4', 'S8']                    2.67               0.1487             0.9926                0.7465                   1.3620        0.90           True
         P5     raw_signal          very_high        Sustained  consistent           ['S3']                   10.09               0.2247             0.9957                0.5793                   1.8871        0.90           True
         P6     raw_signal          very_high        Sustained  consistent           ['S4']                    4.83               0.1624             0.9960                0.8549                   1.0187        0.95           True
         P7     raw_signal          very_high        Sustained  consistent           ['S3']                    2.81               0.1461             0.9909                0.5376                   1.9005        0.90           True
         P8     raw_signal          very_high          Ramp-up  consistent           ['S3']                    7.79               0.1944             0.9982                0.4439                   2.2765        0.95           True
         P9     raw_signal          very_high        Sustained  consistent           ['S2']                    5.16               0.1607             0.9936                0.5529                   1.9757        0.95           True

## Per-Participant Details

### P1
- Feature source: **raw_signal**
- Overall activation: **high**
- Temporal profile: Ramp-down
- Dominant sensors: ['S4']
- Gesture pattern: Strong flexor dominance (S1-S4) compared to extensor group (S5-S8) with a flexor-extensor balance ratio of 2.73.
- Motor unit recruitment: True
- Flags: ['Ramp-down profile in S4 and S7', 'High flexor-extensor imbalance']
- Clinical notes: Sustained contraction shows high flexor activation, particularly in S4. The ramp-down profile in S4 suggests potential fatigue or loss of force maintenance over the 20-minute session. Co-contraction ratio is below the 0.5 threshold, indicating minimal antagonistic interference.

### P2
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Sustained
- Dominant sensors: ['S2']
- Gesture pattern: Strong flexor dominance (S1-S4) compared to extensor group (S5-S8) with a flexor-extensor balance ratio of 4.319.
- Motor unit recruitment: True
- Flags: ['Very high flexor activation', 'Flexor-extensor imbalance', 'S4 ramp-down and S8 ramp-up suggest potential shift in recruitment strategy']
- Clinical notes: The session shows a stable, sustained contraction with high flexor-dominant activity. S2 is the primary driver of the gesture. Low inter-channel correlation suggests minimal crosstalk. The ramp-down in S4 and ramp-up in S8 indicate a shift in neuromuscular strategy over the 20-minute duration.

### P3
- Feature source: **raw_signal**
- Overall activation: **high**
- Temporal profile: Ramp-up
- Dominant sensors: ['S4']
- Gesture pattern: Flexor group (S1-S4) shows significantly higher activation (Very High) compared to extensor group (S5-S8), consistent with palmar flexion dominance.
- Motor unit recruitment: True
- Flags: ['High flexor-extensor imbalance (2.23x)', 'S3/S4 very high activation levels', 'S8 amplitude decline in late phase']
- Clinical notes: Sustained isometric contraction with clear flexor dominance. S3 and S4 demonstrate high recruitment intensity. S8 shows a late-phase amplitude reduction, potentially indicating localized fatigue.

### P4
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Ramp-up
- Dominant sensors: ['S4', 'S8']
- Gesture pattern: Flexor-dominant activation (1.362 ratio). S4 (flexor) and S8 (extensor) show significant ramp-up profiles, suggesting increasing effort over the session duration.
- Duty cycle: N/A
- Motor unit recruitment: True
- Flags: ['High co-contraction ratio (>0.5)', 'Significant ramp-up profile in S4 and S8', 'High variability in S3']
- Clinical notes: The session exhibits a clear ramp-up activation profile with high co-contraction, suggesting a power-grip or high-effort task. Motor unit recruitment indicators are elevated in S8 and S4. The high inter-channel stability suggests a consistent, albeit fatiguing, neuromuscular strategy.

### P5
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Sustained
- Dominant sensors: ['S3']
- Gesture pattern: Flexor group (S1-S4) shows higher overall excitation compared to extensor group (S5-S8), with a flexor-extensor balance ratio of 1.8871.
- Motor unit recruitment: True
- Flags: ['High co-contraction ratio', 'Very high activation in S3', 'Flexor-dominant activation pattern']
- Clinical notes: The session demonstrates a stable, sustained contraction with clear flexor dominance (S3). High WAMP values across multiple sensors suggest significant motor unit recruitment. Co-contraction ratio (0.5793) exceeds the 0.5 threshold, indicating simultaneous agonist/antagonist activity consistent with a power grip.

### P6
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Sustained
- Dominant sensors: ['S4']
- Gesture pattern: Flexor (S1-S4) and Extensor (S5-S8) groups show balanced activation (ratio 1.0187), consistent with co-contraction during power grip.
- Motor unit recruitment: True
- Flags: ['High co-contraction ratio', 'S8 ramp-down profile', 'S6 ramp-up profile']
- Clinical notes: The session demonstrates a stable, high-intensity sustained contraction with significant co-contraction between flexor and extensor groups. S4 is the primary driver of the gesture. S8 shows a notable decrease in amplitude over time, while S6 shows a compensatory ramp-up, suggesting potential localized fatigue or shift in motor control strategy.

### P7
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Sustained
- Dominant sensors: ['S3']
- Gesture pattern: Flexor group (S1-S4) shows higher activation levels (Very High) compared to extensor group (S5-S8, High/Moderate). Flexor-extensor balance of 1.90 confirms flexor dominance.
- Motor unit recruitment: True
- Flags: ['High flexor-extensor co-contraction ratio', 'S1 intermittent activation instability', 'Flexor dominance']
- Clinical notes: Sustained isometric contraction with clear flexor dominance. High co-contraction ratio (0.5376) suggests significant antagonist activity, typical for power grip. S3 is the primary active site with high recruitment indicators.

### P8
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Ramp-up
- Dominant sensors: ['S3']
- Gesture pattern: Flexor group (S1-S4) shows significantly higher activation (Very High) compared to extensor group (S5-S8), with a flexor-extensor balance ratio of 2.2765.
- Motor unit recruitment: True
- Flags: ['High flexor-extensor imbalance', 'Very high activation levels in S2-S5']
- Clinical notes: The session demonstrates a stable, sustained contraction with a clear flexor-dominant pattern. High WAMP values in the flexor group (S2-S4) suggest robust motor unit recruitment. The low inter-channel correlation (0.2949) indicates minimal crosstalk risk.

### P9
- Feature source: **raw_signal**
- Overall activation: **very_high**
- Temporal profile: Sustained
- Dominant sensors: ['S2']
- Gesture pattern: Flexor group (S1-S4) shows significantly higher activation levels compared to extensor group (S5-S8), with a flexor-extensor balance ratio of 1.98.
- Motor unit recruitment: True
- Flags: ['High co-contraction ratio', 'Flexor-dominant activation pattern', 'S1 intermittent profile']
- Clinical notes: The session demonstrates a sustained, flexor-dominant contraction pattern. High co-contraction (0.55) suggests a power-grip style activation. S2 and S3 are the primary contributors to force production. S1 exhibits inconsistent activation compared to other sensors.

## Notes
- P1: Full features from raw signal (12 time/freq domain + onset/duty cycle)
- P2-P5: Limited features from features.csv (RMS, MAV, MeanFreq only)
- Amplitude not normalized to %MVC — cross-subject comparisons unreliable (Rule A13/P6)