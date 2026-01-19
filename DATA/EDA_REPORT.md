# Subjective EMG Fatigue Dataset - Exploratory Data Analysis Report
---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset Overview](#dataset-overview)
3. [Data Structure & Quality](#data-structure--quality)
4. [Descriptive Statistics](#descriptive-statistics)
5. [Fatigue Progression Analysis](#fatigue-progression-analysis)
6. [Gesture-wise Analysis](#gesture-wise-analysis)
7. [Participant Variability](#participant-variability)
8. [Correlation Analysis](#correlation-analysis)
9. [Outlier Analysis](#outlier-analysis)
10. [Key Findings & Insights](#key-findings--insights)
11. [Recommendations](#recommendations)
12. [Appendix: Generated Visualizations](#appendix-generated-visualizations)

---

## Executive Summary

This report presents a comprehensive exploratory data analysis of the **Subjective EMG Fatigue Dataset**, which captures self-reported fatigue scores from **41 participants** performing **5 different hand gestures** over **20-minute sessions**. The dataset uses the **Borg CR-10 scale** (0-10) to measure perceived exertion/fatigue at 20-second intervals (60 segments per session).

### Key Highlights

| Metric | Value |
|--------|-------|
| Total Observations | 205 |
| Participants | 41 |
| Gestures | 5 |
| Segments per Session | 60 |
| Session Duration | 20 minutes |
| Data Completeness | 99.98% |

**Main Finding:** Fatigue scores increase significantly from an average of **0.99** in the early phase to **5.39** in the late phase, representing a **4.4-point increase** on the Borg scale—clear evidence of progressive muscular fatigue during sustained gesture performance.

---

## Dataset Overview

### Study Design

The dataset captures subjective fatigue ratings during EMG (electromyography) experiments where participants performed repetitive hand gestures. The experimental protocol includes:

- **Participants:** 41 individuals (labeled P1-P44, with some IDs missing)
- **Gestures:** 5 distinct hand gestures (Gesture 1 through Gesture 5)
- **Duration:** 20 minutes per gesture session
- **Sampling:** Fatigue ratings collected every 20 seconds (60 total segments)
- **Scale:** Borg CR-10 scale (0 = no exertion, 10 = maximal exertion)

### Data Organization

| Column Type | Description | Count |
|-------------|-------------|-------|
| `P ID` | Participant identifier | 1 |
| `Gesture` | Gesture type (1-5) | 1 |
| `Segment 1-60` | Fatigue scores at 20-sec intervals | 60 |

**Total:** 62 columns × 205 rows

Each row represents one participant performing one gesture, with 60 time-series fatigue measurements spanning the 20-minute session.

---

## Data Structure & Quality

### Missing Value Analysis

| Metric | Value |
|--------|-------|
| Segments with missing values | 2 |
| Total missing values | 2 |
| Missing percentage | **0.016%** |
| Affected segments | Segment 59, Segment 60 |

**Assessment:** The dataset has excellent completeness with only 2 missing values out of 12,300 total data points. The missing values occur in the final two segments, likely due to early session termination for 1 participant.

### Data Types

- **Participant ID:** String (P1, P2, ..., P44)
- **Gesture:** Categorical string (Gesture 1-5)
- **Segment scores:** Numeric (float/integer)

### Data Quality Issues Identified

1. **Participant ID propagation:** The `P ID` column only contains the ID in the first row for each participant; subsequent rows (different gestures) have empty values. This is handled via forward-fill during analysis.

2. **One outlier value:** A single value of **17** was recorded (Segment 51, Gesture 1), which exceeds the Borg CR-10 scale maximum of 10. This may be a data entry error.

---

## Descriptive Statistics

### Global Statistics

| Statistic | Value |
|-----------|-------|
| Minimum | 0.0 |
| Maximum | 17.0 (outlier; 10.0 excluding outlier) |
| Mean | 3.85 |
| Standard Deviation | 2.92 |
| Median | 4.0 |

### Segment-Level Statistics Summary

The fatigue scores show clear temporal progression:

| Segment Group | Time Range | Mean Score | Median | Std Dev |
|---------------|------------|------------|--------|---------|
| Segments 1-10 | 0-3.3 min | 0.99 | 0.0 | 1.35 |
| Segments 11-20 | 3.3-6.7 min | 2.86 | 2.0 | 2.53 |
| Segments 21-30 | 6.7-10 min | 4.10 | 4.0 | 3.14 |
| Segments 31-40 | 10-13.3 min | 4.73 | 5.0 | 3.38 |
| Segments 41-50 | 13.3-16.7 min | 5.19 | 5.0 | 3.52 |
| Segments 51-60 | 16.7-20 min | 5.39 | 6.0 | 3.54 |

### Distribution Characteristics by Phase

| Phase | Mean | Median | Skewness | Interpretation |
|-------|------|--------|----------|----------------|
| Early (0-7 min) | 1.88 | 1.0 | +1.37 | Right-skewed (most values near 0) |
| Middle (7-13 min) | 4.36 | 4.0 | +0.20 | Nearly symmetric |
| Late (13-20 min) | 5.31 | 6.0 | -0.12 | Slightly left-skewed (ceiling effect) |

**Interpretation:** The distribution shifts from right-skewed (low fatigue) to nearly symmetric (moderate fatigue) to slightly left-skewed (high fatigue approaching scale maximum).

---

## Fatigue Progression Analysis

### Overall Progression Pattern

The mean fatigue score follows a **logarithmic growth pattern**:

| Time Point | Segment | Mean Score | Change from Start |
|------------|---------|------------|-------------------|
| Start | 1 | 0.08 | — |
| 2 minutes | 6 | 1.05 | +0.97 |
| 5 minutes | 15 | 2.71 | +2.63 |
| 10 minutes | 30 | 4.48 | +4.40 |
| 15 minutes | 45 | 5.21 | +5.13 |
| 20 minutes | 60 | 5.55 | +5.47 |

### Phase Analysis

```
Phase         | Time Range  | Mean Score | Fatigue Level
--------------|-------------|------------|---------------
Early         | 0-7 min     | 0.99       | Minimal
Middle        | 7-13 min    | 4.36       | Moderate  
Late          | 13-20 min   | 5.39       | Substantial
```

**Key Observation:** The steepest fatigue increase occurs in the **first 10 minutes** (0 → 4.5), after which the rate of increase slows considerably. This suggests:
- Initial rapid fatigue accumulation
- Partial adaptation or pacing behavior in later stages
- Possible ceiling effects as scores approach maximum

### Rate of Fatigue Change

The rate of fatigue increase (derivative) shows:
- **Highest rate:** Segments 5-15 (minutes 1.5-5)
- **Declining rate:** After segment 30 (10 minutes)
- **Near plateau:** Final 10 segments show minimal additional increase

---

## Gesture-wise Analysis

### Gesture Performance Summary

| Gesture | Overall Mean | Initial Mean (0-1.5 min) | Final Mean (18-20 min) | Fatigue Delta |
|---------|--------------|--------------------------|------------------------|---------------|
| Gesture 1 | 3.72 | 0.39 | 5.21 | **+4.82** |
| Gesture 2 | 3.99 | 0.41 | 6.14 | **+5.73** |
| Gesture 3 | 3.63 | 0.42 | 4.98 | **+4.56** |
| Gesture 4 | 3.82 | 0.37 | 5.40 | **+5.03** |
| Gesture 5 | 4.07 | 0.51 | 5.44 | **+4.90** |

### Gesture Rankings

**By Final Fatigue Level (highest to lowest):**
1. **Gesture 2** — 6.14 (most fatiguing)
2. **Gesture 5** — 5.44
3. **Gesture 4** — 5.40
4. **Gesture 1** — 5.21
5. **Gesture 3** — 4.98 (least fatiguing)

**By Fatigue Increase Rate:**
1. **Gesture 2** — +5.73 points (steepest increase)
2. **Gesture 4** — +5.03 points
3. **Gesture 5** — +4.90 points
4. **Gesture 1** — +4.82 points
5. **Gesture 3** — +4.56 points (gentlest increase)

### Interpretation

- **Gesture 2** consistently produces the highest fatigue, suggesting it may involve more demanding muscle activation patterns or sustained contraction
- **Gesture 3** is the least fatiguing, potentially involving less muscular effort or allowing for micro-rest periods
- **Gesture 5** starts with slightly higher initial fatigue (0.51 vs ~0.40 for others), suggesting immediate perceived effort

---

## Participant Variability

### Inter-Participant Differences

| Metric | Value |
|--------|-------|
| Lowest participant mean | 0.03 |
| Highest participant mean | 8.01 |
| Range | 7.98 points |
| Inter-participant Std Dev | 2.26 |

### Participant Clusters

Based on mean fatigue scores, participants can be grouped into:

| Category | Mean Score Range | Approx. % of Participants |
|----------|------------------|---------------------------|
| Low Fatigue Reporters | 0-2 | ~25% |
| Moderate Fatigue Reporters | 2-5 | ~40% |
| High Fatigue Reporters | 5-8+ | ~35% |

### Notable Observations

1. **High variability:** Some participants report almost no fatigue (mean ~0) while others consistently report high fatigue (mean ~8)
2. **Possible factors:**
   - Individual fitness levels
   - Pain/fatigue tolerance differences
   - Interpretation of the Borg scale
   - Muscle strength and endurance capacity

---

## Correlation Analysis

### Temporal Correlation Structure

| Correlation Type | Value | Interpretation |
|------------------|-------|----------------|
| Mean adjacent segment correlation | **0.960** | Very high |
| Early-late phase correlation | **0.391** | Moderate |

### Key Findings

1. **High temporal autocorrelation (0.96):** Adjacent segments are highly correlated, indicating smooth fatigue progression without sudden jumps. This validates the reliability of self-reported measurements.

2. **Moderate early-late correlation (0.39):** Participants who report higher fatigue early tend to report higher fatigue late, but the relationship is not deterministic. This suggests:
   - Individual fatigue trajectories vary
   - Early fatigue is a moderate predictor of final fatigue
   - Other factors influence late-stage fatigue

### Correlation Matrix Insights

The segment correlation matrix (see `segment_correlation.png`) shows:
- Strong positive correlations throughout (all segments positively correlated)
- Correlation strength decreases with temporal distance
- Block structure visible corresponding to session phases

---

## Outlier Analysis

### Identified Outliers

| Type | Count | Details |
|------|-------|---------|
| Values > 10 (exceeding Borg scale) | 1 | Value of 17 at Segment 51, Gesture 1 |
| Values = 0 at session end | Multiple | Some participants report 0 fatigue throughout |

### Outlier Details

**Value of 17 (Row 75, Segment 51):**
- Participant: P17
- Gesture: Gesture 1
- Likely cause: Data entry error (possibly meant to enter "7" or "1")
- Recommendation: Verify with source data or treat as missing

### Unusual Patterns

1. **Zero-fatigue participants:** Several participants (e.g., P22, P23) report 0 or near-0 fatigue for entire sessions. This could indicate:
   - Very high fatigue tolerance
   - Misunderstanding of the scale
   - Lack of engagement with the task

2. **Highly variable participants:** P9 shows erratic fatigue patterns (values jumping between 0-9 within short periods), suggesting either genuine variability or inconsistent reporting.

---

## Key Findings & Insights

### 1. Clear Fatigue Progression
✓ Fatigue increases monotonically from 0.08 to 5.55 over 20 minutes  
✓ Steepest increase in first 10 minutes, then plateaus  
✓ Pattern consistent with physiological fatigue accumulation

### 2. Gesture-Specific Effects
✓ Gesture 2 is most fatiguing (final mean: 6.14)  
✓ Gesture 3 is least fatiguing (final mean: 4.98)  
✓ ~1.2 point difference between most and least fatiguing gestures

### 3. High Individual Variability
✓ Participant means range from 0.03 to 8.01  
✓ Inter-participant std dev: 2.26  
✓ Suggests need for individual baseline normalization in modeling

### 4. Excellent Data Quality
✓ Only 0.016% missing data  
✓ Single outlier identified (value of 17)  
✓ High temporal autocorrelation validates measurement reliability

### 5. Ceiling Effects
✓ Late-phase distribution shows slight left skew  
✓ Many participants reach scores of 9-10 by session end  
✓ Borg CR-10 scale may limit discrimination at high fatigue levels

---

## Recommendations

### For Data Preprocessing

1. **Handle the outlier:** Replace the value of 17 with NaN or impute using neighboring values
2. **Forward-fill participant IDs:** Ensure each row has a valid participant identifier
3. **Consider normalization:** Given high inter-participant variability, z-score normalization within participants may improve model performance

### For Modeling

1. **Time-series approach:** The high temporal autocorrelation suggests LSTM or other sequence models would be appropriate
2. **Gesture as feature:** Include gesture type as a categorical feature given significant gesture effects
3. **Participant random effects:** Consider mixed-effects models to account for individual differences
4. **Phase-based features:** Early, middle, and late phase means could serve as summary features

### For Future Data Collection

1. **Verify scale understanding:** Ensure participants understand the Borg CR-10 scale before sessions
2. **Add rest periods:** Consider adding brief rest periods to study fatigue recovery
3. **Collect additional metadata:** Age, fitness level, and hand dominance could explain variability
4. **Extended sessions:** 20 minutes may not capture full fatigue plateau for all participants

---

## Appendix: Generated Visualizations

All visualizations are saved in the `eda_outputs/` directory:

### Main Plots

| Filename | Description |
|----------|-------------|
| `overall_progression.png` | Mean fatigue score across all 60 segments |
| `progression_with_ci.png` | Fatigue progression with 95% confidence interval |
| `gesture_comparison.png` | All 5 gestures overlaid on single plot |
| `gesture_phase_heatmap.png` | Heatmap of mean fatigue by gesture and phase |
| `phase_distributions.png` | Histograms of fatigue scores by session phase |
| `fatigue_rate.png` | Rate of change between consecutive segments |
| `participant_variability.png` | Bar chart of participant mean scores |
| `segment_correlation.png` | Correlation matrix of all 60 segments |

### Per-Gesture Plots

| Filename | Description |
|----------|-------------|
| `gesture_progression_Gesture_1.png` | Fatigue curve for Gesture 1 |
| `gesture_progression_Gesture_2.png` | Fatigue curve for Gesture 2 |
| `gesture_progression_Gesture_3.png` | Fatigue curve for Gesture 3 |
| `gesture_progression_Gesture_4.png` | Fatigue curve for Gesture 4 |
| `gesture_progression_Gesture_5.png` | Fatigue curve for Gesture 5 |

### Per-Segment Plots (First 12 Segments)

| Type | Files |
|------|-------|
| Histograms | `hist_Segment_*.png` |
| Boxplots | `box_Segment_*.png` |

### Data Files

| Filename | Description |
|----------|-------------|
| `segment_statistics.csv` | Count, mean, median, std, min, max per segment |
| `detailed_statistics.csv` | Summary metrics from detailed analysis |

---

## Conclusion

This EDA reveals a well-structured dataset capturing the progression of subjective fatigue during sustained hand gesture performance. The clear temporal patterns, gesture-specific effects, and high data quality make this dataset suitable for:

- **Fatigue prediction models** using time-series approaches
- **Gesture classification** based on fatigue signatures
- **Individual difference studies** examining fatigue tolerance
- **Ergonomic research** on gesture design for reduced fatigue

The primary challenges for modeling will be handling the high inter-participant variability and the ceiling effects observed in late-session measurements.

---