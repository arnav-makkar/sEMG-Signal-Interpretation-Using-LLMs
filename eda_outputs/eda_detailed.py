#!/usr/bin/env python3
"""
Detailed EDA script for Subjective EMG Fatigue Dataset
Extended analysis with comprehensive statistics and visualizations
"""

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ===============================
# CONFIG
# ===============================
CSV_PATH = "data.csv"
OUTPUT_DIR = "eda_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH, low_memory=False)

# Identify columns
pid_col = "P ID"
gesture_col = "Gesture"
segment_cols = [c for c in df.columns if "segment" in c.lower() or c.lower().startswith("seg")]

def extract_segment_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 999

segment_cols = sorted(segment_cols, key=extract_segment_number)

# Create segment dataframe
seg_df = df[segment_cols].apply(pd.to_numeric, errors="coerce")

# ===============================
# DETAILED ANALYSIS
# ===============================
results = {}

# 1. Basic Dataset Info
results['total_rows'] = df.shape[0]
results['total_columns'] = df.shape[1]
results['n_participants'] = df[pid_col].nunique()
results['n_gestures'] = df[gesture_col].nunique()
results['n_segments'] = len(segment_cols)
results['session_duration_minutes'] = 20

# 2. Gesture Distribution
gesture_counts = df[gesture_col].value_counts()
results['gesture_distribution'] = gesture_counts.to_dict()

# 3. Value Range Analysis
results['global_min'] = seg_df.min().min()
results['global_max'] = seg_df.max().max()
results['global_mean'] = seg_df.mean().mean()
results['global_std'] = seg_df.std().mean()

# 4. Missing Value Analysis
missing_counts = seg_df.isnull().sum()
results['segments_with_missing'] = (missing_counts > 0).sum()
results['total_missing_values'] = missing_counts.sum()
results['missing_percentage'] = (seg_df.isnull().sum().sum() / seg_df.size * 100)

# 5. Outlier Detection (values > 10 for Borg scale)
outlier_mask = seg_df > 10
results['outlier_count'] = outlier_mask.sum().sum()
outlier_locations = []
for col in seg_df.columns:
    for idx in seg_df[seg_df[col] > 10].index:
        outlier_locations.append({
            'row': idx,
            'segment': col,
            'value': seg_df.loc[idx, col],
            'gesture': df.loc[idx, gesture_col] if idx < len(df) else 'Unknown'
        })
results['outlier_locations'] = outlier_locations

# 6. Fatigue Progression Analysis
overall_mean = seg_df.mean()
overall_std = seg_df.std()

# Calculate rate of fatigue increase
early_mean = seg_df[segment_cols[:10]].mean().mean()
mid_mean = seg_df[segment_cols[20:40]].mean().mean()
late_mean = seg_df[segment_cols[50:]].mean().mean()

results['early_phase_mean'] = early_mean
results['mid_phase_mean'] = mid_mean
results['late_phase_mean'] = late_mean
results['fatigue_increase_early_to_late'] = late_mean - early_mean

# 7. Gesture-wise Analysis
gesture_stats = {}
for g in df[gesture_col].unique():
    sub = df[df[gesture_col] == g]
    sub_seg = sub[segment_cols].apply(pd.to_numeric, errors="coerce")
    gesture_stats[g] = {
        'mean': sub_seg.mean().mean(),
        'std': sub_seg.std().mean(),
        'max': sub_seg.max().max(),
        'final_mean': sub_seg[segment_cols[-5:]].mean().mean(),
        'initial_mean': sub_seg[segment_cols[:5]].mean().mean(),
        'fatigue_delta': sub_seg[segment_cols[-5:]].mean().mean() - sub_seg[segment_cols[:5]].mean().mean()
    }
results['gesture_stats'] = gesture_stats

# 8. Participant Variability Analysis
participant_means = []
participant_maxes = []
df_filled = df[pid_col].ffill()
for p in df_filled.unique():
    sub = df[df_filled == p]
    sub_seg = sub[segment_cols].apply(pd.to_numeric, errors="coerce")
    participant_means.append(sub_seg.mean().mean())
    participant_maxes.append(sub_seg.max().max())

results['participant_mean_range'] = (min(participant_means), max(participant_means))
results['participant_variability_std'] = np.std(participant_means)

# 9. Correlation Analysis
corr_matrix = seg_df.corr()
# Adjacent segment correlation
adjacent_corrs = []
for i in range(len(segment_cols) - 1):
    adjacent_corrs.append(corr_matrix.iloc[i, i+1])
results['mean_adjacent_correlation'] = np.mean(adjacent_corrs)

# Early-late correlation
early_cols = segment_cols[:10]
late_cols = segment_cols[-10:]
early_late_corr = seg_df[early_cols].mean(axis=1).corr(seg_df[late_cols].mean(axis=1))
results['early_late_correlation'] = early_late_corr

# 10. Distribution Analysis per Phase
phases = {
    'early': segment_cols[:20],
    'middle': segment_cols[20:40],
    'late': segment_cols[40:]
}

phase_distributions = {}
for phase_name, cols in phases.items():
    phase_data = seg_df[cols].values.flatten()
    phase_data = phase_data[~np.isnan(phase_data)]
    phase_distributions[phase_name] = {
        'mean': np.mean(phase_data),
        'median': np.median(phase_data),
        'std': np.std(phase_data),
        'skewness': stats.skew(phase_data),
        'kurtosis': stats.kurtosis(phase_data),
        'q25': np.percentile(phase_data, 25),
        'q75': np.percentile(phase_data, 75)
    }
results['phase_distributions'] = phase_distributions

# ===============================
# ADDITIONAL VISUALIZATIONS
# ===============================

# 1. Gesture Comparison Plot
fig, ax = plt.subplots(figsize=(12, 6))
for g in sorted(df[gesture_col].unique()):
    sub = df[df[gesture_col] == g]
    sub_seg = sub[segment_cols].apply(pd.to_numeric, errors="coerce")
    mean_vals = sub_seg.mean()
    ax.plot(range(1, len(mean_vals) + 1), mean_vals, marker='', linewidth=2, label=g)

ax.set_xlabel("Segment Index (1-60)")
ax.set_ylabel("Mean Subjective Fatigue Score")
ax.set_title("Fatigue Progression Comparison Across All Gestures")
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gesture_comparison.png"), dpi=150)
plt.close()

# 2. Phase Distribution Boxplot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, (phase_name, cols) in enumerate(phases.items()):
    phase_data = seg_df[cols].values.flatten()
    phase_data = phase_data[~np.isnan(phase_data)]
    axes[idx].hist(phase_data, bins=11, range=(-0.5, 10.5), edgecolor='black', alpha=0.7)
    axes[idx].set_title(f"{phase_name.capitalize()} Phase (Segments {idx*20+1}-{(idx+1)*20})")
    axes[idx].set_xlabel("Fatigue Score")
    axes[idx].set_ylabel("Frequency")
    axes[idx].axvline(np.mean(phase_data), color='red', linestyle='--', label=f'Mean: {np.mean(phase_data):.2f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase_distributions.png"), dpi=150)
plt.close()

# 3. Heatmap of Mean Fatigue by Gesture and Time Phase
gesture_phase_matrix = []
gesture_names = sorted(df[gesture_col].unique())
for g in gesture_names:
    sub = df[df[gesture_col] == g]
    sub_seg = sub[segment_cols].apply(pd.to_numeric, errors="coerce")
    row = []
    for phase_name, cols in phases.items():
        row.append(sub_seg[cols].mean().mean())
    gesture_phase_matrix.append(row)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(gesture_phase_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Early\n(0-7 min)', 'Middle\n(7-13 min)', 'Late\n(13-20 min)'])
ax.set_yticks(range(len(gesture_names)))
ax.set_yticklabels(gesture_names)
ax.set_title("Mean Fatigue Score by Gesture and Session Phase")
plt.colorbar(im, label='Mean Score')
for i in range(len(gesture_names)):
    for j in range(3):
        ax.text(j, i, f'{gesture_phase_matrix[i][j]:.1f}', ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gesture_phase_heatmap.png"), dpi=150)
plt.close()

# 4. Fatigue Rate Plot (derivative)
fatigue_rate = overall_mean.diff()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(2, len(fatigue_rate) + 1), fatigue_rate[1:], color='steelblue', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel("Segment Index")
ax.set_ylabel("Change in Mean Fatigue Score")
ax.set_title("Rate of Fatigue Change Between Consecutive Segments")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fatigue_rate.png"), dpi=150)
plt.close()

# 5. Participant Variability Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(participant_means)), sorted(participant_means), color='teal', alpha=0.7)
ax.axhline(np.mean(participant_means), color='red', linestyle='--', label=f'Overall Mean: {np.mean(participant_means):.2f}')
ax.set_xlabel("Participant (sorted by mean fatigue)")
ax.set_ylabel("Mean Fatigue Score")
ax.set_title("Participant Variability in Mean Fatigue Scores")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "participant_variability.png"), dpi=150)
plt.close()

# 6. Confidence Interval Plot
fig, ax = plt.subplots(figsize=(12, 5))
x = range(1, len(overall_mean) + 1)
ci = 1.96 * overall_std / np.sqrt(205)
ax.fill_between(x, overall_mean - ci, overall_mean + ci, alpha=0.3, color='blue', label='95% CI')
ax.plot(x, overall_mean, 'b-', linewidth=2, label='Mean')
ax.set_xlabel("Segment Index (1-60)")
ax.set_ylabel("Subjective Fatigue Score")
ax.set_title("Overall Fatigue Progression with 95% Confidence Interval")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "progression_with_ci.png"), dpi=150)
plt.close()

# ===============================
# PRINT SUMMARY
# ===============================
print("\n" + "="*60)
print("DETAILED EDA RESULTS")
print("="*60)

print(f"\n--- DATASET OVERVIEW ---")
print(f"Total Observations: {results['total_rows']}")
print(f"Participants: {results['n_participants']}")
print(f"Gestures: {results['n_gestures']}")
print(f"Segments per session: {results['n_segments']}")
print(f"Session Duration: {results['session_duration_minutes']} minutes")

print(f"\n--- VALUE STATISTICS ---")
print(f"Global Min: {results['global_min']}")
print(f"Global Max: {results['global_max']}")
print(f"Global Mean: {results['global_mean']:.3f}")
print(f"Global Std: {results['global_std']:.3f}")

print(f"\n--- MISSING VALUES ---")
print(f"Segments with missing: {results['segments_with_missing']}")
print(f"Total missing values: {results['total_missing_values']}")
print(f"Missing percentage: {results['missing_percentage']:.4f}%")

print(f"\n--- OUTLIERS (>10) ---")
print(f"Outlier count: {results['outlier_count']}")
if results['outlier_locations']:
    print("Outlier details:")
    for o in results['outlier_locations'][:10]:
        print(f"  Row {o['row']}, {o['segment']}: {o['value']} ({o['gesture']})")

print(f"\n--- FATIGUE PROGRESSION ---")
print(f"Early phase mean (0-7 min): {results['early_phase_mean']:.3f}")
print(f"Mid phase mean (7-13 min): {results['mid_phase_mean']:.3f}")
print(f"Late phase mean (13-20 min): {results['late_phase_mean']:.3f}")
print(f"Total fatigue increase: {results['fatigue_increase_early_to_late']:.3f}")

print(f"\n--- GESTURE ANALYSIS ---")
for g, stats in sorted(results['gesture_stats'].items()):
    print(f"{g}: mean={stats['mean']:.2f}, final={stats['final_mean']:.2f}, delta={stats['fatigue_delta']:.2f}")

print(f"\n--- CORRELATION ANALYSIS ---")
print(f"Mean adjacent segment correlation: {results['mean_adjacent_correlation']:.3f}")
print(f"Early-late phase correlation: {results['early_late_correlation']:.3f}")

print(f"\n--- PHASE DISTRIBUTIONS ---")
for phase, dist in results['phase_distributions'].items():
    print(f"{phase.capitalize()}: mean={dist['mean']:.2f}, median={dist['median']:.2f}, skew={dist['skewness']:.2f}")

print(f"\n--- PARTICIPANT VARIABILITY ---")
print(f"Mean fatigue range: {results['participant_mean_range'][0]:.2f} - {results['participant_mean_range'][1]:.2f}")
print(f"Inter-participant std: {results['participant_variability_std']:.3f}")

print("\n" + "="*60)
print("Additional visualizations saved to:", OUTPUT_DIR)
print("="*60)

# Save detailed results to CSV
detailed_stats = pd.DataFrame({
    'Metric': [
        'Total Observations', 'Participants', 'Gestures', 'Segments',
        'Global Min', 'Global Max', 'Global Mean', 'Global Std',
        'Missing Values', 'Outliers (>10)',
        'Early Phase Mean', 'Mid Phase Mean', 'Late Phase Mean',
        'Fatigue Increase', 'Adjacent Correlation', 'Early-Late Correlation'
    ],
    'Value': [
        results['total_rows'], results['n_participants'], results['n_gestures'], results['n_segments'],
        results['global_min'], results['global_max'], round(results['global_mean'], 3), round(results['global_std'], 3),
        results['total_missing_values'], results['outlier_count'],
        round(results['early_phase_mean'], 3), round(results['mid_phase_mean'], 3), round(results['late_phase_mean'], 3),
        round(results['fatigue_increase_early_to_late'], 3), round(results['mean_adjacent_correlation'], 3), round(results['early_late_correlation'], 3)
    ]
})
detailed_stats.to_csv(os.path.join(OUTPUT_DIR, "detailed_statistics.csv"), index=False)
print("\nDetailed statistics saved to detailed_statistics.csv")
