#!/usr/bin/env python3
"""
Complete EDA script for Subjective EMG Fatigue Dataset

This script performs:
1. Dataset loading and basic inspection
2. Column classification (ID, gesture, segments)
3. Missing value analysis
4. Descriptive statistics
5. Segment-wise progression analysis
6. Gesture-wise fatigue trends
7. Outlier and range checks
8. Visualization (histograms, boxplots, progression curves)

Outputs:
- Printed summaries
- Saved plots (PNG)
- Optional CSV summaries

Author: EDA pipeline
"""

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
CSV_PATH = "data.csv"
OUTPUT_DIR = "eda_outputs"
MAX_NUMERIC_PLOTS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH, low_memory=False)

print("\n=== BASIC INFO ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())

# ===============================
# IDENTIFY KEY COLUMNS
# ===============================
pid_col = None
gesture_col = None

for c in df.columns:
    cl = c.lower()
    if cl in ["p id", "pid"] or "participant" in cl or "subject" in cl:
        pid_col = c
    if "gesture" in cl:
        gesture_col = c

segment_cols = [c for c in df.columns if "segment" in c.lower() or c.lower().startswith("seg")]

def extract_segment_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 999

segment_cols = sorted(segment_cols, key=extract_segment_number)

print("\nParticipant column:", pid_col)
print("Gesture column:", gesture_col)
print("Number of segment columns:", len(segment_cols))

# ===============================
# BASIC COUNTS
# ===============================
if pid_col:
    print("Unique participants:", df[pid_col].nunique())

if gesture_col:
    print("Unique gestures:", df[gesture_col].nunique())
    print("\nGesture counts:")
    print(df[gesture_col].value_counts())

# ===============================
# SEGMENT DATAFRAME
# ===============================
seg_df = df[segment_cols].apply(pd.to_numeric, errors="coerce")

# ===============================
# MISSING VALUE ANALYSIS
# ===============================
missing_counts = seg_df.isnull().sum()
missing_pct = (seg_df.isnull().mean() * 100).round(2)

print("\n=== MISSING VALUE SUMMARY ===")
print("Segments with any missing values:", (missing_counts > 0).sum())
print("Median non-null percentage:", (100 - missing_pct).median())

# ===============================
# RANGE CHECK
# ===============================
global_min = seg_df.min().min()
global_max = seg_df.max().max()

print("\n=== VALUE RANGE CHECK ===")
print("Global min:", global_min)
print("Global max:", global_max)

if global_max > 10:
    print("WARNING: Values above expected Borg CR range (0–10) detected")

# ===============================
# DESCRIPTIVE STATISTICS
# ===============================
seg_stats = seg_df.agg(
    ["count", "mean", "median", "std", "min", "max"]
).transpose()

seg_stats.to_csv(os.path.join(OUTPUT_DIR, "segment_statistics.csv"))
print("\nSegment-level statistics saved")

# ===============================
# OVERALL PROGRESSION PLOT
# ===============================
overall_mean = seg_df.mean()

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(overall_mean) + 1), overall_mean, marker="o", linewidth=1)
plt.xlabel("Segment index (1–60)")
plt.ylabel("Mean subjective score")
plt.title("Overall fatigue progression across session")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overall_progression.png"))
plt.close()

# ===============================
# GESTURE-WISE PROGRESSION
# ===============================
if gesture_col:
    for g in df[gesture_col].unique():
        sub = df[df[gesture_col] == g]
        sub_seg = sub[segment_cols].apply(pd.to_numeric, errors="coerce")
        mean_sub = sub_seg.mean()

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(mean_sub) + 1), mean_sub, marker="o", linewidth=1)
        plt.xlabel("Segment index")
        plt.ylabel("Mean subjective score")
        plt.title(f"Fatigue progression for gesture: {g}")
        plt.tight_layout()
        fname = f"gesture_progression_{str(g).replace(' ', '_')}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close()

# ===============================
# HISTOGRAMS AND BOXPLOTS
# ===============================
numeric_cols = seg_df.columns[:MAX_NUMERIC_PLOTS]

for col in numeric_cols:
    data = seg_df[col].dropna()

    plt.figure(figsize=(8, 3))
    plt.hist(data, bins=40)
    plt.title(f"Histogram: {col}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{col}.png"))
    plt.close()

    plt.figure(figsize=(8, 2.5))
    plt.boxplot(data, vert=False)
    plt.title(f"Boxplot: {col}")
    plt.xlabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"box_{col}.png"))
    plt.close()

# ===============================
# CORRELATION MATRIX
# ===============================
if seg_df.shape[1] >= 2:
    corr = seg_df.corr()

    plt.figure(figsize=(6, 6))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title("Correlation matrix of segment scores")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "segment_correlation.png"))
    plt.close()

# ===============================
# FINAL SUMMARY
# ===============================
print("\n=== FINAL SUMMARY ===")
print("Total rows:", df.shape[0])
print("Participants:", df[pid_col].nunique() if pid_col else "NA")
print("Gestures:", df[gesture_col].nunique() if gesture_col else "NA")
print("Segments per session:", len(segment_cols))
print("EDA outputs saved to:", OUTPUT_DIR)
