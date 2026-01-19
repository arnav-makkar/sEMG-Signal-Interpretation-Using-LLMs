#!/usr/bin/env python3
"""
Feature Extraction for EMG-Based Fatigue Prediction

This script processes raw EMG data and extracts time-domain and frequency-domain
features for each 20-second segment across 8 sensors.

Input:  Gesture_1_filtered.csv (8 sensors × 2 columns [EMG mV, Time s])
Output: features.csv (60 segments × 32 features)
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
INPUT_FILE = 'Gesture_1_filtered.csv'
OUTPUT_FILE = 'features.csv'
NUM_SENSORS = 8
SEGMENT_DURATION = 20  # seconds
NUM_SEGMENTS = 60
SESSION_DURATION = 1200  # 20 minutes in seconds

print("=" * 60)
print("EMG FEATURE EXTRACTION PIPELINE")
print("=" * 60)

# ===============================
# LOAD DATA
# ===============================
print(f"\n[1/5] Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

# Identify sensor columns
sensor_data = {}
for sensor_id in range(1, NUM_SENSORS + 1):
    emg_col = [c for c in df.columns if f'Sensor {sensor_id}' in c and 'EMG 1 (mV)' in c][0]
    time_col = [c for c in df.columns if f'Sensor {sensor_id}' in c and 'Time Series (s)' in c][0]
    sensor_data[sensor_id] = {
        'emg': df[emg_col].values,
        'time': df[time_col].values
    }
    print(f"  Sensor {sensor_id}: {emg_col[:30]}... ({len(df[emg_col].dropna())} samples)")

# ===============================
# FEATURE EXTRACTION FUNCTIONS
# ===============================

def compute_rms(signal_segment):
    """Root Mean Square - measures signal energy/amplitude"""
    return np.sqrt(np.mean(signal_segment ** 2))

def compute_mav(signal_segment):
    """Mean Absolute Value - average signal magnitude"""
    return np.mean(np.abs(signal_segment))

def compute_mean_frequency(signal_segment, sampling_rate):
    """
    Mean Frequency via FFT
    Fatigue typically causes shift to lower frequencies
    """
    # Remove DC component
    signal_segment = signal_segment - np.mean(signal_segment)
    
    # Compute FFT
    fft_vals = np.fft.rfft(signal_segment)
    fft_freq = np.fft.rfftfreq(len(signal_segment), 1/sampling_rate)
    
    # Power spectrum
    power = np.abs(fft_vals) ** 2
    
    # Mean frequency (weighted average)
    if np.sum(power) == 0:
        return 0
    mean_freq = np.sum(fft_freq * power) / np.sum(power)
    return mean_freq

def compute_trend(signal_segment, timestamps):
    """
    Linear regression slope - rate of signal change
    Positive = increasing activation, Negative = decreasing
    """
    if len(signal_segment) < 2:
        return 0
    slope, _, _, _, _ = linregress(timestamps, signal_segment)
    return slope

# ===============================
# SEGMENT DATA
# ===============================
print(f"\n[2/5] Segmenting into {NUM_SEGMENTS} windows ({SEGMENT_DURATION}s each)...")

features = []
feature_names = []

# Generate feature column names
for sensor_id in range(1, NUM_SENSORS + 1):
    feature_names.extend([
        f'S{sensor_id}_RMS',
        f'S{sensor_id}_MeanFreq',
        f'S{sensor_id}_MAV',
        f'S{sensor_id}_Trend'
    ])

for seg_idx in range(NUM_SEGMENTS):
    start_time = seg_idx * SEGMENT_DURATION
    end_time = (seg_idx + 1) * SEGMENT_DURATION
    
    segment_features = []
    
    for sensor_id in range(1, NUM_SENSORS + 1):
        # Extract time window for this sensor
        times = sensor_data[sensor_id]['time']
        emg_vals = sensor_data[sensor_id]['emg']
        
        # Find indices within time window
        mask = (times >= start_time) & (times < end_time)
        segment_emg = emg_vals[mask]
        segment_time = times[mask]
        
        if len(segment_emg) == 0:
            # No data in this segment - use zeros
            segment_features.extend([0, 0, 0, 0])
            continue
        
        # Compute sampling rate
        if len(segment_time) > 1:
            sampling_rate = 1 / np.median(np.diff(segment_time))
        else:
            sampling_rate = 1000  # Default assumption
        
        # Extract features
        rms = compute_rms(segment_emg)
        mean_freq = compute_mean_frequency(segment_emg, sampling_rate)
        mav = compute_mav(segment_emg)
        trend = compute_trend(segment_emg, segment_time)
        
        segment_features.extend([rms, mean_freq, mav, trend])
    
    features.append(segment_features)
    
    if (seg_idx + 1) % 10 == 0:
        print(f"  Processed segments {seg_idx - 8}-{seg_idx + 1}")

print(f"  Total segments: {len(features)}")

# ===============================
# CREATE DATAFRAME
# ===============================
print(f"\n[3/5] Creating feature matrix...")
features_df = pd.DataFrame(features, columns=feature_names)
features_df.insert(0, 'Segment', range(1, NUM_SEGMENTS + 1))
features_df.insert(1, 'TimeStart', [i * SEGMENT_DURATION for i in range(NUM_SEGMENTS)])
features_df.insert(2, 'TimeEnd', [(i + 1) * SEGMENT_DURATION for i in range(NUM_SEGMENTS)])

print(f"  Shape: {features_df.shape[0]} segments × {features_df.shape[1]} columns")

# ===============================
# STATISTICS
# ===============================
print(f"\n[4/5] Feature statistics:")
print(f"\n  RMS Summary (Sensor 1):")
print(f"    Min:  {features_df['S1_RMS'].min():.6f}")
print(f"    Max:  {features_df['S1_RMS'].max():.6f}")
print(f"    Mean: {features_df['S1_RMS'].mean():.6f}")

print(f"\n  Mean Frequency Summary (Sensor 1):")
print(f"    Min:  {features_df['S1_MeanFreq'].min():.2f} Hz")
print(f"    Max:  {features_df['S1_MeanFreq'].max():.2f} Hz")
print(f"    Mean: {features_df['S1_MeanFreq'].mean():.2f} Hz")

# ===============================
# SAVE OUTPUT
# ===============================
print(f"\n[5/5] Saving to {OUTPUT_FILE}...")
features_df.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved successfully!")

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT_FILE}")
print(f"Shape:  {features_df.shape[0]} rows × {features_df.shape[1]} columns")
print(f"\nPreview:")
print(features_df.head(3).to_string())
