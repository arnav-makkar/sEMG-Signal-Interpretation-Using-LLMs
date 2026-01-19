#!/usr/bin/env python3
"""
Feature Extraction for Multiple Participants

Extracts RMS, MeanFreq, MAV, Trend features for all participants.
Outputs: DATA/{P}/features.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
DATA_DIR = 'DATA'
NUM_SENSORS = 8
SEGMENT_DURATION = 20  # seconds
NUM_SEGMENTS = 60

# ===============================
# FEATURE FUNCTIONS
# ===============================
def compute_rms(signal_segment):
    return np.sqrt(np.mean(signal_segment ** 2))

def compute_mav(signal_segment):
    return np.mean(np.abs(signal_segment))

def compute_mean_frequency(signal_segment, sampling_rate):
    signal_segment = signal_segment - np.mean(signal_segment)
    fft_vals = np.fft.rfft(signal_segment)
    fft_freq = np.fft.rfftfreq(len(signal_segment), 1/sampling_rate)
    power = np.abs(fft_vals) ** 2
    if np.sum(power) == 0:
        return 0
    return np.sum(fft_freq * power) / np.sum(power)

def compute_trend(signal_segment, timestamps):
    if len(signal_segment) < 2:
        return 0
    slope, _, _, _, _ = linregress(timestamps, signal_segment)
    return slope

# ===============================
# MAIN EXTRACTION
# ===============================
def extract_features(participant):
    """Extract features for a single participant."""
    input_file = os.path.join(DATA_DIR, participant, 'Gesture_1_filtered.csv')
    output_file = os.path.join(DATA_DIR, participant, 'features.csv')
    
    print(f"\n{'='*60}")
    print(f"Processing {participant}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_file):
        print(f"  ⚠️ {input_file} not found. Run preprocessing first.")
        return False
    
    # Load data
    print(f"  Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows")
    
    # Identify sensor columns
    sensor_data = {}
    for sensor_id in range(1, NUM_SENSORS + 1):
        emg_cols = [c for c in df.columns if f'Sensor {sensor_id}' in c and 'EMG 1 (mV)' in c]
        time_cols = [c for c in df.columns if f'Sensor {sensor_id}' in c and 'Time Series (s)' in c]
        
        if not emg_cols or not time_cols:
            print(f"  ⚠️ Sensor {sensor_id} columns not found")
            continue
            
        sensor_data[sensor_id] = {
            'emg': df[emg_cols[0]].values,
            'time': df[time_cols[0]].values
        }
    
    if len(sensor_data) < NUM_SENSORS:
        print(f"  ⚠️ Only {len(sensor_data)} sensors found")
    
    # Feature column names
    feature_names = []
    for sensor_id in range(1, NUM_SENSORS + 1):
        feature_names.extend([
            f'S{sensor_id}_RMS',
            f'S{sensor_id}_MeanFreq',
            f'S{sensor_id}_MAV',
            f'S{sensor_id}_Trend'
        ])
    
    # Extract features per segment
    print(f"  Extracting features for {NUM_SEGMENTS} segments...")
    features = []
    
    for seg_idx in range(NUM_SEGMENTS):
        start_time = seg_idx * SEGMENT_DURATION
        end_time = (seg_idx + 1) * SEGMENT_DURATION
        
        segment_features = []
        
        for sensor_id in range(1, NUM_SENSORS + 1):
            if sensor_id not in sensor_data:
                segment_features.extend([0, 0, 0, 0])
                continue
                
            times = sensor_data[sensor_id]['time']
            emg_vals = sensor_data[sensor_id]['emg']
            
            mask = (times >= start_time) & (times < end_time)
            segment_emg = emg_vals[mask]
            segment_time = times[mask]
            
            if len(segment_emg) == 0:
                segment_features.extend([0, 0, 0, 0])
                continue
            
            # Sampling rate
            if len(segment_time) > 1:
                sampling_rate = 1 / np.median(np.diff(segment_time))
            else:
                sampling_rate = 1000
            
            # Compute features
            rms = compute_rms(segment_emg)
            mean_freq = compute_mean_frequency(segment_emg, sampling_rate)
            mav = compute_mav(segment_emg)
            trend = compute_trend(segment_emg, segment_time)
            
            segment_features.extend([rms, mean_freq, mav, trend])
        
        features.append(segment_features)
    
    # Create DataFrame
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df.insert(0, 'Segment', range(1, NUM_SEGMENTS + 1))
    features_df.insert(1, 'TimeStart', [i * SEGMENT_DURATION for i in range(NUM_SEGMENTS)])
    features_df.insert(2, 'TimeEnd', [(i + 1) * SEGMENT_DURATION for i in range(NUM_SEGMENTS)])
    
    # Save
    features_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file} ({features_df.shape[0]} rows × {features_df.shape[1]} cols)")
    
    return True

def main():
    print("=" * 60)
    print("FEATURE EXTRACTION - ALL PARTICIPANTS")
    print("=" * 60)
    
    # Allow single participant or all
    if len(sys.argv) > 1:
        participants = [sys.argv[1]]
    else:
        participants = PARTICIPANTS
    
    success = 0
    for p in participants:
        if extract_features(p):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success}/{len(participants)} participants")
    print("=" * 60)

if __name__ == "__main__":
    main()
