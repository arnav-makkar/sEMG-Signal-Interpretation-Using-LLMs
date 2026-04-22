#!/usr/bin/env python3
"""
Full Activation Feature Extraction from Raw EMG Signal
=======================================================
Processes DATA/{P}/Gesture_1.csv for each participant to extract
comprehensive features for muscle activation analysis.

Auto-detects sensor IDs from CSV headers (handles different device
serial numbers across participants).

Input:  DATA/{P}/Gesture_1.csv  (raw 8-sensor EMG, ~1259 Hz, ~20 min)
Output: DATA/{P}/features_long.csv         (60 segments × 12 features × 8 sensors + cross-channel)
        DATA/{P}/features_short.csv         (~4800 windows × envelope per sensor)
        DATA/{P}/activation_features.json   (session summary for LLM prompt)

Usage:
    python3 scripts/extract_activation_features_full.py          # all participants
    python3 scripts/extract_activation_features_full.py P1       # single participant
    python3 scripts/extract_activation_features_full.py P1 P3 P7 # specific participants
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress, pearsonr
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIG
# ==============================

PARTICIPANTS = [f'P{i}' for i in range(1, 11)]  # P1-P10
DATA_DIR = 'DATA'
RAW_FILENAME = 'Gesture_1.csv'
NUM_SENSORS = 8
SEGMENT_DURATION = 20   # seconds per long-window segment
NUM_SEGMENTS = 60
SHORT_WINDOW = 0.250    # seconds per short-window
ONSET_K = 2             # threshold multiplier for onset detection
ONSET_BASELINE_S = 5    # seconds of baseline for onset detection
MIN_BURST_MS = 50       # minimum burst duration (ms)
BANDPASS_LOW = 20       # Hz
BANDPASS_HIGH = 450     # Hz
BANDPASS_ORDER = 4
WAMP_THRESHOLD_MV = 0.015  # fixed WAMP threshold in mV (literature-based)


# ==============================
# AUTO-DETECT SENSOR COLUMNS
# ==============================

def detect_sensor_columns(csv_path):
    """
    Auto-detect EMG and timestamp columns from CSV header.
    Handles different device serial numbers across participants.
    Returns dict: {sensor_num: {'emg_col': str, 'time_col': str}}
    """
    header = pd.read_csv(csv_path, nrows=0)
    columns = header.columns.tolist()

    sensor_map = {}
    for sensor_num in range(1, NUM_SENSORS + 1):
        pattern = f'Sensor {sensor_num}'
        emg_cols = [c for c in columns if pattern in c and 'EMG 1 (mV)' in c and 'Time' not in c]
        time_cols = [c for c in columns if pattern in c and 'EMG 1 Time Series' in c]

        if emg_cols and time_cols:
            sensor_map[sensor_num] = {
                'emg_col': emg_cols[0],
                'time_col': time_cols[0],
            }

    return sensor_map


# ==============================
# SIGNAL PROCESSING
# ==============================

def bandpass_filter(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=BANDPASS_ORDER):
    """Apply Butterworth band-pass filter."""
    nyq = fs / 2
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


# ==============================
# LONG-WINDOW FEATURES (20s segments)
# ==============================

def compute_long_features(segment, fs):
    """Compute 12 features for a single 20s segment of a single sensor."""
    n = len(segment)
    if n < 10:
        return {k: 0 for k in ['RMS', 'MAV', 'IEMG', 'WL', 'ZC', 'SSC',
                                'WAMP', 'PeakAmp', 'Variance', 'MNF', 'MDF', 'TotalPower']}

    features = {}

    # Time-domain
    features['RMS'] = float(np.sqrt(np.mean(segment ** 2)))
    features['MAV'] = float(np.mean(np.abs(segment)))
    features['IEMG'] = float(np.sum(np.abs(segment)) / fs)
    features['WL'] = float(np.sum(np.abs(np.diff(segment))))
    features['PeakAmp'] = float(np.max(np.abs(segment)))
    features['Variance'] = float(np.var(segment))

    # Zero crossings (with noise dead-zone)
    dead_zone = features['RMS'] * 0.01
    shifted = segment.copy()
    shifted[np.abs(shifted) < dead_zone] = 0
    signs = np.sign(shifted[shifted != 0])
    features['ZC'] = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0

    # Slope sign changes
    d = np.diff(segment)
    features['SSC'] = int(np.sum(np.diff(np.sign(d)) != 0)) if len(d) > 1 else 0

    # WAMP — fixed threshold (literature-based)
    features['WAMP'] = int(np.sum(np.abs(np.diff(segment)) > WAMP_THRESHOLD_MV))

    # Frequency-domain
    dc_removed = segment - np.mean(segment)
    fft_vals = np.fft.rfft(dc_removed)
    fft_freq = np.fft.rfftfreq(n, 1.0 / fs)
    power = np.abs(fft_vals) ** 2
    total_power = np.sum(power)

    if total_power > 0:
        features['MNF'] = float(np.sum(fft_freq * power) / total_power)
        cumpower = np.cumsum(power)
        mdf_idx = np.searchsorted(cumpower, total_power / 2)
        features['MDF'] = float(fft_freq[min(mdf_idx, len(fft_freq) - 1)])
    else:
        features['MNF'] = 0.0
        features['MDF'] = 0.0
    features['TotalPower'] = float(total_power)

    return features


# ==============================
# SHORT-WINDOW / ONSET DETECTION
# ==============================

def compute_envelope(emg, fs, window_s=SHORT_WINDOW):
    rectified = np.abs(emg)
    window_samples = max(int(window_s * fs), 1)
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(rectified, kernel, mode='same')


def detect_onset_offset(envelope, timestamps, fs):
    """
    Detect activation onset/offset using threshold method.
    Returns (onset_s, offset_s, duty_cycle, burst_count, is_sustained).

    For sustained-contraction data (baseline already active), returns all None
    with is_sustained=True, since onset/offset/duty_cycle are not meaningful.
    """
    baseline_samples = int(ONSET_BASELINE_S * fs)
    if baseline_samples < 10 or len(envelope) < baseline_samples:
        return None, None, None, None, False

    baseline_mean = np.mean(envelope[:baseline_samples])
    baseline_std = np.std(envelope[:baseline_samples])
    overall_mean = np.mean(envelope)

    # Detect sustained contraction: baseline is already active
    # If baseline mean > 50% of overall envelope mean, there's no quiet rest period
    if baseline_mean > 0.5 * overall_mean:
        # Sustained contraction — onset/offset/duty_cycle not meaningful
        return None, None, None, None, True

    # Normal onset detection
    threshold = baseline_mean + ONSET_K * baseline_std
    active = envelope > threshold
    duty_cycle = float(np.sum(active)) / len(active)

    min_samples = max(int(MIN_BURST_MS / 1000 * fs), 1)
    changes = np.diff(active.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if active[0]:
        starts = np.concatenate([[0], starts])
    if active[-1]:
        ends = np.concatenate([ends, [len(active)]])

    valid_bursts = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_samples]

    burst_count = len(valid_bursts)
    onset_s = float(timestamps[valid_bursts[0][0]]) if valid_bursts else None
    offset_s = float(timestamps[valid_bursts[-1][1] - 1]) if valid_bursts else None

    return onset_s, offset_s, duty_cycle, burst_count, False


def compute_short_window_features(emg, timestamps, fs):
    envelope = compute_envelope(emg, fs)
    onset_s, offset_s, duty_cycle, burst_count, is_sustained = detect_onset_offset(envelope, timestamps, fs)

    window_samples = int(SHORT_WINDOW * fs)
    n_windows = len(emg) // window_samples
    records = []
    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        t_start = timestamps[start] if start < len(timestamps) else 0
        t_end = timestamps[min(end - 1, len(timestamps) - 1)] if end <= len(timestamps) else 0
        records.append({
            'Window': w + 1,
            'TimeStart': float(t_start),
            'TimeEnd': float(t_end),
            'Envelope': float(np.mean(envelope[start:end])),
        })

    return records, onset_s, offset_s, duty_cycle, burst_count, is_sustained


# ==============================
# CROSS-CHANNEL FEATURES
# ==============================

def compute_activation_map(rms_array):
    total = rms_array.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1e-9, total)
    return rms_array / total

def compute_concentration_index(act_map):
    return (act_map ** 2).sum(axis=1)

def compute_pattern_stability(act_map):
    stab = []
    for i in range(len(act_map) - 1):
        a, b = act_map[i], act_map[i + 1]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        stab.append(float(np.dot(a, b) / d) if d > 1e-9 else 1.0)
    return np.array(stab)

def compute_inter_channel_corr(rms_array):
    n = rms_array.shape[1]
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(rms_array[:, i]) > 1e-9 and np.std(rms_array[:, j]) > 1e-9:
                r, _ = pearsonr(rms_array[:, i], rms_array[:, j])
                corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0

def compute_co_contraction_ratio(rms_array):
    flexor = rms_array[:, :4].sum(axis=1)
    extensor = rms_array[:, 4:].sum(axis=1)
    denom = np.maximum(flexor, extensor)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return np.minimum(flexor, extensor) / denom

def compute_flexor_extensor_balance(rms_array):
    flexor = rms_array[:, :4].sum(axis=1)
    extensor = rms_array[:, 4:].sum(axis=1)
    extensor = np.where(extensor < 1e-9, 1e-9, extensor)
    return flexor / extensor

def classify_temporal_profile(rms_series):
    slope, _, r, _, _ = linregress(np.arange(len(rms_series)), rms_series)
    cv = np.std(rms_series) / (np.mean(rms_series) + 1e-9)
    if cv > 0.40:
        return 'Intermittent'
    if slope > 0 and r ** 2 > 0.3:
        return 'Ramp-up'
    if slope < 0 and r ** 2 > 0.3:
        return 'Ramp-down'
    return 'Sustained'

def activation_level(sensor_rms, session_mean):
    if session_mean < 1e-9:
        return 'Unknown'
    ratio = sensor_rms / session_mean * 100
    if ratio < 30:
        return 'Low'
    elif ratio < 70:
        return 'Moderate'
    elif ratio <= 100:
        return 'High'
    else:
        return 'VeryHigh'


# ==============================
# PROCESS ONE PARTICIPANT
# ==============================

def process_participant(participant):
    """Full feature extraction from DATA/{P}/Gesture_1.csv."""
    raw_path = os.path.join(DATA_DIR, participant, RAW_FILENAME)
    out_long = os.path.join(DATA_DIR, participant, 'features_long.csv')
    out_short = os.path.join(DATA_DIR, participant, 'features_short.csv')
    out_json = os.path.join(DATA_DIR, participant, 'activation_features.json')

    print(f"\n{'='*60}")
    print(f"Activation Feature Extraction — {participant}")
    print(f"{'='*60}")

    if not os.path.exists(raw_path):
        print(f"  SKIP: {raw_path} not found.")
        return False

    # Auto-detect sensor columns
    sensor_map = detect_sensor_columns(raw_path)
    n_sensors_found = len(sensor_map)
    if n_sensors_found == 0:
        print(f"  ERROR: No EMG sensor columns found in {raw_path}")
        return False
    if n_sensors_found < NUM_SENSORS:
        print(f"  WARNING: Only {n_sensors_found}/{NUM_SENSORS} sensors found")

    print(f"  Detected {n_sensors_found} sensors")

    # Load only EMG + time columns
    use_cols = []
    for sid in sorted(sensor_map.keys()):
        use_cols.extend([sensor_map[sid]['emg_col'], sensor_map[sid]['time_col']])

    print(f"  Loading {raw_path} ({len(use_cols)} columns)...")
    df = pd.read_csv(raw_path, usecols=use_cols, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Build sensor data arrays
    sensor_data = {}
    for sid in sorted(sensor_map.keys()):
        emg_vals = df[sensor_map[sid]['emg_col']].values.astype(float)
        time_vals = df[sensor_map[sid]['time_col']].values.astype(float)
        valid = ~(np.isnan(emg_vals) | np.isnan(time_vals))
        sensor_data[sid] = {'emg': emg_vals[valid], 'time': time_vals[valid]}

    # Sampling rate from first sensor
    first_sid = sorted(sensor_data.keys())[0]
    times = sensor_data[first_sid]['time']
    dt = np.median(np.diff(times[:10000]))
    fs = 1.0 / dt
    total_duration = times[-1] - times[0]
    print(f"  Sampling rate: {fs:.1f} Hz | Duration: {total_duration:.1f}s")

    # Band-pass filter
    print("  Applying band-pass filter (20-450 Hz)...")
    for sid in sensor_data:
        sensor_data[sid]['emg_filtered'] = bandpass_filter(sensor_data[sid]['emg'], fs)

    # ── LONG-WINDOW FEATURES ─────────────────────────────────────
    print(f"  Computing long-window features ({NUM_SEGMENTS} × {SEGMENT_DURATION}s)...")

    feature_names = ['RMS', 'MAV', 'IEMG', 'WL', 'ZC', 'SSC',
                     'WAMP', 'PeakAmp', 'Variance', 'MNF', 'MDF', 'TotalPower']

    long_records = []
    rms_array = np.zeros((NUM_SEGMENTS, NUM_SENSORS))

    for seg in range(NUM_SEGMENTS):
        t_start = seg * SEGMENT_DURATION
        t_end = (seg + 1) * SEGMENT_DURATION
        row = {'Segment': seg + 1, 'TimeStart': t_start, 'TimeEnd': t_end}

        for sid in range(1, NUM_SENSORS + 1):
            if sid not in sensor_data:
                for fname in feature_names:
                    row[f'S{sid}_{fname}'] = 0
                continue

            t = sensor_data[sid]['time']
            emg = sensor_data[sid]['emg_filtered']
            mask = (t >= t_start) & (t < t_end)
            seg_emg = emg[mask]

            feats = compute_long_features(seg_emg, fs)
            for fname in feature_names:
                row[f'S{sid}_{fname}'] = feats[fname]
            rms_array[seg, sid - 1] = feats['RMS']

        long_records.append(row)

    # Cross-channel features
    act_map = compute_activation_map(rms_array)
    conc_idx = compute_concentration_index(act_map)
    pat_stab = compute_pattern_stability(act_map)
    co_contr = compute_co_contraction_ratio(rms_array)
    fe_bal = compute_flexor_extensor_balance(rms_array)

    for seg in range(NUM_SEGMENTS):
        long_records[seg]['ConcentrationIndex'] = float(conc_idx[seg])
        long_records[seg]['PatternStability'] = float(pat_stab[seg]) if seg < len(pat_stab) else None
        long_records[seg]['CoContractionRatio'] = float(co_contr[seg])
        long_records[seg]['FlexorExtensorBalance'] = float(fe_bal[seg])
        for sid in range(1, NUM_SENSORS + 1):
            long_records[seg][f'S{sid}_ActMap'] = float(act_map[seg, sid - 1])

    long_df = pd.DataFrame(long_records)
    long_df.to_csv(out_long, index=False)
    print(f"  Saved: {out_long} ({long_df.shape[0]} × {long_df.shape[1]})")

    # ── SHORT-WINDOW FEATURES ────────────────────────────────────
    print(f"  Computing short-window features ({SHORT_WINDOW}s windows)...")

    all_short_records = None
    onset_data = {}
    sustained_flags = []

    for sid in range(1, NUM_SENSORS + 1):
        if sid not in sensor_data:
            onset_data[sid] = {'onset_s': None, 'offset_s': None, 'duty_cycle': None, 'burst_count': None}
            continue

        emg = sensor_data[sid]['emg_filtered']
        t = sensor_data[sid]['time']

        records, onset_s, offset_s, duty_cycle, burst_count, is_sustained = \
            compute_short_window_features(emg, t, fs)

        sustained_flags.append(is_sustained)

        onset_data[sid] = {
            'onset_s': onset_s,
            'offset_s': offset_s,
            'duty_cycle': round(duty_cycle, 4) if duty_cycle is not None else None,
            'burst_count': burst_count,
        }

        if all_short_records is None:
            all_short_records = [{'Window': r['Window'], 'TimeStart': r['TimeStart'],
                                  'TimeEnd': r['TimeEnd']} for r in records]
        for i, r in enumerate(records):
            if i < len(all_short_records):
                all_short_records[i][f'S{sid}_Envelope'] = r['Envelope']

    short_df = pd.DataFrame(all_short_records)
    short_df.to_csv(out_short, index=False)
    print(f"  Saved: {out_short} ({short_df.shape[0]} × {short_df.shape[1]})")

    # ── SESSION SUMMARY JSON ─────────────────────────────────────
    print("  Building session summary...")

    session_mean_rms = float(rms_array.mean())
    sensor_mean_rms = rms_array.mean(axis=0)
    sensor_cv_rms = rms_array.std(axis=0) / (sensor_mean_rms + 1e-9)
    dominant_sensor = int(np.argmax(sensor_mean_rms)) + 1
    weakest_sensor = int(np.argmin(sensor_mean_rms)) + 1
    d_w_ratio = float(sensor_mean_rms.max() / (sensor_mean_rms.min() + 1e-9))
    inter_corr = compute_inter_channel_corr(rms_array)

    early, mid, late = slice(0, 20), slice(20, 40), slice(40, 60)

    # Sustained contraction if majority of sensors flagged
    is_sustained_session = sum(sustained_flags) > len(sustained_flags) / 2 if sustained_flags else False

    onset_list = [(sid, onset_data[sid]['onset_s'])
                  for sid in range(1, NUM_SENSORS + 1)
                  if onset_data[sid]['onset_s'] is not None]
    onset_list.sort(key=lambda x: x[1])
    onset_sequence = [f"S{sid}" for sid, _ in onset_list]

    sensors_json = {}
    for sid in range(1, NUM_SENSORS + 1):
        s_rms = rms_array[:, sid - 1]
        slope, _, _, _, _ = linregress(np.arange(NUM_SEGMENTS), s_rms)
        tp = classify_temporal_profile(s_rms)
        al = activation_level(float(sensor_mean_rms[sid - 1]), session_mean_rms)

        sensors_json[f'S{sid}'] = {
            'mean_rms': round(float(sensor_mean_rms[sid - 1]), 6),
            'cv_rms': round(float(sensor_cv_rms[sid - 1]), 4),
            'iemg_total': round(float(long_df[f'S{sid}_IEMG'].sum()), 4),
            'mean_wl': round(float(long_df[f'S{sid}_WL'].mean()), 2),
            'mean_zc': round(float(long_df[f'S{sid}_ZC'].mean()), 1),
            'mean_wamp': round(float(long_df[f'S{sid}_WAMP'].mean()), 1),
            'mean_mnf': round(float(long_df[f'S{sid}_MNF'].mean()), 2),
            'mean_mdf': round(float(long_df[f'S{sid}_MDF'].mean()), 2),
            'peak_amplitude': round(float(long_df[f'S{sid}_PeakAmp'].max()), 6),
            'rms_slope': round(slope, 8),
            'activation_level': al,
            'temporal_profile': tp,
            'onset_s': onset_data[sid]['onset_s'],
            'offset_s': onset_data[sid]['offset_s'],
            'duty_cycle': onset_data[sid]['duty_cycle'],
            'burst_count': onset_data[sid]['burst_count'],
            'rms_early': round(float(s_rms[early].mean()), 6),
            'rms_mid': round(float(s_rms[mid].mean()), 6),
            'rms_late': round(float(s_rms[late].mean()), 6),
        }

    summary = {
        'participant': participant,
        'gesture': 'Gesture_1',
        'n_segments': NUM_SEGMENTS,
        'duration_s': round(total_duration, 1),
        'n_sensors': n_sensors_found,
        'sampling_rate_hz': round(fs, 1),
        'session_mean_rms': round(session_mean_rms, 6),
        'raw_data_source': raw_path,
        'feature_source': 'raw_signal',
        'sensors': sensors_json,
        'cross_channel': {
            'dominant_sensor': dominant_sensor,
            'weakest_sensor': weakest_sensor,
            'dominant_weakest_ratio': round(d_w_ratio, 2),
            'mean_concentration_index': round(float(conc_idx.mean()), 4),
            'mean_pattern_stability': round(float(pat_stab.mean()), 4),
            'inter_channel_correlation': round(inter_corr, 4),
            'mean_co_contraction_ratio': round(float(co_contr.mean()), 4),
            'mean_flexor_extensor_balance': round(float(fe_bal.mean()), 4),
            'overall_temporal_profile': classify_temporal_profile(rms_array.mean(axis=1)),
        },
        'phase_summary': {
            'early_rms': [round(float(rms_array[early, s].mean()), 6) for s in range(NUM_SENSORS)],
            'mid_rms': [round(float(rms_array[mid, s].mean()), 6) for s in range(NUM_SENSORS)],
            'late_rms': [round(float(rms_array[late, s].mean()), 6) for s in range(NUM_SENSORS)],
        },
        'is_sustained_contraction': is_sustained_session,
        'onset_summary': None if is_sustained_session else {
            'onset_sequence': onset_sequence,
            'first_onset_s': onset_list[0][1] if onset_list else None,
            'last_onset_s': onset_list[-1][1] if onset_list else None,
            'all_active_by_s': onset_list[-1][1] if onset_list else None,
            'per_sensor': {f'S{sid}': onset_data[sid] for sid in range(1, NUM_SENSORS + 1)},
        },
    }

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_json}")

    print(f"\n  Session mean RMS: {session_mean_rms:.6f} mV")
    print(f"  Dominant: S{dominant_sensor} ({d_w_ratio:.1f}x)")
    print(f"  Profile: {summary['cross_channel']['overall_temporal_profile']}")
    print(f"  Co-contraction ratio: {summary['cross_channel']['mean_co_contraction_ratio']:.3f}")
    if is_sustained_session:
        print(f"  Sustained contraction detected — onset/offset/duty_cycle not reported")
    else:
        print(f"  Onset sequence: {onset_sequence}")
        print(f"  Duty cycles: {[onset_data[s]['duty_cycle'] for s in range(1, 9)]}")

    return True


# ==============================
# MAIN
# ==============================

def main():
    if len(sys.argv) > 1:
        participants = sys.argv[1:]
    else:
        # Auto-discover: find all P* folders that have Gesture_1.csv
        participants = []
        for p in PARTICIPANTS:
            raw = os.path.join(DATA_DIR, p, RAW_FILENAME)
            if os.path.exists(raw):
                participants.append(p)

        if not participants:
            print(f"No participants found with {RAW_FILENAME} in {DATA_DIR}/P*/")
            print(f"Expected: DATA/P1/Gesture_1.csv, DATA/P2/Gesture_1.csv, ...")
            sys.exit(1)

    print(f"Participants to process: {participants}")
    print(f"Looking for: DATA/{{P}}/{RAW_FILENAME}\n")

    success = 0
    skipped = 0
    for p in participants:
        raw = os.path.join(DATA_DIR, p, RAW_FILENAME)
        if os.path.exists(raw):
            if process_participant(p):
                success += 1
        else:
            print(f"\n  SKIP {p}: {raw} not found")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Done: {success} processed | {skipped} skipped | {len(participants)} total")
    print("="*60)


if __name__ == '__main__':
    main()
