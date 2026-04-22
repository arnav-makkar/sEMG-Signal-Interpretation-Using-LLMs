#!/usr/bin/env python3
"""
Activation Evaluation & Visualization
=======================================
Loads activation_result.json + activation_features.json for each participant.
Produces:
  - results/activation_plots/{P}_activation.png
  - results/activation_summary.csv
  - results/activation_report.md
"""

import json
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PARTICIPANTS = [f'P{i}' for i in range(1, 11)]  # P1-P10
DATA_DIR = 'DATA'
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'activation_plots')
NUM_SENSORS = 8


def load_json(participant, filename):
    path = os.path.join(DATA_DIR, participant, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(participant, filename):
    path = os.path.join(DATA_DIR, participant, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ==============================
# PLOTTING
# ==============================

def plot_participant(participant, result, features, long_df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    has_full = features.get('feature_source') == 'raw_signal'
    onset = features.get('onset_summary')

    n_rows = 3 if has_full else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    fig.suptitle(f'{participant} — Muscle Activation Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.5, wspace=0.35)

    sensors = [f'S{i+1}' for i in range(NUM_SENSORS)]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63',
              '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']
    flexor_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    extensor_colors = ['#9C27B0', '#00BCD4', '#FF5722', '#607D8B']

    mean_rms = [features['sensors'][s]['mean_rms'] for s in sensors]

    # ── 1. Mean RMS per sensor ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bar_colors = flexor_colors + extensor_colors
    bars = ax1.bar(sensors, mean_rms, color=bar_colors)
    ax1.axhline(features['session_mean_rms'], color='red', linestyle='--', linewidth=1.2, label='Session mean')
    ax1.set_title('Mean RMS per Sensor\n(blue/green=flexor, purple/teal=extensor)')
    ax1.set_ylabel('RMS (mV)')
    ax1.legend(fontsize=7)
    ax1.tick_params(axis='x', rotation=45)
    per_sensor_llm = {s['sensor']: s for s in result.get('per_sensor', [])} if 'per_sensor' in result else {}
    for bar, s in zip(bars, sensors):
        level = per_sensor_llm.get(s, {}).get('activation_level', '')
        if level:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                     level[:3], ha='center', va='bottom', fontsize=6, color='#333')

    # ── 2. Activation map ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if long_df is not None and 'S1_ActMap' in long_df.columns:
        act_map_cols = [f'S{i+1}_ActMap' for i in range(NUM_SENSORS)]
        mean_act = long_df[act_map_cols].mean().values
        ax2.bar(sensors, mean_act * 100, color=bar_colors)
        ax2.set_title('Activation Map (% of total RMS)')
        ax2.set_ylabel('Activation %')
        ax2.tick_params(axis='x', rotation=45)
        cc = features['cross_channel']
        ax2.text(0.02, 0.95,
                 f"Conc.Idx={cc['mean_concentration_index']:.3f}\n"
                 f"Pat.Stab={cc['mean_pattern_stability']:.3f}\n"
                 f"CoCont={cc.get('mean_co_contraction_ratio', 'N/A')}",
                 transform=ax2.transAxes, fontsize=7, va='top')

    # ── 3. RMS over time (top 4 sensors) ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if long_df is not None:
        rms_cols = [f'S{i+1}_RMS' for i in range(NUM_SENSORS)]
        if rms_cols[0] in long_df.columns:
            top4 = sorted(range(NUM_SENSORS), key=lambda i: mean_rms[i], reverse=True)[:4]
            for idx in top4:
                s = f'S{idx+1}'
                ax3.plot(long_df['Segment'], long_df[f'{s}_RMS'], label=s, color=colors[idx], linewidth=1.2)
            ax3.set_title('RMS Over Time — Top 4 Sensors')
            ax3.set_xlabel('Segment (20s each)')
            ax3.set_ylabel('RMS (mV)')
            ax3.legend(fontsize=8)
            for x, label in [(20, 'Mid'), (40, 'Late')]:
                ax3.axvline(x, color='gray', linestyle=':', linewidth=0.8)

    # ── 4. Co-contraction ratio + Flexor/Extensor balance ──────────
    ax4 = fig.add_subplot(gs[1, 0])
    if long_df is not None and 'CoContractionRatio' in long_df.columns:
        ax4.plot(long_df['Segment'], long_df['CoContractionRatio'], color='#E91E63', linewidth=1.5, label='Co-contraction')
        ax4.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Both groups active')
        ax4.set_title('Co-Contraction Ratio\n(min/max of flexor vs extensor)')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('Ratio')
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize=7)
    elif long_df is not None and 'ConcentrationIndex' in long_df.columns:
        ax4.plot(long_df['Segment'], long_df['ConcentrationIndex'], color='#E91E63', linewidth=1.5)
        ax4.set_title('Concentration Index Over Time')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('Concentration Index')
        ax4.set_ylim(0, 1)

    # ── 5. Duty cycle OR WAMP (motor unit recruitment) ─────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if onset and onset.get('per_sensor'):
        duty_vals = []
        for sid in range(1, NUM_SENSORS + 1):
            dc = onset['per_sensor'].get(f'S{sid}', {}).get('duty_cycle')
            duty_vals.append(dc if dc is not None else 0)
        ax5.bar(sensors, [d * 100 for d in duty_vals], color=bar_colors)
        ax5.set_title('Duty Cycle per Sensor (%)')
        ax5.set_ylabel('% Time Active')
        ax5.set_ylim(0, 105)
        ax5.tick_params(axis='x', rotation=45)
        for i, v in enumerate(duty_vals):
            ax5.text(i, v * 100 + 1, f'{v:.0%}', ha='center', fontsize=7)
    else:
        # For sustained contractions, show WAMP per sensor (MU recruitment indicator)
        wamp_vals = []
        for sid in range(1, NUM_SENSORS + 1):
            w = features['sensors'][f'S{sid}'].get('mean_wamp')
            wamp_vals.append(w if w is not None else 0)
        ax5.bar(sensors, wamp_vals, color=bar_colors)
        ax5.set_title('Mean WAMP per Sensor\n(motor unit recruitment)')
        ax5.set_ylabel('WAMP count')
        ax5.tick_params(axis='x', rotation=45)
        max_w = max(wamp_vals) if max(wamp_vals) > 0 else 1
        ax5.set_ylim(0, max_w * 1.15)
        for i, v in enumerate(wamp_vals):
            if v > 0:
                ax5.text(i, v + max_w * 0.02, f'{int(v)}', ha='center', fontsize=7)

    # ── 6. LLM result summary ─────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    if 'parse_error' not in result:
        lines = [
            f"Overall: {result.get('overall_activation', '?').upper()}",
            f"Profile: {result.get('temporal_profile', '?')}",
            f"Consistency: {result.get('activation_consistency', '?')}",
            f"Dominant: {result.get('dominant_sensors', [])}",
            f"Weakest: {result.get('weakest_sensors', [])}",
            f"D/W ratio: {result.get('dominant_weakest_ratio', '?')}x",
        ]
        cc = result.get('cross_channel', {})
        lines.append(f"Co-contract: {cc.get('co_contraction_likely', '?')}")
        lines.append(f"Spat.comp: {cc.get('spatial_compensation', '?')}")
        lines.append(f"Confidence: {result.get('confidence', '?')}")

        onset_r = result.get('onset_assessment', {})
        if onset_r:
            seq = onset_r.get('onset_sequence', [])
            if seq:
                lines.append(f"Onset: {seq[:4]}...")

        dc_r = result.get('duty_cycle_assessment', {})
        if dc_r:
            lines.append(f"Duty: {dc_r.get('overall', '?')}")

        mur = result.get('motor_unit_recruitment_signs', {})
        if mur:
            lines.append(f"MU recruit: {mur.get('detected', '?')}")

        flags = result.get('flags', [])
        if flags:
            lines.append(f"Flags: {', '.join(flags[:3])}")

        notes = result.get('clinical_notes', '')
        if notes:
            words = notes.split()
            wrapped = []
            line = ''
            for w in words:
                if len(line) + len(w) < 40:
                    line += w + ' '
                else:
                    wrapped.append(line.strip())
                    line = w + ' '
            if line:
                wrapped.append(line.strip())
            lines.append('Notes:')
            lines.extend(wrapped[:3])
    else:
        lines = ['PARSE ERROR', 'LLM did not return valid JSON']

    ax6.text(0.05, 0.95, '\n'.join(lines),
             transform=ax6.transAxes, fontsize=8, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('LLM Assessment', fontsize=9)

    # ── Row 3: Extra plots for full-feature participants ───────────
    if has_full and n_rows >= 3:
        ax7 = fig.add_subplot(gs[2, 0])

        # If onset data is available (non-sustained), show onset timeline.
        # Otherwise (sustained contraction), show phase-wise RMS comparison.
        if onset and onset.get('onset_sequence'):
            onset_times = [onset['per_sensor'].get(s, {}).get('onset_s', None)
                           for s in [f'S{i+1}' for i in range(NUM_SENSORS)]]
            valid = [(s, t) for s, t in zip(sensors, onset_times) if t is not None]
            valid.sort(key=lambda x: x[1])
            y_pos = range(len(valid))
            bar_c = [bar_colors[int(s[1]) - 1] for s, _ in valid]
            ax7.barh(y_pos, [t for _, t in valid], color=bar_c, height=0.6)
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels([s for s, _ in valid])
            ax7.set_xlabel('Onset Time (s)')
            ax7.set_title('Activation Onset Sequence')
            for i, (s, t) in enumerate(valid):
                ax7.text(t + 0.01, i, f'{t:.3f}s', va='center', fontsize=7)
        else:
            # Phase comparison: early/mid/late RMS per sensor
            phases = features.get('phase_summary', {})
            early = phases.get('early_rms', [0] * NUM_SENSORS)
            mid = phases.get('mid_rms', [0] * NUM_SENSORS)
            late = phases.get('late_rms', [0] * NUM_SENSORS)

            x = list(range(NUM_SENSORS))
            width = 0.27
            ax7.bar([i - width for i in x], early, width,
                    label='Early (0-400s)', color='#60a5fa')
            ax7.bar(x, mid, width,
                    label='Mid (400-800s)', color='#a78bfa')
            ax7.bar([i + width for i in x], late, width,
                    label='Late (800-1200s)', color='#f87171')
            ax7.set_xticks(x)
            ax7.set_xticklabels(sensors, rotation=45)
            ax7.set_ylabel('RMS (mV)')
            ax7.set_title('Phase-wise RMS\n(fatigue trend indicator)')
            ax7.legend(fontsize=7, loc='upper right')
            ax7.grid(axis='y', alpha=0.3, linestyle=':')

        # MNF/MDF over time
        ax8 = fig.add_subplot(gs[2, 1:])
        if long_df is not None and 'S1_MNF' in long_df.columns:
            top4 = sorted(range(NUM_SENSORS), key=lambda i: mean_rms[i], reverse=True)[:4]
            for idx in top4:
                s = f'S{idx+1}'
                ax8.plot(long_df['Segment'], long_df[f'{s}_MNF'], label=f'{s} MNF',
                         color=colors[idx], linewidth=1.2)
                ax8.plot(long_df['Segment'], long_df[f'{s}_MDF'], label=f'{s} MDF',
                         color=colors[idx], linewidth=1.0, linestyle='--')
            ax8.set_title('MNF & MDF Over Time — Top 4 Sensors\n(MDF decline = fatigue)')
            ax8.set_xlabel('Segment (20s each)')
            ax8.set_ylabel('Frequency (Hz)')
            ax8.legend(fontsize=7, ncol=2)
            for x in [20, 40]:
                ax8.axvline(x, color='gray', linestyle=':', linewidth=0.8)

    out_path = os.path.join(PLOTS_DIR, f'{participant}_activation.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ==============================
# SUMMARY TABLE + REPORT
# ==============================

def build_summary_table(all_results, all_features):
    rows = []
    for p in PARTICIPANTS:
        result = all_results.get(p)
        features = all_features.get(p)
        if not result or not features:
            continue

        cc = features['cross_channel']
        meta = result.get('_meta', {})
        rows.append({
            'participant': p,
            'feature_source': features.get('feature_source', 'unknown'),
            'overall_activation': result.get('overall_activation', '?'),
            'temporal_profile': result.get('temporal_profile', '?'),
            'consistency': result.get('activation_consistency', '?'),
            'dominant_sensors': str(result.get('dominant_sensors', [])),
            'dominant_weakest_ratio': result.get('dominant_weakest_ratio', None),
            'concentration_index': cc['mean_concentration_index'],
            'pattern_stability': cc['mean_pattern_stability'],
            'co_contraction_ratio': cc.get('mean_co_contraction_ratio', None),
            'flexor_extensor_balance': cc.get('mean_flexor_extensor_balance', None),
            'confidence': result.get('confidence', None),
            'parse_success': meta.get('parse_success', False),
        })
    return pd.DataFrame(rows)


def build_report(summary_df, all_results, all_features):
    lines = [
        "# Muscle Activation Analysis Report",
        "",
        "## Overview",
        f"Participants analyzed: {len(summary_df)}",
        "",
        "## RAG System",
        "Backend: ChromaDB with sentence-transformers (all-MiniLM-L6-v2)",
        "Papers indexed: Konrad 2005, Nazmi 2016, Phinyomark 2012, Vigotsky 2018, Yang 2024 (EMGBench)",
        "Rules: A1-A13 (activation) + P1-P7 (physiological context) embedded directly in prompt",
        "",
        "## Summary Table",
    ]

    try:
        lines.append(summary_df.to_markdown(index=False))
    except ImportError:
        lines.append(summary_df.to_string(index=False))

    lines.append("")
    lines.append("## Per-Participant Details")

    for p in PARTICIPANTS:
        result = all_results.get(p)
        features = all_features.get(p)
        if not result or 'parse_error' in result:
            continue

        lines.append(f"\n### {p}")
        lines.append(f"- Feature source: **{features.get('feature_source', 'unknown')}**")
        lines.append(f"- Overall activation: **{result.get('overall_activation', '?')}**")
        lines.append(f"- Temporal profile: {result.get('temporal_profile', '?')}")
        lines.append(f"- Dominant sensors: {result.get('dominant_sensors', [])}")
        lines.append(f"- Gesture pattern: {result.get('gesture_pattern_notes', 'N/A')}")

        onset_a = result.get('onset_assessment', {})
        if onset_a and onset_a.get('onset_sequence'):
            lines.append(f"- Onset sequence: {onset_a['onset_sequence']}")

        dc_a = result.get('duty_cycle_assessment', {})
        if dc_a:
            lines.append(f"- Duty cycle: {dc_a.get('overall', 'N/A')}")

        mur = result.get('motor_unit_recruitment_signs', {})
        if mur:
            lines.append(f"- Motor unit recruitment: {mur.get('detected', 'N/A')}")

        lines.append(f"- Flags: {result.get('flags', [])}")
        notes = result.get('clinical_notes', '')
        if notes:
            lines.append(f"- Clinical notes: {notes}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- P1: Full features from raw signal (12 time/freq domain + onset/duty cycle)")
    lines.append("- P2-P5: Limited features from features.csv (RMS, MAV, MeanFreq only)")
    lines.append("- Amplitude not normalized to %MVC — cross-subject comparisons unreliable (Rule A13/P6)")
    return "\n".join(lines)


# ==============================
# MAIN
# ==============================

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    participants = [sys.argv[1]] if len(sys.argv) > 1 else PARTICIPANTS
    all_results = {}
    all_features = {}

    for p in participants:
        result = load_json(p, 'activation_result.json')
        features = load_json(p, 'activation_features.json')
        long_df = load_csv(p, 'features_long.csv')
        if long_df is None:
            long_df = load_csv(p, 'activation_features_long.csv')

        if not result:
            print(f"  SKIP {p}: no activation_result.json")
            continue
        if not features:
            print(f"  SKIP {p}: no activation_features.json")
            continue

        all_results[p] = result
        all_features[p] = features
        print(f"Plotting {p}...")
        plot_participant(p, result, features, long_df)

    if not all_results:
        print("No results to evaluate.")
        return

    summary_df = build_summary_table(all_results, all_features)
    csv_path = os.path.join(RESULTS_DIR, 'activation_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    report = build_report(summary_df, all_results, all_features)
    report_path = os.path.join(RESULTS_DIR, 'activation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")

    print("\n=== SUMMARY ===")
    cols = ['participant', 'overall_activation', 'temporal_profile', 'consistency', 'confidence', 'parse_success']
    avail_cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[avail_cols].to_string(index=False))


if __name__ == '__main__':
    main()
