#!/usr/bin/env python3
"""
Activation Prompt Generation
==============================
Builds LLM prompts for Workflow 4 — Muscle Activation Analysis.
Structure:
  1. System instruction
  2. Clinical rules (A1-A13 + P1-P7) — hardcoded directly in prompt
  3. RAG context — relevant research paper chunks from ChromaDB
  4. Session feature data (expanded: 12 features + onset/duty cycle)
  5. Task instruction + output schema

Output: DATA/{P}/activation_prompt.txt
"""

import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import EMGPaperRetriever
from rag.rules import SYSTEM_PROMPT, ACTIVATION_RULES, PHYSIOLOGICAL_CONTEXT

PARTICIPANTS = [f'P{i}' for i in range(1, 11)]  # P1-P10
DATA_DIR = 'DATA'

OUTPUT_SCHEMA = """{
  "overall_activation": "low|moderate|high|very_high",
  "dominant_sensors": ["S4", "S6"],
  "weakest_sensors": ["S1", "S5"],
  "dominant_weakest_ratio": 3.75,
  "temporal_profile": "Sustained|Ramp-up|Ramp-down|Intermittent|Compensatory",
  "activation_consistency": "consistent|moderate|inconsistent",
  "per_sensor": [
    {
      "sensor": "S1",
      "activation_level": "low|moderate|high|very_high",
      "temporal_profile": "Sustained|Ramp-up|Ramp-down|Intermittent",
      "cv": 0.15,
      "notable": "string or null"
    }
  ],
  "cross_channel": {
    "concentration_index": 0.18,
    "pattern_stability": 0.85,
    "inter_channel_corr": 0.45,
    "co_contraction_likely": true,
    "spatial_compensation": false
  },
  "onset_assessment": null_or_object,
  "duty_cycle_assessment": null_or_object,
  "motor_unit_recruitment_signs": {
    "detected": true,
    "sensors_affected": ["S4"],
    "evidence": "string"
  },
  "gesture_pattern_notes": "description of S1-S4 vs S5-S8 activation balance",
  "normalization_note": "Features in absolute mV; cross-subject comparisons unreliable per A13/P6.",
  "confidence": 0.85,
  "flags": ["list of notable findings"],
  "clinical_notes": "brief clinical summary"
}"""


def build_session_block(summary: dict) -> str:
    sensors = summary['sensors']
    cc = summary['cross_channel']
    phases = summary['phase_summary']
    onset = summary.get('onset_summary')
    has_full = summary.get('feature_source') == 'raw_signal'

    lines = ["<SESSION_DATA>"]
    lines.append(f"participant={summary['participant']} | gesture={summary['gesture']} | "
                 f"duration={summary['duration_s']}s | n_segments={summary['n_segments']} | "
                 f"n_sensors={summary['n_sensors']} | session_mean_rms={summary['session_mean_rms']:.6f}mV")
    if summary.get('sampling_rate_hz'):
        lines.append(f"sampling_rate={summary['sampling_rate_hz']}Hz | "
                     f"feature_source={summary.get('feature_source', 'unknown')}")
    lines.append("")

    # Per-sensor summary
    lines.append("PER-SENSOR SUMMARY:")
    if has_full:
        lines.append("Sensor | RMS(mV) | CV   | MNF(Hz) | MDF(Hz) | WAMP  | ZC     | WL     | Level     | Profile")
        lines.append("-" * 100)
        for sid, s in sensors.items():
            lines.append(
                f"  {sid}: rms={s['mean_rms']:.6f} cv={s['cv_rms']:.3f} "
                f"mnf={s['mean_mnf']:.1f} mdf={s.get('mean_mdf', 'N/A')} "
                f"wamp={s.get('mean_wamp', 'N/A')} zc={s.get('mean_zc', 'N/A')} "
                f"wl={s.get('mean_wl', 'N/A')} "
                f"level={s['activation_level']} profile={s['temporal_profile']}"
            )
    else:
        lines.append("Sensor | RMS(mV) | CV   | MNF(Hz) | Level     | Profile")
        lines.append("-" * 70)
        for sid, s in sensors.items():
            lines.append(
                f"  {sid}: rms={s['mean_rms']:.6f} cv={s['cv_rms']:.3f} "
                f"mnf={s.get('mean_mnf', 'N/A')} "
                f"level={s['activation_level']} profile={s['temporal_profile']}"
            )

    # Phase averages
    lines.append("")
    lines.append("PHASE AVERAGES (RMS mV, S1..S8):")
    lines.append(f"  early(0-400s):    {[round(v,6) for v in phases['early_rms']]}")
    lines.append(f"  mid(400-800s):    {[round(v,6) for v in phases['mid_rms']]}")
    lines.append(f"  late(800-1200s):  {[round(v,6) for v in phases['late_rms']]}")

    # Onset/duty cycle — only include if NOT a sustained contraction
    is_sustained = summary.get('is_sustained_contraction', False)
    if is_sustained:
        lines.append("")
        lines.append("ACTIVATION TIMING: NOT APPLICABLE")
        lines.append("  This is a sustained-contraction recording (participant actively gripping")
        lines.append("  from start of recording). Onset, offset, duty_cycle, and burst_count are")
        lines.append("  not meaningful and are omitted. Do NOT speculate about activation timing.")
    elif onset and onset.get('onset_sequence'):
        lines.append("")
        lines.append("ACTIVATION TIMING:")
        lines.append(f"  onset_sequence={onset['onset_sequence']}")
        lines.append(f"  first_onset={onset.get('first_onset_s')}s | "
                     f"all_active_by={onset.get('all_active_by_s')}s")
        duty_str = []
        for sid in range(1, 9):
            dc = onset['per_sensor'].get(f'S{sid}', {}).get('duty_cycle')
            duty_str.append(f"S{sid}:{dc}" if dc is not None else f"S{sid}:N/A")
        lines.append(f"  duty_cycle=[{', '.join(duty_str)}]")
        burst_str = []
        for sid in range(1, 9):
            bc = onset['per_sensor'].get(f'S{sid}', {}).get('burst_count')
            burst_str.append(f"S{sid}:{bc}" if bc is not None else f"S{sid}:N/A")
        lines.append(f"  burst_count=[{', '.join(burst_str)}]")

    # Cross-channel
    lines.append("")
    lines.append("CROSS-CHANNEL:")
    lines.append(f"  dominant=S{cc['dominant_sensor']} weakest=S{cc['weakest_sensor']} "
                 f"ratio={cc['dominant_weakest_ratio']:.2f}x")
    lines.append(f"  concentration_index={cc['mean_concentration_index']:.4f} [0=diffuse,1=focal]")
    lines.append(f"  pattern_stability={cc['mean_pattern_stability']:.4f} [0=changing,1=stable]")
    lines.append(f"  inter_channel_corr={cc['inter_channel_correlation']:.4f} [>0.8=crosstalk_risk]")
    lines.append(f"  co_contraction_ratio={cc.get('mean_co_contraction_ratio', 'N/A')} [>0.5=both_groups_active]")
    lines.append(f"  flexor_extensor_balance={cc.get('mean_flexor_extensor_balance', 'N/A')} [>1=flexor-dominant]")
    lines.append(f"  overall_profile={cc['overall_temporal_profile']}")

    lines.append("")
    lines.append("GESTURE_CONTEXT: S1-S4=palmar/flexor | S5-S8=dorsal/extensor")
    if not has_full:
        lines.append("MISSING: WL, ZC, WAMP, MDF, onset/offset, duty_cycle — raw signal not available for this participant.")
    lines.append("</SESSION_DATA>")
    return "\n".join(lines)


def detect_signal_flags(summary: dict) -> list:
    flags = []
    cc = summary['cross_channel']
    sensors = summary['sensors']

    if cc['inter_channel_correlation'] > 0.8:
        flags.append('crosstalk_detected')
    if cc['mean_concentration_index'] < 0.15:
        flags.append('co_contraction')
    if cc.get('mean_co_contraction_ratio', 0) > 0.5:
        flags.append('co_contraction')
    if cc['mean_pattern_stability'] < 0.9:
        flags.append('spatial_compensation')

    cvs = [s['cv_rms'] for s in sensors.values()]
    if any(cv > 0.30 for cv in cvs):
        flags.append('rms_variability')

    profiles = [s['temporal_profile'] for s in sensors.values()]
    if 'Ramp-up' in profiles:
        flags.append('motor_unit_recruitment')

    # Check WAMP if available
    wamps = [s.get('mean_wamp') for s in sensors.values() if s.get('mean_wamp') is not None]
    if wamps and max(wamps) > 2 * (sum(wamps) / len(wamps)):
        flags.append('motor_unit_recruitment')

    # Duty cycle
    dcs = [s.get('duty_cycle') for s in sensors.values() if s.get('duty_cycle') is not None]
    if dcs and all(dc > 0.9 for dc in dcs):
        flags.append('high_duty_cycle')
    if dcs:
        flags.append('duty_cycle')

    if summary.get('onset_summary') and summary['onset_summary'].get('onset_sequence'):
        flags.append('onset_detection')

    flags.extend(['activation_level', 'normalization'])
    return list(set(flags))


def generate_prompt(participant: str, retriever: EMGPaperRetriever) -> bool:
    json_path = os.path.join(DATA_DIR, participant, 'activation_features.json')
    output_path = os.path.join(DATA_DIR, participant, 'activation_prompt.txt')

    if not os.path.exists(json_path):
        print(f"  ERROR: {json_path} not found.")
        return False

    with open(json_path) as f:
        summary = json.load(f)

    flags = detect_signal_flags(summary)
    paper_chunks = retriever.retrieve_for_workflow('activation', signal_flags=flags, top_k=4)
    rag_context = retriever.format_context(paper_chunks, header="RESEARCH_CONTEXT")
    session_block = build_session_block(summary)

    prompt = f"""SYSTEM:
{SYSTEM_PROMPT}

<CLINICAL_RULES>
--- Physiological Context ---
{PHYSIOLOGICAL_CONTEXT}

--- Workflow 4: Muscle Activation Rules ---
{ACTIVATION_RULES}
</CLINICAL_RULES>

{rag_context}

{session_block}

TASK:
Perform Workflow 4 — Muscle Activation Analysis.
Assess: activation intensity (A1), spatial distribution (A2, A5), co-contraction (A3),
consistency (A4), temporal profile (A6), motor unit recruitment indicators (A8),
and cross-channel patterns. Apply rules A1-A13 and physiological context P1-P7.
Use the research context to ground your clinical reasoning.
Flag anomalies. Note where features are missing (null) and adjust confidence.

IF the SESSION_DATA says "ACTIVATION TIMING: NOT APPLICABLE", then set
"onset_assessment" and "duty_cycle_assessment" to null in the output.
Do NOT invent onset times or duty cycles for sustained-contraction data.

For motor_unit_recruitment_signs, cite specific sensor IDs with ACTUAL WAMP/ZC
values that support the claim. WAMP is the primary indicator — higher WAMP
means more MU recruitment. Do NOT confuse high MNF with high WAMP.

OUTPUT (return ONLY valid JSON — no markdown, no explanation outside JSON):
{OUTPUT_SCHEMA}
"""

    with open(output_path, 'w') as f:
        f.write(prompt)

    token_est = len(prompt.split()) * 1.3
    papers_cited = [c.get('paper', '?') for c in paper_chunks]
    has_full = summary.get('feature_source') == 'raw_signal'
    print(f"  {participant}: {len(paper_chunks)} paper chunks | ~{token_est:.0f} tokens | "
          f"{'full features' if has_full else 'limited (fallback)'} | papers={papers_cited}")
    return True


def main():
    retriever = EMGPaperRetriever()
    print(f"RAG backend: {retriever.get_backend()}\n")

    participants = [sys.argv[1]] if len(sys.argv) > 1 else PARTICIPANTS
    success = 0
    for p in participants:
        print(f"Generating prompt for {p}...")
        if generate_prompt(p, retriever):
            success += 1

    print(f"\nDone: {success}/{len(participants)} prompts generated")


if __name__ == '__main__':
    main()
