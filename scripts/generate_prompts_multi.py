#!/usr/bin/env python3
"""
Generate batch prompts for all participants.
Creates one prompt per participant containing all 60 segments.
"""

import pandas as pd
import os

# ===============================
# CONFIGURATION
# ===============================
PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
DATA_DIR = 'DATA'
TEMPLATE = """You are a fatigue prediction expert. Predict fatigue scores (0-10) for 60 EMG segments.

## Context
20-min session, 8 EMG sensors. Features: RMS (intensity), Freq (Hz), MAV (amplitude).
Fatigue signs: ↑RMS, ↓Freq over time.

## Few-Shot Examples (Format: Seg|Phase|S1..S8 -> Score)
1|Early|0.032,95,0.028|0.038,92,0.031|... -> Score: 0
12|Early|0.045,87,0.038|0.052,82,0.044|... -> Score: 1
30|Middle|0.068,75,0.058|0.063,77,0.052|... -> Score: 5
50|Late|0.089,65,0.076|0.082,68,0.071|... -> Score: 6
60|End|0.092,63,0.078|0.088,61,0.075|... -> Score: 6

## Data Format
Each line: SegmentID|Phase|S1_data|S2_data|...|S8_data
Where Sx_data = RMS,Frequency,MAV

## All 60 Segments
{segment_data}

## Task
You are a JSON-only API. You must output predictions in strict JSON array format.
Do NOT output any text, markdown, or code blocks.
Do NOT explain your reasoning.

Input: All 60 segments provided above.
Output: A single JSON array containing exactly 60 integers (0-10).

Output Requirements:
1. Output MUST be a valid JSON array.
2. Array MUST contain exactly 60 integers.
3. Generate NEW values based on the input data.

Your JSON Response:
"""

def get_phase(segment_id):
    if segment_id <= 20:
        return "Early"
    elif segment_id <= 40:
        return "Middle"
    else:
        return "Late"

def generate_prompt(participant):
    """Generate batch prompt for a single participant."""
    features_file = os.path.join(DATA_DIR, participant, 'features.csv')
    output_file = os.path.join(DATA_DIR, participant, 'prompt.txt')
    
    print(f"\n{participant}:")
    
    if not os.path.exists(features_file):
        print(f"  ⚠️ {features_file} not found")
        return False
    
    features_df = pd.read_csv(features_file)
    
    # Format all segments
    segment_blocks = []
    for idx, row in features_df.iterrows():
        segment_id = int(row['Segment'])
        phase = get_phase(segment_id)
        
        # Compact format: RMS,Freq,MAV per sensor
        sensor_data = []
        for sensor_id in range(1, 9):
            rms = row[f'S{sensor_id}_RMS']
            freq = row[f'S{sensor_id}_MeanFreq']
            mav = row[f'S{sensor_id}_MAV']
            sensor_data.append(f"{rms:.3f},{freq:.0f},{mav:.3f}")
        
        block = f"{segment_id}|{phase}|" + "|".join(sensor_data)
        segment_blocks.append(block)
    
    # Create prompt
    all_segments_text = "\n".join(segment_blocks)
    prompt = TEMPLATE.replace('{segment_data}', all_segments_text)
    
    # Save
    with open(output_file, 'w') as f:
        f.write(prompt)
    
    print(f"  ✓ Saved: {output_file} ({len(prompt):,} chars, ~{len(prompt)//4:,} tokens)")
    return True

def main():
    print("=" * 60)
    print("PROMPT GENERATION - ALL PARTICIPANTS")
    print("=" * 60)
    
    success = 0
    for p in PARTICIPANTS:
        if generate_prompt(p):
            success += 1
    
    print(f"\n✓ Generated {success}/{len(PARTICIPANTS)} prompts")

if __name__ == "__main__":
    main()
