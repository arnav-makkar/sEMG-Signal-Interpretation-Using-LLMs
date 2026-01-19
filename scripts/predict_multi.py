#!/usr/bin/env python3
"""
Run predictions for all participants.
Makes 1 API call per participant (5 total).
"""

import pandas as pd
import os
import re
import json
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================
PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
DATA_DIR = 'DATA'
GROUND_TRUTH_FILE = 'DATA/subjective_fatigue.csv'
API_KEY_FILE = '.env'
MODEL_NAME = 'llama-3.1-8b-instant'
TEMPERATURE = 0.1
MAX_TOKENS = 500

# ===============================
# LOAD API KEY
# ===============================
def load_env_file(path=".env"):
    env_vars = {}
    env_path = Path(path)
    if not env_path.is_file():
        return env_vars
    with env_path.open(encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_vars[key] = value.strip().strip('"').strip("'")
    return env_vars

env_vars = load_env_file(API_KEY_FILE)
GROQ_API_KEY = env_vars.get('GROQ_API_KEY')

if not GROQ_API_KEY:
    print(f"ERROR: GROQ_API_KEY not found in {API_KEY_FILE}")
    exit(1)

from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

# ===============================
# LOAD GROUND TRUTH
# ===============================
print("Loading ground truth...")
gt_df = pd.read_csv(GROUND_TRUTH_FILE)
segment_cols = [c for c in gt_df.columns if 'Segment' in c]

def get_ground_truth(participant):
    """Get 60 ground truth scores for a participant."""
    row = gt_df[(gt_df['P ID'] == participant) & (gt_df['Gesture'] == 'Gesture 1')]
    if len(row) == 0:
        return [0] * 60
    return row[segment_cols].values[0].tolist()

# ===============================
# PREDICTION
# ===============================
def predict_participant(participant):
    """Run prediction for a single participant."""
    prompt_file = os.path.join(DATA_DIR, participant, 'prompt.txt')
    output_file = os.path.join(DATA_DIR, participant, 'predictions.csv')
    
    print(f"\n{'='*60}")
    print(f"Predicting {participant}")
    print(f"{'='*60}")
    
    if not os.path.exists(prompt_file):
        print(f"  ⚠️ {prompt_file} not found")
        return None
    
    # Load prompt
    with open(prompt_file, 'r') as f:
        prompt_text = f.read()
    print(f"  Prompt: {len(prompt_text):,} chars")
    
    # Get ground truth
    ground_truth = get_ground_truth(participant)
    print(f"  Ground truth: {ground_truth[:5]}...{ground_truth[-5:]}")
    
    # API call
    print(f"  Calling API ({MODEL_NAME})...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.95
        )
        raw_output = response.choices[0].message.content.strip()
        print(f"  ✓ Response: {len(raw_output)} chars")
    except Exception as e:
        print(f"  ✗ API Error: {str(e)}")
        return None
    
    # Parse response
    predictions = []
    try:
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            json_data = json.loads(json_match.group(0))
            predictions = [int(x) for x in json_data if isinstance(x, (int, float))]
            print(f"  ✓ Parsed JSON: {len(predictions)} values")
    except Exception as e:
        print(f"  JSON parse failed: {e}")
    
    # Fallback to regex
    if not predictions:
        numbers = re.findall(r'\b(\d+)\b', raw_output)
        predictions = [int(n) for n in numbers if 0 <= int(n) <= 10]
        print(f"  Regex fallback: {len(predictions)} values")
    
    # Pad/truncate
    if len(predictions) < 60:
        predictions.extend([5] * (60 - len(predictions)))
    predictions = predictions[:60]
    
    # Create DataFrame
    results = []
    for i in range(60):
        results.append({
            'Segment': i + 1,
            'Predicted': predictions[i],
            'GroundTruth': ground_truth[i],
            'Error': predictions[i] - ground_truth[i],
            'AbsError': abs(predictions[i] - ground_truth[i])
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Quick metrics
    mae = df['AbsError'].mean()
    corr = df[['Predicted', 'GroundTruth']].corr().iloc[0, 1]
    print(f"  MAE: {mae:.3f}, Correlation: {corr:.3f}")
    print(f"  ✓ Saved: {output_file}")
    
    return {'participant': participant, 'mae': mae, 'correlation': corr}

def main():
    print("=" * 60)
    print("PREDICTION - ALL PARTICIPANTS (1 API CALL EACH)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"API calls: {len(PARTICIPANTS)}")
    
    results = []
    for p in PARTICIPANTS:
        r = predict_participant(p)
        if r:
            results.append(r)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print(f"\nAverage MAE: {results_df['mae'].mean():.3f}")
        print(f"Average Correlation: {results_df['correlation'].mean():.3f}")
        
        # Save summary
        results_df.to_csv('results/metrics_summary.csv', index=False)
        print(f"\n✓ Saved: results/metrics_summary.csv")

if __name__ == "__main__":
    main()
