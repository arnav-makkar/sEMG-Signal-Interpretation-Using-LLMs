#!/usr/bin/env python3
"""
Evaluation script - creates comparison visualization for all participants.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
DATA_DIR = 'DATA'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EVALUATION - ALL PARTICIPANTS")
print("=" * 60)

# Load all predictions
all_data = []
for p in PARTICIPANTS:
    pred_file = os.path.join(DATA_DIR, p, 'predictions.csv')
    if os.path.exists(pred_file):
        df = pd.read_csv(pred_file)
        df['Participant'] = p
        all_data.append(df)
        print(f"{p}: Loaded {len(df)} rows")

if not all_data:
    print("No predictions found!")
    exit(1)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(os.path.join(RESULTS_DIR, 'all_predictions.csv'), index=False)
print(f"\n✓ Combined: {len(combined_df)} rows → results/all_predictions.csv")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, p in enumerate(PARTICIPANTS):
    ax = axes[i]
    df = combined_df[combined_df['Participant'] == p]
    
    ax.plot(df['Segment'], df['GroundTruth'], 'b-o', label='Ground Truth', markersize=3)
    ax.plot(df['Segment'], df['Predicted'], 'r-s', label='Predicted', markersize=3, alpha=0.7)
    
    mae = df['AbsError'].mean()
    corr = df[['Predicted', 'GroundTruth']].corr().iloc[0, 1]
    
    ax.set_title(f'{p} - MAE: {mae:.2f}, r: {corr:.2f}')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Fatigue Score')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 10.5)

# Summary plot
ax = axes[5]
summary_df = combined_df.groupby('Participant').agg({
    'AbsError': 'mean',
}).reset_index()
summary_df['Correlation'] = [
    combined_df[combined_df['Participant'] == p][['Predicted', 'GroundTruth']].corr().iloc[0, 1]
    for p in summary_df['Participant']
]

x = range(len(summary_df))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], summary_df['AbsError'], width, label='MAE', color='red', alpha=0.7)
ax2 = ax.twinx()
bars2 = ax2.bar([i + width/2 for i in x], summary_df['Correlation'], width, label='Correlation', color='blue', alpha=0.7)

ax.set_xlabel('Participant')
ax.set_ylabel('MAE', color='red')
ax2.set_ylabel('Correlation', color='blue')
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Participant'])
ax.set_title('Summary: MAE vs Correlation')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'), dpi=150)
print(f"✓ Saved: results/comparison.png")
plt.close()

# Print summary
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(summary_df.to_string(index=False))
print(f"\nAverage MAE: {summary_df['AbsError'].mean():.3f}")
print(f"Average Correlation: {summary_df['Correlation'].mean():.3f}")
