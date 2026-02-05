#!/usr/bin/env python3
"""
Generate figures for FACC correction report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path('output/facc_detection/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_feature_importance():
    """Bar chart of top RF regressor features."""
    importance = pd.read_csv('output/facc_detection/rf_regressor_importance.csv')
    top15 = importance.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top15))]
    bars = ax.barh(range(len(top15)), top15['importance'] * 100, color=colors)

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('RF Regressor: Top 15 Features for Predicting FACC')

    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, top15['importance'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=9)

    # Highlight hydro_dist_hw
    ax.annotate('Distance from headwater\nexplains 56% of variance',
                xy=(56, 0), xytext=(40, 3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='#27ae60')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_feature_importance.png', bbox_inches='tight')
    plt.close()
    print(f"Saved fig1_feature_importance.png")


def fig2_hydro_dist_vs_facc():
    """Scatter plot showing hydro_dist_hw vs facc relationship."""
    # Load features
    df = pd.read_csv('output/facc_detection/rf_features.csv')

    # Load anomaly IDs
    anomalies = gpd.read_file('output/facc_detection/all_anomalies.geojson')
    anomaly_ids = set(anomalies['reach_id'].values)

    # Sample for plotting (too many points otherwise)
    clean = df[~df['reach_id'].isin(anomaly_ids)].sample(n=min(10000, len(df)), random_state=42)
    anom = df[df['reach_id'].isin(anomaly_ids)]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot clean reaches
    ax.scatter(clean['hydro_dist_hw'] / 1000, clean['facc'],
               alpha=0.3, s=5, c='#3498db', label=f'Clean reaches (n={len(clean):,})')

    # Plot anomalies
    ax.scatter(anom['hydro_dist_hw'] / 1000, anom['facc'],
               alpha=0.7, s=20, c='#e74c3c', label=f'Anomalies (n={len(anom):,})')

    ax.set_xlabel('Distance from Headwater (km)')
    ax.set_ylabel('Flow Accumulation (km²)')
    ax.set_yscale('log')
    ax.set_title('FACC vs Network Position: Anomalies Stand Out')
    ax.legend(loc='upper left')

    # Add annotation
    ax.annotate('Anomalies have facc values\n10-1000x too high for\ntheir network position',
                xy=(200, 1e6), xytext=(400, 1e7),
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                fontsize=10, color='#c0392b')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_hydro_dist_vs_facc.png', bbox_inches='tight')
    plt.close()
    print(f"Saved fig2_hydro_dist_vs_facc.png")


def fig3_fwr_before_after():
    """Histogram showing FWR distribution before/after correction."""
    # Load predictions
    predictions = pd.read_csv('output/facc_detection/rf_regressor_predictions.csv')

    # Load features to get width
    features = pd.read_csv('output/facc_detection/rf_features.csv')
    predictions = predictions.merge(features[['reach_id', 'width']], on='reach_id')

    # Calculate FWR before and after
    predictions['fwr_before'] = predictions['facc'] / predictions['width']
    predictions['fwr_after'] = predictions['predicted_facc'] / predictions['width']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before
    ax1 = axes[0]
    fwr_before = predictions['fwr_before'].clip(upper=50000)
    ax1.hist(fwr_before, bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    ax1.axvline(fwr_before.median(), color='black', linestyle='--', linewidth=2,
                label=f'Median: {fwr_before.median():,.0f}')
    ax1.set_xlabel('Flow-Width Ratio (FWR)')
    ax1.set_ylabel('Number of Reaches')
    ax1.set_title('BEFORE Correction')
    ax1.legend()
    ax1.set_xlim(0, 50000)

    # After
    ax2 = axes[1]
    fwr_after = predictions['fwr_after'].clip(upper=500)
    ax2.hist(fwr_after, bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    ax2.axvline(fwr_after.median(), color='black', linestyle='--', linewidth=2,
                label=f'Median: {fwr_after.median():,.0f}')
    ax2.set_xlabel('Flow-Width Ratio (FWR)')
    ax2.set_ylabel('Number of Reaches')
    ax2.set_title('AFTER Correction')
    ax2.legend()
    ax2.set_xlim(0, 500)

    # Add normal range annotation
    ax2.axvspan(0, 100, alpha=0.2, color='green', label='Normal range')
    ax2.legend()

    plt.suptitle('FWR Distribution: 1,725 Anomalous Reaches', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_fwr_before_after.png', bbox_inches='tight')
    plt.close()
    print(f"Saved fig3_fwr_before_after.png")


def fig4_detection_rules():
    """Pie chart of detection rules breakdown."""
    rules = {
        'fwr_drop': 815,
        'entry_point': 466,
        'extreme_fwr': 200,
        'jump_entry': 99,
        'facc_sum_inflation': 45,
        'Other': 100
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    explode = (0.05, 0, 0, 0, 0, 0)

    wedges, texts, autotexts = ax.pie(
        rules.values(),
        labels=rules.keys(),
        autopct='%1.0f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 10}
    )

    ax.set_title('Detection Rules: How 1,725 Anomalies Were Found', fontsize=12)

    # Legend with descriptions
    legend_labels = [
        'fwr_drop (815): FWR drops >5x downstream',
        'entry_point (466): facc jump + high ratio_to_median',
        'extreme_fwr (200): FWR > 15,000',
        'jump_entry (99): disconnected + high facc',
        'facc_sum_inflation (45): facc > 3x upstream sum',
        'Other (100): additional rules'
    ]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.18),
              fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_detection_rules.png', bbox_inches='tight')
    plt.close()
    print(f"Saved fig4_detection_rules.png")


def fig5_correction_logic():
    """Conceptual diagram showing RF correction logic."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'RF Regressor Correction Logic', fontsize=14, ha='center', fontweight='bold')

    # Box 1: Input
    rect1 = plt.Rectangle((0.5, 4), 2.5, 1.5, facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 5.1, 'Anomalous Reach', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.75, 4.6, 'hydro_dist_hw = 150 km\nwidth = 80 m\nfacc = 2,500,000 km²',
            ha='center', fontsize=8, family='monospace')

    # Arrow 1
    ax.annotate('', xy=(3.5, 4.75), xytext=(3.0, 4.75),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Box 2: RF Model
    rect2 = plt.Rectangle((3.5, 4), 3, 1.5, facecolor='#3498db', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 5.1, 'RF Regressor', ha='center', fontsize=10, fontweight='bold', color='white')
    ax.text(5, 4.5, '"At 150km from headwater,\nfacc should be ~5,000 km²"',
            ha='center', fontsize=8, color='white', style='italic')

    # Arrow 2
    ax.annotate('', xy=(7, 4.75), xytext=(6.5, 4.75),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Box 3: Output
    rect3 = plt.Rectangle((7, 4), 2.5, 1.5, facecolor='#2ecc71', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect3)
    ax.text(8.25, 5.1, 'Corrected Value', ha='center', fontsize=10, fontweight='bold')
    ax.text(8.25, 4.6, 'facc = 5,000 km²\nFWR: 31,250 → 63',
            ha='center', fontsize=8, family='monospace')

    # Training data box
    rect4 = plt.Rectangle((2, 1.5), 6, 1.8, facecolor='#f8f9fa', edgecolor='#95a5a6',
                           linewidth=1, linestyle='--')
    ax.add_patch(rect4)
    ax.text(5, 3, 'Training: 247,000 Clean Reaches', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 2.4, 'Model learns: "reaches at position X typically have facc Y"',
            ha='center', fontsize=9, style='italic')
    ax.text(5, 1.9, 'hydro_dist_hw explains 56% of variance',
            ha='center', fontsize=9, color='#27ae60')

    # Arrow from training to model
    ax.annotate('', xy=(5, 4), xytext=(5, 3.3),
                arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5, linestyle='--'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_correction_logic.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved fig5_correction_logic.png")


def main():
    print("Generating FACC correction figures...")
    print("=" * 50)

    fig1_feature_importance()
    fig2_hydro_dist_vs_facc()
    fig3_fwr_before_after()
    fig4_detection_rules()
    fig5_correction_logic()

    print("=" * 50)
    print(f"All figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
