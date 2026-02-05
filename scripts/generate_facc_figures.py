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

    fig, ax = plt.subplots(figsize=(8, 5))

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
                f'{val*100:.1f}%', va='center', fontsize=8)

    # Highlight hydro_dist_hw - position annotation better
    ax.annotate('Dijkstra distance from\nfurthest headwater',
                xy=(56, 0.3), xytext=(35, 2.5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                fontsize=9, color='#27ae60', fontweight='bold')

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

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot clean reaches
    ax.scatter(clean['hydro_dist_hw'] / 1000, clean['facc'],
               alpha=0.3, s=3, c='#3498db', label=f'Clean reaches (n={len(clean):,})')

    # Plot anomalies
    ax.scatter(anom['hydro_dist_hw'] / 1000, anom['facc'],
               alpha=0.7, s=15, c='#e74c3c', label=f'Anomalies (n={len(anom):,})')

    ax.set_xlabel('hydro_dist_hw: Dijkstra Distance from Headwater (km)')
    ax.set_ylabel('Flow Accumulation (km²)')
    ax.set_yscale('log')
    ax.set_title('FACC vs Network Position')

    # Legend in upper left to avoid overlap
    ax.legend(loc='upper left')

    # Add annotation - position in upper area where anomalies cluster
    ax.annotate('Anomalies: facc 10-1000x\ntoo high for position',
                xy=(150, 2e6), xytext=(250, 5e6),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                fontsize=9, color='#c0392b', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add explanation - move to top right area to avoid overlap with legend
    ax.text(0.98, 0.98, 'facc accumulates downstream\n→ increases with hydro_dist_hw',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

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

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Before
    ax1 = axes[0]
    fwr_before = predictions['fwr_before'].clip(upper=50000)
    ax1.hist(fwr_before, bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    ax1.axvline(fwr_before.median(), color='black', linestyle='--', linewidth=2,
                label=f'Median: {fwr_before.median():,.0f}')
    ax1.set_xlabel('Flow-Width Ratio (FWR)')
    ax1.set_ylabel('Number of Reaches')
    ax1.set_title('BEFORE Correction')
    ax1.legend(fontsize=8)
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
    ax2.set_xlim(0, 500)

    # Add normal range annotation
    ax2.axvspan(0, 100, alpha=0.2, color='green', label='Normal range')
    ax2.legend(fontsize=8)

    plt.suptitle('FWR Distribution: 1,725 Anomalous Reaches', fontsize=11, y=1.02)
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

    fig, ax = plt.subplots(figsize=(5, 4))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    explode = (0.05, 0, 0, 0, 0, 0)

    wedges, texts, autotexts = ax.pie(
        rules.values(),
        labels=rules.keys(),
        autopct='%1.0f%%',
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={'fontsize': 8}
    )

    ax.set_title('Detection Rules: 1,725 Anomalies', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_detection_rules.png', bbox_inches='tight')
    plt.close()
    print(f"Saved fig4_detection_rules.png")


def fig5_correction_logic():
    """Conceptual diagram showing RF correction logic."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'RF Regressor Correction Logic', fontsize=12, ha='center', fontweight='bold')

    # Box 1: Input
    rect1 = plt.Rectangle((0.5, 4), 2.5, 1.5, facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 5.1, 'Anomalous Reach', ha='center', fontsize=9, fontweight='bold')
    ax.text(1.75, 4.55, 'hydro_dist_hw = 150 km\nwidth = 80 m\nfacc = 2,500,000 km²',
            ha='center', fontsize=7, family='monospace')

    # Arrow 1
    ax.annotate('', xy=(3.5, 4.75), xytext=(3.0, 4.75),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Box 2: RF Model
    rect2 = plt.Rectangle((3.5, 4), 3, 1.5, facecolor='#3498db', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect2)
    ax.text(5, 5.1, 'RF Regressor', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(5, 4.5, '"At 150km from headwater,\nfacc should be ~5,000 km²"',
            ha='center', fontsize=7, color='white', style='italic')

    # Arrow 2
    ax.annotate('', xy=(7, 4.75), xytext=(6.5, 4.75),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Box 3: Output
    rect3 = plt.Rectangle((7, 4), 2.5, 1.5, facecolor='#2ecc71', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(rect3)
    ax.text(8.25, 5.1, 'Corrected Value', ha='center', fontsize=9, fontweight='bold')
    ax.text(8.25, 4.55, 'facc = 5,000 km²\nFWR: 31,250 → 63',
            ha='center', fontsize=7, family='monospace')

    # Training data box
    rect4 = plt.Rectangle((2, 1.5), 6, 1.8, facecolor='#f8f9fa', edgecolor='#95a5a6',
                           linewidth=1, linestyle='--')
    ax.add_patch(rect4)
    ax.text(5, 3, 'Training: 247,000 Clean Reaches', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 2.4, 'Model learns: "reaches at position X typically have facc Y"',
            ha='center', fontsize=8, style='italic')
    ax.text(5, 1.9, 'hydro_dist_hw (Dijkstra from headwater) explains 56%',
            ha='center', fontsize=8, color='#27ae60')

    # Arrow from training to model
    ax.annotate('', xy=(5, 4), xytext=(5, 3.3),
                arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5, linestyle='--'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_correction_logic.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved fig5_correction_logic.png")


def schematic_fwr_drop():
    """Schematic: fwr_drop detection rule."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5, 5.7, 'fwr_drop: FWR drops >5x at downstream boundary', fontsize=10,
            ha='center', fontweight='bold')

    # Bad reach (upstream)
    rect1 = plt.Rectangle((1, 3), 3, 1.5, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 4.1, 'Anomalous Reach', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(2.5, 3.5, 'FWR = 15,000', ha='center', fontsize=9, family='monospace', color='white')

    # Arrow
    ax.annotate('', xy=(5.5, 3.75), xytext=(4.2, 3.75),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.text(4.85, 4.3, 'flow', ha='center', fontsize=7, style='italic')

    # Good reach (downstream)
    rect2 = plt.Rectangle((6, 3), 3, 1.5, facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 4.1, 'Normal Reach', ha='center', fontsize=8, fontweight='bold')
    ax.text(7.5, 3.5, 'FWR = 50', ha='center', fontsize=9, family='monospace')

    # Ratio indicator
    ax.annotate('', xy=(2.5, 2.7), xytext=(7.5, 2.7),
                arrowprops=dict(arrowstyle='<->', color='#8e44ad', lw=2))
    ax.text(5, 2.2, 'Ratio: 15,000 / 50 = 300x > 5x', ha='center', fontsize=9,
            color='#8e44ad', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#8e44ad'))

    # Explanation
    ax.text(5, 0.8, 'Bad facc doesn\'t propagate everywhere. When flow\nrejoins correct MERIT path, FWR returns to normal.',
            ha='center', fontsize=8, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'schematic_fwr_drop.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved schematic_fwr_drop.png")


def schematic_entry_point():
    """Schematic: entry_point detection rule."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.7, 'entry_point: facc jump + ratio_to_median > 40', fontsize=10,
            ha='center', fontweight='bold')

    # Upstream reach (normal)
    rect1 = plt.Rectangle((1, 4), 2.5, 1.2, facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.25, 4.85, 'Upstream', ha='center', fontsize=8, fontweight='bold')
    ax.text(2.25, 4.35, 'facc = 5,000', ha='center', fontsize=8, family='monospace')

    # Arrow
    ax.annotate('', xy=(4.5, 4.6), xytext=(3.7, 4.6),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Bad reach (facc jumped)
    rect2 = plt.Rectangle((5, 4), 2.5, 1.2, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect2)
    ax.text(6.25, 4.85, 'Anomaly', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(6.25, 4.35, 'facc = 500,000', ha='center', fontsize=8, family='monospace', color='white')

    # Jump indicator
    ax.annotate('', xy=(6.25, 3.8), xytext=(2.25, 3.8),
                arrowprops=dict(arrowstyle='<->', color='#e67e22', lw=2))
    ax.text(4.25, 3.3, 'Jump: 500K / 5K = 100x > 10x', ha='center', fontsize=8,
            color='#e67e22', fontweight='bold')

    # Width comparison box
    rect3 = plt.Rectangle((1.5, 1.2), 7, 1.5, facecolor='#f8f9fa', edgecolor='#95a5a6',
                           linewidth=1, linestyle='--')
    ax.add_patch(rect3)
    ax.text(5, 2.35, 'But width is only 80m (narrow side channel)', ha='center', fontsize=8)
    ax.text(5, 1.75, 'ratio_to_median = FWR / regional_median = 6,250 / 50 = 125x > 40',
            ha='center', fontsize=8, color='#c0392b', fontweight='bold')

    # Key insight
    ax.text(5, 0.5, 'facc jump alone isn\'t enough - a tributary joining mainstem\nalso jumps. The FWR ratio distinguishes real anomalies.',
            ha='center', fontsize=7, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'schematic_entry_point.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved schematic_entry_point.png")


def schematic_extreme_fwr():
    """Schematic: extreme_fwr detection rule."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(5, 5.7, 'extreme_fwr: FWR > 15,000 (physically impossible)', fontsize=10,
            ha='center', fontweight='bold')

    # Side channel box
    rect1 = plt.Rectangle((2, 2.5), 6, 2.5, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect1)
    ax.text(5, 4.4, 'Narrow Side Channel', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(5, 3.7, 'width = 100 m', ha='center', fontsize=9, family='monospace', color='white')
    ax.text(5, 3.1, 'facc = 2,500,000 km² (got mainstem value)', ha='center',
            fontsize=8, family='monospace', color='white')

    # FWR calculation
    ax.text(5, 2.0, 'FWR = 2,500,000 / 100 = 25,000 > 15,000', ha='center',
            fontsize=10, color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#c0392b'))

    # Comparison
    ax.text(5, 0.8, 'For reference: Amazon FWR ≈ 2,000\nNo natural river exceeds 15,000 km²/m',
            ha='center', fontsize=8, style='italic', color='#555')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'schematic_extreme_fwr.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved schematic_extreme_fwr.png")


def schematic_jump_entry():
    """Schematic: jump_entry detection rule."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(5, 6.2, 'jump_entry: Disconnected reach with high facc', fontsize=10,
            ha='center', fontweight='bold')

    # Main network (connected)
    ax.text(2.5, 5.2, 'Main Network', ha='center', fontsize=8, color='#27ae60', fontweight='bold')
    for y in [4.5, 4.0, 3.5]:
        rect = plt.Rectangle((1.5, y-0.15), 2, 0.3, facecolor='#2ecc71', edgecolor='#27ae60', linewidth=1)
        ax.add_patch(rect)
    # Arrows between connected reaches
    for y in [4.35, 3.85]:
        ax.annotate('', xy=(2.5, y-0.2), xytext=(2.5, y+0.15),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1))
    ax.text(2.5, 2.9, 'path_freq = 15', ha='center', fontsize=7, family='monospace')

    # Disconnected side channel
    ax.text(7.5, 5.2, 'Disconnected', ha='center', fontsize=8, color='#e74c3c', fontweight='bold')
    rect2 = plt.Rectangle((6.5, 4.0), 2, 0.8, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect2)
    ax.text(7.5, 4.55, 'path_freq = -9999', ha='center', fontsize=7, family='monospace', color='white')
    ax.text(7.5, 4.15, 'facc = 500,000', ha='center', fontsize=7, family='monospace', color='white')

    # X mark showing disconnection
    ax.plot([5.5, 6.3], [4.5, 4.1], 'r-', lw=2)
    ax.plot([5.5, 6.3], [4.1, 4.5], 'r-', lw=2)
    ax.text(5.9, 4.8, 'no link', ha='center', fontsize=7, color='#c0392b')

    # Detection box
    rect3 = plt.Rectangle((2, 1.3), 6, 1.2, facecolor='#f8f9fa', edgecolor='#8e44ad',
                           linewidth=1)
    ax.add_patch(rect3)
    ax.text(5, 2.15, 'path_freq ≤ 0  AND  facc_jump > 5  AND  facc > 1000', ha='center',
            fontsize=8, color='#8e44ad', fontweight='bold', family='monospace')
    ax.text(5, 1.6, 'If disconnected, shouldn\'t have accumulated drainage', ha='center',
            fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'schematic_jump_entry.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved schematic_jump_entry.png")


def schematic_facc_sum_inflation():
    """Schematic: facc_sum_inflation detection rule."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.7, 'facc_sum_inflation: facc > 3x upstream sum at confluence', fontsize=10,
            ha='center', fontweight='bold')

    # Tributary 1 (left)
    rect1 = plt.Rectangle((1, 4.5), 2, 0.8, facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2, 5.1, 'Trib 1', ha='center', fontsize=8, fontweight='bold')
    ax.text(2, 4.7, 'facc = 100K', ha='center', fontsize=7, family='monospace')

    # Tributary 2 (right)
    rect2 = plt.Rectangle((7, 4.5), 2, 0.8, facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
    ax.add_patch(rect2)
    ax.text(8, 5.1, 'Trib 2', ha='center', fontsize=8, fontweight='bold')
    ax.text(8, 4.7, 'facc = 50K', ha='center', fontsize=7, family='monospace')

    # Arrows to confluence
    ax.annotate('', xy=(4.5, 3.5), xytext=(2.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.annotate('', xy=(5.5, 3.5), xytext=(7.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Confluence reach (bad)
    rect3 = plt.Rectangle((3.5, 2.5), 3, 1, facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
    ax.add_patch(rect3)
    ax.text(5, 3.25, 'Confluence Reach', ha='center', fontsize=8, fontweight='bold', color='white')
    ax.text(5, 2.75, 'facc = 500K', ha='center', fontsize=8, family='monospace', color='white')

    # Sum comparison
    ax.text(5, 1.8, 'Expected: 100K + 50K = 150K', ha='center', fontsize=9, color='#27ae60')
    ax.text(5, 1.3, 'Actual: 500K', ha='center', fontsize=9, color='#c0392b', fontweight='bold')
    ax.text(5, 0.7, 'Ratio: 500K / 150K = 3.3x > 3x', ha='center', fontsize=9,
            color='#8e44ad', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#8e44ad'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'schematic_facc_sum_inflation.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved schematic_facc_sum_inflation.png")


def main():
    print("Generating FACC correction figures...")
    print("=" * 50)

    fig1_feature_importance()
    fig2_hydro_dist_vs_facc()
    fig3_fwr_before_after()
    fig4_detection_rules()
    fig5_correction_logic()

    # Detection rule schematics
    print("\nGenerating detection rule schematics...")
    schematic_fwr_drop()
    schematic_entry_point()
    schematic_extreme_fwr()
    schematic_jump_entry()
    schematic_facc_sum_inflation()

    print("=" * 50)
    print(f"All figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
