import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.core.display_functions import display


def create_visualizations(results_df):
    # Set style
    plt.style.use('dark_background')
    sns.set_style("darkgrid", {'axes.facecolor': '#1e1e1e'})

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Spider/Radar Plot for Metrics
    ax1 = fig.add_subplot(221, projection='polar')
    metrics = ['Precision', 'Recall', 'Accuracy', 'F1 Score']
    spellcheckers = results_df.columns

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle

    for spellchecker in spellcheckers:
        values = results_df.loc[metrics, spellchecker].values
        values = np.concatenate((values, [values[0]]))  # complete the circle
        ax1.plot(angles, values, 'o-', linewidth=2, label=spellchecker)
        ax1.fill(angles, values, alpha=0.25)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, color='white')
    ax1.set_ylim(0, 1.0)
    ax1.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.grid(True)
    ax1.set_title('Performance Metrics Comparison', color='white')
    legend = ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    legend.get_frame().set_facecolor('#D3D3D3')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_alpha(0.8)

    # 2. Stacked Bar Chart for Correct vs Incorrect Decisions (in percentages)
    ax2 = fig.add_subplot(222)
    correct_decisions = results_df.loc['True Corrections'] + results_df.loc['True Negatives']
    incorrect_decisions = results_df.loc['False Corrections'] + results_df.loc['False Alarms']
    total_decisions = correct_decisions + incorrect_decisions

    # Calculate percentages
    correct_percentage = (correct_decisions / total_decisions * 100)
    incorrect_percentage = (incorrect_decisions / total_decisions * 100)

    decision_data = pd.DataFrame({
        'Correct Decisions (%)': correct_percentage,
        'Incorrect Decisions (%)': incorrect_percentage
    })

    decision_data.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Correct vs Incorrect Decisions (%)', color='white')
    ax2.set_ylabel('Percentage', color='white')
    ax2.set_ylim(0, 100)  # Set y-axis limit to 100%

    # Add percentage labels on the bars
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.1f%%', label_type='center', color='black')

    plt.yticks(color='white')
    plt.xticks(rotation=45, color='white')
    legend = ax2.legend(bbox_to_anchor=(0.1, 1.1))
    legend.get_frame().set_facecolor('#D3D3D3')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_alpha(0.8)

    # 3. Processing Time Comparison (Log Scale) - maintain consistent order
    ax3 = fig.add_subplot(223)
    time_data = results_df.loc['Time (s)']  # Keep original order

    # Create bar plot with fixed x-axis
    bars = ax3.bar(range(len(time_data)), time_data.values)
    ax3.set_yscale('log')
    ax3.set_title('Processing Time (seconds, log scale)', color='white')

    # Set fixed ticks and labels
    ax3.set_xticks(range(len(time_data)))
    ax3.set_xticklabels(time_data.index, rotation=45, color='white')
    ax3.set_ylabel('Time (seconds)', color='white')
    plt.yticks(color='white')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}s',
                 ha='center', va='bottom', color='white')

    # 4. Detailed Error Analysis
    ax4 = fig.add_subplot(224)
    error_data = pd.DataFrame({
        'True Corrections': results_df.loc['True Corrections'],
        'False Corrections': results_df.loc['False Corrections'],
        'False Alarms': results_df.loc['False Alarms']
    })

    error_data.plot(kind='bar', ax=ax4)
    ax4.set_title('Detailed Error Analysis', color='white')
    ax4.set_ylabel('Number of Cases', color='white')
    plt.yticks(color='white')
    plt.xticks(rotation=45, color='white')
    legend = ax4.legend(bbox_to_anchor=(0.1, 1.1))
    legend.get_frame().set_facecolor('#D3D3D3')
    legend.get_frame().set_edgecolor('white')
    legend.get_frame().set_alpha(0.8)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('results/graphs/spellchecker_analysis.png', dpi=300, bbox_inches='tight',  pad_inches=0.5, facecolor='#1e1e1e')
    plt.close()

    # Create additional visualization: Success Rate Plot with consistent order
    plt.figure(figsize=(12, 6))
    success_rate = (correct_decisions / total_decisions * 100)  # Keep original order

    # Create bar plot
    ax = plt.gca()
    bars = ax.bar(range(len(success_rate)), success_rate.values)

    # Set proper ticks and labels
    ax.set_xticks(range(len(success_rate)))
    ax.set_xticklabels(success_rate.index, rotation=45, color='white')

    plt.title('Overall Success Rate by Spellchecker', color='white')
    plt.xlabel('Spellchecker', color='white')
    plt.ylabel('Success Rate (%)', color='white')
    plt.yticks(color='white')

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', color='white')

    plt.tight_layout()
    plt.savefig('results/graphs/success_rate.png', dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()


# Usage
def visualize_results(csv_path='results/results.csv'):
    # Load results
    results_df = pd.read_csv(csv_path, index_col=0)

    # Create visualizations
    create_visualizations(results_df)

    print("Graphs saved in results/graphs/")

if __name__ == "__main__":
    visualize_results()