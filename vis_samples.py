import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap

LETTERS = ['A', 'B', 'C', 'D', 'E']
# Define the colormap
red_to_green_cmap = LinearSegmentedColormap.from_list(
    "red_to_green", [(1, 0, 0), (1, 1, 1), (0, 1, 0)]  # Red to white to green
)

# Function to normalize and map values
def normalize_and_map(values, cmap, vmin=-1, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    return [cmap(norm(val), alpha=0.7) for val in values]

def highlight_steps(question, options, reasoning_steps, correct, predicted, salience_scores, line_width=60):
    """
    Displays a question, answer options, reasoning steps, and highlights the text of steps based on salience.
    
    Args:
        question (str): The question text.
        options (list of str): List of answer options.
        reasoning_steps (list of str): List of reasoning steps.
        salience_scores (list of float): Salience scores for the reasoning steps, in the range [0, 1].
    """
    # Ensure salience_scores are normalized between 0 and 1
    assert len(reasoning_steps) == len(salience_scores), "Each reasoning step must have a salience score."
    
    # Create a colormap for salience highlighting
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, len(reasoning_steps) * 0.3 + 3))
    ax.axis('off')
    
    # Display the question
    y_pos = 1.0
    wrapped_question = "\n".join(wrap(question, line_width))
    ax.text(0.01, y_pos, f"Question: {wrapped_question}", fontsize=12, wrap=True) # , weight="bold"
    y_pos -= 0.05  # Move down slightly

    # Display answer options in a single row
    x_pos = 0.01
    col_width = 1.0 / len(options)  # Split the width evenly among columns
    for i, option in enumerate(options):
        option_lines = wrap(option, line_width)
        n_lines = len(option_lines)
        wrapped_option = "\n".join(option_lines)
        if i == correct:
            ax.text(
                0.01, y_pos, wrapped_option,
                fontsize=11, ha='left', wrap=True,
                bbox=dict(facecolor='green', edgecolor='none')
            )
        elif i == predicted:
            ax.text(
                0.01, y_pos, wrapped_option,
                fontsize=11, ha='left', wrap=True,
                bbox=dict(facecolor='blue', edgecolor='none')
            )
        else:
            ax.text(0.01, y_pos, wrapped_option, fontsize=11, ha="left", wrap=True)
        
        y_pos -= 0.048 * n_lines # Move down for each option

    salience_colors = normalize_and_map(salience_scores, red_to_green_cmap)
    # Display reasoning steps with text highlighting based on salience
    for i, (step, salience_color) in enumerate(zip(reasoning_steps, salience_colors)):

        step_lines = wrap(step, line_width)
        n_lines = len(step_lines)
        wrapped_step = "\n".join(step_lines)
        ax.text(
            0.01, y_pos, wrapped_step,
            fontsize=10, va="top", wrap=True,
            bbox=dict(facecolor=salience_color, edgecolor='none') # , pad=0.5
        )
        y_pos -= 0.05 * n_lines  # Decrease vertical space between reasoning steps

    # Add colorbar
    sm = ScalarMappable(cmap=red_to_green_cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, shrink=0.3)
    cbar.set_label("Salience Score", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()
