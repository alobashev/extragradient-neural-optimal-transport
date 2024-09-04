import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_generated_3D(sample_list, fig_size=(15, 5.4), dpi=150):
    """
    Plots a 3D scatter plot for each tensor in a list.
    
    Parameters:
    - sample_list: List of tensors. Each tensor should have shape (N, 3).
    - fig_size: Tuple specifying the size of the figure.
    - dpi: Dots per inch for the figure.
    
    Returns:
    - fig, axes: The figure and axes objects.
    """
    num_plots = len(sample_list)
    if num_plots == 0:
        raise ValueError("The sample list is empty.")
    
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    axes = fig.subplots(1, num_plots, subplot_kw={'projection': '3d'}, sharex=True, sharey=True)

    # Ensure axes is always iterable (even if there's only one plot)
    if num_plots == 1:
        axes = [axes]
    
    # Calculate the global range for all samples
    all_samples = torch.cat(sample_list, dim=0)
    min_range = all_samples.min(dim=0).values.cpu().numpy()
    max_range = all_samples.max(dim=0).values.cpu().numpy()
    
    # Set up the axis limits and equal aspect ratio for all axes
    for ax in axes:
        ax.set_xlim(min_range[0], max_range[0])
        ax.set_ylim(min_range[1], max_range[1])
        ax.set_zlim(min_range[2], max_range[2])
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.grid(True)

    # Plot each tensor in the list
    colors = ['darkseagreen', 'peru', 'wheat', 'skyblue', 'lightcoral']  # Adjust as needed
    for i, (samples, ax) in enumerate(zip(sample_list, axes)):
        samples_np = samples.cpu().numpy()
        ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], c=colors[i % len(colors)], edgecolors='black')
        ax.set_title(f'Plot {i + 1}', fontsize=22, pad=10)

    fig.tight_layout()
    return fig, axes
