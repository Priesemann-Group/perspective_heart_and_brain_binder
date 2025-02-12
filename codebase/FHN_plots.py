import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.colors as colors

import os, sys
##sys.path.append(os.getcwd())
#from .FHN_model import *

projectfolder = "../figures/"
simulation_name = "default"
mpl.rcParams.update({"font.size": 16})



def plot_graph_on_circle(
    y,
    J,
    ax,
    subset_size=50,
    format_axes=True,
    plot_connections=True,
    arrows=True,
    marker_size=200,
    idle_color="white",
    spiking_color="royalblue",
    weighted_connections_cmap=None,
    weighted_connections_norm=None,
):
    """
    Plot the graph on a circle with nodes colored based on their V value.
    Parameters:
        J (scipy.sparse matrix): adjacency matrix
        y (numpy.ndarray): (N) states of the neurons
        subset_size: number of nodes to plot, if 'all' then subset_size=N
    """
    J_dense = J.todense()
    #else:
   
    N = len(y)

    # Select a subset of nodes to plot
    if type(subset_size) == str and subset_size == "all":
        subset_size = N
    subset_indices = np.random.choice(N, subset_size, replace=False)

    # Create positions for the nodes on a circle
    theta = np.linspace(0, 2 * np.pi, subset_size, endpoint=False)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))

    if format_axes:
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")

    # Plot the connections of the subset
    if plot_connections:
        i_indices, j_indices = np.meshgrid(
            subset_indices, subset_indices, indexing="ij"
        )
        mask = J_dense[i_indices, j_indices] != 0
        i_indices, j_indices = i_indices[mask], j_indices[mask]

        for i, j in zip(i_indices, j_indices):
            if type(weighted_connections_cmap) == type(None):
                color = "black"
            else:
                color = weighted_connections_cmap(
                    weighted_connections_norm(J_dense[i, j])
                )
            if arrows:
                ax.arrow(
                    x=positions[i % subset_size, 0],
                    y=positions[i % subset_size, 1],
                    dx=positions[j % subset_size, 0] - positions[i % subset_size, 0],
                    dy=positions[j % subset_size, 1] - positions[i % subset_size, 1],
                    color=color,
                    alpha=1,
                    head_width=0.02,
                    length_includes_head=True,
                )
            else:
                ax.plot(
                    [positions[i % subset_size, 0], positions[j % subset_size, 0]],
                    [positions[i % subset_size, 1], positions[j % subset_size, 1]],
                    "-",
                    color=color,
                    alpha=0.1,
                )
    # Add a colorbar with specific ticks
    subset_y = y[subset_indices]
    
    norm = plt.Normalize(vmin=-np.max(np.abs(subset_y)), vmax=np.max(np.abs(subset_y)))
    cmap = plt.get_cmap("seismic")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('V', rotation=0, labelpad=15, fontsize=16, color='black')
    cbar.set_ticks([-np.max(np.abs(subset_y)), 0,np.max(np.abs(subset_y))])
    cbar.ax.tick_params(labelsize=14)
    colors = cmap(norm(subset_y))
    ax.scatter(
        positions[:, 0], positions[:, 1], c=colors, edgecolors="black", s=marker_size
    )



def heart_plot(
        fig,
        ax,
        model,
        state_index=100,
        cbar_axes=None,
        cbar_shrink=0.6,
        cbar_location="right",
        cbar_orientation="vertical",
        cmap='seismic'

  ):


    array=model.vs.T
# Display the first frame
    array=array.reshape(model.N,model.N,-1)
    array=array[4:model.N-4,4:model.N-4,:]
    img = ax.imshow(array[:,:, state_index], cmap=cmap, interpolation="bilinear", vmin=-np.max(np.abs(array)), vmax=np.max(np.abs(array)))
	
    
# Add a colorbar with specific ticks
    if type(cbar_axes) == type(None):
        cbar_axes = ax
    cbar = fig.colorbar(
        img,
        ax=cbar_axes,
        location=cbar_location,
        shrink=cbar_shrink,
        orientation=cbar_orientation,
    )
    ax.set_aspect('auto')
    cbar.set_label('V', rotation=0, labelpad=15, fontsize=12, color='black')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  
    plt.show()

def plot_kymograph(
    fig,
    ax,
    model,
    plot_from=100,
    cmap="seismic",
    xlabel="time (a.u.)",
    ylabel="Node",
    cbar_axes=None,
    cbar_shrink=0.6,
    cbar_location="top",
    cbar_orientation="horizontal",
):
    # Plot the Kymograph
    if model.organ == "brain":
        im=ax.imshow(
            model.vs[plot_from:, ::-1].T,
            extent=(np.min(model.ts[plot_from:]), np.max(model.ts[plot_from:]), 0, model.N),
            aspect="auto",
            cmap=cmap,
            interpolation="none",
            vmin=-np.max(np.abs(model.vs)),
            vmax=np.max(np.abs(model.vs))
        )
    if model.organ == "heart":
        array=model.vs.reshape(len(model.ts), model.N, model.N)
        a=int(model.N/2)
        im=ax.imshow(
            array[plot_from:, a, :].T,
            extent=(np.min(model.ts[plot_from:]), np.max(model.ts[plot_from:]), 0, model.N),
            aspect="auto",
            cmap=cmap,
            interpolation="none",
            vmin=-np.max(np.abs(model.vs)),
            vmax=np.max(np.abs(model.vs))

        )
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
   
    ax.set_xticks([])
    ax.set_yticks([])
    # plot colorbar
    if type(cbar_axes) == type(None):
        cbar_axes = ax
    cbar = fig.colorbar(
        im,
        ax=cbar_axes,
        location=cbar_location,
        shrink=cbar_shrink,
        orientation=cbar_orientation,
    )
    cbar.set_label(r'$V_i$', fontsize=18)
    if cbar_orientation == "horizontal":
        cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    else:
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
   
    return im, cbar


def collective_signal(
    fig,
    ax,
    model,
    plot_from=1000
):
    array=model.vs.T
    #here the y scale is set to match the one of the heart simulation

    EEG=array[ :, plot_from:].sum(axis=0)/10000
    
    if model.organ=='brain':
        
        ax.plot(model.ts[plot_from:],EEG, color='darkblue')
    elif model.organ=='heart':
        ax.plot( model.ts[plot_from:],EEG, color='darkred')
    
    ax.set_ylim([-0.09,0.24])
    ax.set_xlim(model.ts[plot_from], model.ts[-1])
    ax.set_aspect('auto')
    ax.set_ylabel('colective signal (a.u.)', fontsize=18)
    ax.set_xlabel('time (a.u.)', fontsize=18)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax.set_xticks([])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def plot_kymograph_and_collective_signal(
    fig,
    ax_kymograph,
    ax_signal,
    model,
    plot_from=1000,
    cmap="seismic",
    xlabel="time (a.u.)",
    ylabel="Node",
    cbar_axes=None,
    cbar_shrink=0.6,
    cbar_location="top",
    cbar_orientation="horizontal",
):
    # Plot the Kymograph
    im, cbar = plot_kymograph(
        fig,
        ax_kymograph,
        model,
        plot_from=plot_from,
        cmap=cmap,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_axes=cbar_axes,
        cbar_shrink=cbar_shrink,
        cbar_location=cbar_location,
        cbar_orientation=cbar_orientation,
    )
    # Plot the collective signal
    collective_signal(fig, ax_signal, model, plot_from=plot_from)
    
    return im, cbar