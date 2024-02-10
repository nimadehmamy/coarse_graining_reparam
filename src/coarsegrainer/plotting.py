
# implement the plot_graph function using py3Dmol

import py3Dmol
from matplotlib.colors import to_hex
from matplotlib import colormaps

import matplotlib.pyplot as plt
import numpy as np
import torch

V = lambda x: x.detach().cpu().numpy()


# function to plot a graph of nodes at positions x with adjacency matrix A
def plot_graph(x, A, ax=None, node_radius=10, edge_thickness=1, eps=1e-4, edge_color='k', ls='-', alpha=1):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if type(A) is torch.Tensor:
        A = V(A)
    if type(x) is torch.Tensor:
        x = V(x)
    ax.plot(x[:, 0], x[:, 1], 'o', ms=node_radius, zorder = 100, alpha=alpha)
    idx = np.where(np.abs(A) >= eps)
    for i, j in zip(idx[0], idx[1]):
        ax.plot([x[i, 0], x[j, 0]], [x[i, 1], x[j, 1]], 
                color=edge_color, ls=ls, lw=edge_thickness)
    # ax.set_xticks([])
    # ax.set_yticks([])
    return ax


def plot_graph_3D(x, A, node_radius=.5, edge_thickness=.2, eps=1e-4, 
                edge_color='#FFFFFF',
                node_color='blue',
                alpha=1,
                width=600, height=600
                ):
    view = py3Dmol.view(width=width, height=height)
    # Add cylinders along the edges in A and spheres at the nodes
    # 1. get the indices of the edges in A
    if type(A) is torch.Tensor:
        A = V(A)
    if type(x) is torch.Tensor:
        x = V(x)
    idx = np.where(np.abs(A) >= eps)
    x = np.float64(x) # convert to float64
    for i, j in zip(idx[0], idx[1]):
        start = {'x': x[i,0], 'y': x[i,1], 'z': x[i,2]}
        end = {'x': x[j,0], 'y': x[j,1], 'z': x[j,2]}
        view.addCylinder({'start': start, 'end': end, 'radius': edge_thickness/2, 'color': edge_color, 'opacity':alpha})

    for p in x:
        # add spheres at the nodes
        view.addSphere({'center': {'x': p[0], 'y': p[1], 'z': p[2]}, 'radius': node_radius, 'color': node_color, 'opacity':alpha})

    view.setStyle({'stick': {}})
    # zoom to fit
    view.zoomTo()
    view.show()
    return view


def plot_line_3D(x, A, r = 5e-1, eps=1e-4, 
                colormap='jet',
                alpha=1,
                width=600, height=600
                ):
    node_radius=r/2
    edge_thickness=r
    # make a rainbow colored version of the 3d plot using py3Dmol
    view = py3Dmol.view(width=width, height=height)
    # Add cylinders along the edges in A and spheres at the nodes
    # 1. get the indices of the edges in A
    if type(A) is torch.Tensor:
        A = V(A)
    if type(x) is torch.Tensor:
        x = V(x)
    idx = np.where(np.abs(A) >= eps)
    x = np.float64(x) # convert to float64
    # instead of using indices, since we know we have aline graph
    # we can just use arange(len(x)-1) to get the indices
    idx = (np.arange(len(x)-1), np.arange(1, len(x)))
    for i, j in zip(idx[0], idx[1]):
        start = {'x': x[i,0], 'y': x[i,1], 'z': x[i,2]}
        end = {'x': x[j,0], 'y': x[j,1], 'z': x[j,2]}
        # choose the edge_color as a rainbow using jet colormap
        # use the given 'colormap' to choose the color
        edge_color = colormaps[colormap](i/len(x))
        # edge_color = plt.cm.jet(i/len(x))
        # since py3dmol uses colors of format "#RRGGBB" we need to convert the color to this format
        edge_color = to_hex(edge_color)
        view.addCylinder({'start': start, 'end': end, 'radius': edge_thickness/2, 'color': edge_color, 'opacity':alpha})

    for i,p in enumerate(x):
        # add spheres at the nodes
        # similarly, we can use the jet colormap to color the nodes
        node_color = colormaps[colormap](i/len(x))
        node_color = to_hex(node_color)
        view.addSphere({'center': {'x': p[0], 'y': p[1], 'z': p[2]}, 'radius': node_radius, 'color': node_color, 'opacity':alpha})

    view.setStyle({'stick': {}})
    # zoom to fit
    view.zoomTo()
    view.show()
    return view