import torch
import numpy as np
import torch_sparse

from kplex_pool.pool import cover_pool_node

from torch_geometric.utils import degree



def pool_pos(pos: np.ndarray, cover_index: torch.LongTensor, num_clusters=None):
    """Compute the position of the contracted node by averaging the position
    of the nodes within the same cluster.
    
    Args:
        pos (ndarray): Matrix containing the position of every node in the
            Euclidean space.
        cover_index (torch.LongTensor): Cover assignment matrix, in sparse
            coordinate form.
        num_clusters (int, optional): Number of clusters in the cover.
            Defaults to `None`.
    
    Returns:
        ndarray: the positions of the coarsened nodes.
    """
    t_pos = torch.from_numpy(pos.values()).type(torch.float)
    t_pos = cover_pool_node(cover_index, t_pos, num_clusters)

    return t_pos.numpy()

def pool_color(color: np.ndarray, cover_index: torch.LongTensor, num_clusters=None):
    """Compute the color of the contracted node by averaging the position
    of the nodes within the same cluster.
    
    Args:
        color (ndarray): Matrix containing the colors of every node.
        cover_index (torch.LongTensor): Cover assignment matrix, in sparse
            coordinate form.
        num_clusters (int, optional): Number of clusters in the cover.
            Defaults to `None`.
    
    Returns:
        ndarray: the color of the coarsened nodes.
    """
    t_color = torch.from_numpy(color.reshape((-1, 1))).type(torch.float)
    t_color = cover_pool_node(cover_index, t_color, num_clusters)

    return t_color.numpy().flatten()

def count_duplicates(cover_index: torch.LongTensor, normalize=False):
    """Count the number of node repetitions in the cover sets. 
    
    Args:
        cover_index (torch.LongTensor): Cover assignment matrix, in sparse
            coordinate form.
        normalize (bool, optional): Normalize the results with respect to the
            total nodes in the graph. Defaults to `False`.
    
    Returns:
        int or float: Node repetitions.
    """
    num_nodes = cover_index[0].max().item() + 1
    duplicates = cover_index.size(1) - num_nodes

    if normalize:
        duplicates /= num_nodes
    
    return duplicates

def coverage(cover_index_list):
    """Compute the coverage of the nodes in the highest level in a hierarchy
    of graphs in terms of percentage of node covered in the lowest one.
    
    Args:
        cover_index_list (list): List of cover assignment matrices.
    
    Returns:
        ndarray: Coverage of each node in the topmost level of the hierarchy.
    """
    last_idx = cover_index_list[0].clone()
    last_val = torch.ones_like(last_idx[0], dtype=torch.float)
    num_nodes = last_idx[0].max().item() + 1
    num_clusters = last_idx[1].max().item() + 1

    for mat in cover_index_list[1:]:
        dim = mat[1].max().item() + 1
        last_idx, last_val = torch_sparse.spspmm(last_idx, last_val,
                                                 mat, torch.ones_like(mat[0], dtype=torch.float),
                                                 num_nodes, num_clusters, dim)
        num_clusters = dim
    
    last_val = torch.ones_like(last_val)
    last_idx[0] = torch.zeros_like(last_idx[0])
    _, coverage = torch_sparse.coalesce(last_idx, last_val, 1, num_clusters)

    return coverage.numpy()/num_nodes

def node_covering_index(cover_index:torch.LongTensor, distribution=False, num_nodes=None):
    """Compute the node covering index of a given cover matrix, i.e., the
    number of occurrences of each node in the cover matrix.
    
    Args:
        cover_index (torch.LongTensor): Cover assignment matrix in sparse
            coordinate form.
        distribution (bool, optional): If `True`, returns the distribution of
            number of occurrences. Defaults to False.
        num_nodes (int, optional): Number of nodes. Defaults to `None`.
    
    Returns:
        torch.LongTensor: Node covering index, or the distribution of node
            occurrences.
    """
    counts = torch.bincount(cover_index[0], minlength=0 if num_nodes is None else num_nodes)

    if distribution:
        counts = torch.bincount(counts)
    
    return counts

def hub_promotion(cover_index:torch.LongTensor, q=0.95, num_nodes=None, num_clusters=None, batch=None):
    """Promote hub nodes to a singleton cluster in a given covering matrix.
    
    Args:
        cover_index (torch.LongTensor): Cover assignment matrix in sparse
            coordinate form.
        q (float, optional): Quantile threshold. Defaults to 0.95.
        num_nodes (int, optional): Number of nodes in the graph. Defaults to
            `None`.
        num_clusters (int, optional): Number of clusters in the cover. Defaults
            to `None`.
        batch (LongTensor, optional): Batch vector, assigning every node
            to a specific example in the batch. Defaults to `None`.
    
    Returns:
        (torch.LongTensor, int, torch.LongTensor): The modified cover matrix,
            its number of clusters, and the batch vector of its clusters. 
    """
    counts = node_covering_index(cover_index, num_nodes=num_nodes)
    limit = np.quantile(counts.cpu().numpy(), q)
    device = cover_index.device

    if num_nodes is None:
        num_nodes = counts.size(0)
    
    if num_clusters is None:
        num_clusters = cover_index[1].max().item() + 1
    
    mask = counts <= limit
    masked_index = cover_index[:, mask[cover_index[0]]]

    hub_index = (mask == 0).nonzero().view(-1)    
    out_clusters = num_clusters + hub_index.size(0)
    hub_values = torch.arange(
        start=num_clusters,
        end=out_clusters,
        device=device
    )

    out_index = torch.cat([masked_index, torch.stack([hub_index, hub_values])], dim=1)
    out_batch = None if batch is None else batch[out_index[0]] 

    return out_index, out_clusters, out_batch

def add_node_features(dataset):
    """Add degree features to a dataset.
    
    Args:
        dataset (torch_geometric.Dataset): A graph dataset.
    
    Returns:
        torch_geometric.Dataset: The same dataset, with `x` containing the
            degree vector of the nodes. 
    """
    max_degree = 0.
    degrees = []
    slices = [0]

    for data in dataset:
        degrees.append(degree(data.edge_index[0], data.num_nodes, torch.float))
        max_degree = max(max_degree, degrees[-1].max().item())
        slices.append(data.num_nodes)

    dataset.data.x = torch.cat(degrees, dim=0).div_(max_degree).view(-1, 1)
    dataset.slices['x'] = torch.tensor(slices, dtype=torch.long, device=dataset.data.x.device).cumsum(0)

    return dataset
