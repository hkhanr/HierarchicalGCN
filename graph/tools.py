import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_subgraph_adjacency(full_A_binary, indices):
    """
    Extracts the adjacency matrix for a subgraph defined by a list of node indices.

    Args:
        full_A_binary (np.ndarray): The 25x25 binary adjacency matrix (A + I).
        indices (list): The list of 0-based joint indices for the subgraph.

    Returns:
        np.ndarray: The V_part x V_part adjacency matrix for the subgraph.
    """
    # Filter rows and columns based on the indices
    sub_A = full_A_binary[np.ix_(indices, indices)]

    # Normalize the subgraph adjacency matrix (as done in get_spatial_graph)
    # We will use the same spatial partitioning (I, In, Out) as the original GCN

    # 1. Self-link: I
    I = np.eye(len(indices))

    # 2. Inward/Outward links: This is complex to extract perfectly from an index list.
    # For simplicity and to fit the unit_gcn input (which expects 3 partitions),
    # we'll use a simplified binary adjacency (A+I) and let unit_gcn's internal
    # mechanics (k-hop, rpe) handle the local structure.

    # If the sub_A matrix is binary (1 for connection, 0 otherwise):
    A_binary = (sub_A > 0).astype(np.float32)

    # We create the simplified 3-part adjacency matrix (I, In, Out) expected by the original GCN unit
    # Since we can't perfectly separate In/Out for an arbitrary index list, we use the
    # original method's output (normalized binary) for all 3:
    In = normalize_digraph(A_binary - np.eye(A_binary.shape[0]))  # Inward (excluding self-link)
    Out = normalize_digraph((A_binary - np.eye(A_binary.shape[0])).T)  # Outward (transpose)

    A_subgraph = np.stack((I, In, Out))
    return A_subgraph