import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

# --- NEW: 6-Part Hierarchical Skeleton Definition (0-based indexing) ---

# We use 0-based indexing for the 25 NTU joints:
# 0: Base, 1: Mid, 2: Neck, 3: Head, 4: RS, 5: RE, 6: RW, 7: RH, 8: LS, 9: LE, 10: LW, 11: LH,
# 12: RHip, 13: RKnee, 14: RAnkle, 15: RFoot, 16: LHip, 17: LKnee, 18: LAnkle, 19: LFoot,
# 20: SShoulder, 21: RTip, 22: RThumb, 23: LTip, 24: LThumb.

# 1. Left Arm (Includes Shoulder (8) and SpineShoulder (20) for connection/overlap)
p1_left_arm_indices = [8, 9, 10, 11, 23, 24, 20] # LS, LE, LW, LH, LTip, LThumb, SShoulder

# 2. Right Arm (Includes Shoulder (4) and SpineShoulder (20) for connection/overlap)
p2_right_arm_indices = [4, 5, 6, 7, 21, 22, 20] # RS, RE, RW, RH, RTip, RThumb, SShoulder

# 3. Left Leg (Includes Hip (16) and SpineBase (0) for connection/overlap)
p3_left_leg_indices = [16, 17, 18, 19, 0] # LHip, LKnee, LAnkle, LFoot, SBase

# 4. Right Leg (Includes Hip (12) and SpineBase (0) for connection/overlap)
p4_right_leg_indices = [12, 13, 14, 15, 0] # RHip, RKnee, RAnkle, RFoot, SBase

# 5. Torso/Spine (Includes base, mid, neck, and ALL overlap nodes: 4, 8, 12, 16, 20)
p5_torso_indices = [0, 1, 2, 4, 8, 12, 16, 20] # SBase, SMid, Neck, RS, LS, RHip, LHip, SShoulder

# 6. Head/Neck (Includes Neck (2) and Head (3). Neck is also in Torso.)
p6_head_indices = [2, 3] # Neck, Head

# Group them for easy iteration in the model
part_indices = [
    p1_left_arm_indices, p2_right_arm_indices, p3_left_leg_indices,
    p4_right_leg_indices, p5_torso_indices, p6_head_indices
]
# --- END NEW DEFINITION ---

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)

        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]

        # --- NEW: Hierarchical Graph Components ---
        self.part_indices = part_indices
        self.parts_A = []

        # Get A+I (Adjacency + Identity matrix) for subgraph extraction
        full_A_plus_I = self.A_binary + np.eye(self.num_node)

        # Generate Adjacency Matrix for each of the 6 Parts
        for indices in self.part_indices:
            # Use the new helper function
            sub_A = tools.get_subgraph_adjacency(full_A_plus_I, indices)
            self.parts_A.append(sub_A)

        # Generate Adjacency Matrix for the Supernode Graph (6 nodes)
        # Supernodes are ordered: LA, RA, LL, RL, Torso, Head
        # Define connections: Arms->Torso, Legs->Torso, Head->Torso (5->4)
        super_edges = [(0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 5)]  # Bidirectional

        num_supernodes = len(self.part_indices)  # 6
        self.super_A_binary = tools.edge2mat(super_edges, num_supernodes)

        # Create the 3-channel (I, In, Out) format for the Supernode Graph
        self.super_A = tools.get_spatial_graph(
            num_supernodes,
            [(i, i) for i in range(num_supernodes)],  # Self-link
            [(j, i) for i, j in super_edges if i != j],  # Inward (just the directed edges)
            [(i, j) for i, j in super_edges if i != j]  # Outward (just the directed edges)
        )
        # --- END NEW HIERARCHICAL COMPONENTS ---


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
