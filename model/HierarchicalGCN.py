# BlockGCN/model/HierarchicalGCN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# Import necessary components from the original BlockGCN.py file
from .BlockGCN import import_class, TCN_GCN_unit, bn_init

EMBEDDING_CHANNELS = 64 # Use 64 as per our design


class CTR_SuperGraph(nn.Module):
    def __init__(self, in_channels, num_nodes=6, reduction=4):
        super(CTR_SuperGraph, self).__init__()
        self.num_nodes = num_nodes
        inter_channels = in_channels // reduction

        # Two linear transformations to generate the channel-wise queries and keys
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, inter_channels, 1)

        # Transforms the correlation back to the original channel size
        self.conv3 = nn.Conv2d(inter_channels, in_channels, 1)

        # A learnable baseline topology (like your original super_A)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x is your Supergraph features: (B, C, T, V_super=6)
        B, C, T, V = x.size()

        # 1. Compress time to get the global state of each node for the current clip
        # Shape: (B, C, 1, V)
        x_pool = x.mean(dim=2, keepdim=True)

        # 2. Generate Queries and Keys
        q = self.conv1(x_pool)  # (B, C/4, 1, V)
        k = self.conv2(x_pool)  # (B, C/4, 1, V)

        # 3. Calculate Channel-wise Topology (Correlation between nodes)
        # We use einsum to do batch matrix multiplication across the spatial nodes
        # q: (B, C/4, 1, V) -> V is node 'i'
        # k: (B, C/4, 1, V) -> V is node 'j'
        # Result: (B, C/4, V, V) - A unique 6x6 graph for each compressed channel!
        correlation = torch.einsum('b c t i, b c t j -> b c i j', q, k)

        # 4. Expand back to full channels and apply Tanh to bound the weights
        correlation = self.conv3(correlation)  # (B, C, V, V)
        correlation = torch.tanh(correlation)

        # 5. Apply this dynamic, channel-specific graph to the original features
        # x: (B, C, T, V) @ correlation: (B, C, V, V) -> (B, C, T, V)
        x_out = torch.einsum('b c t i, b c i j -> b c t j', x, correlation)

        # Add residual with a learnable scale
        return x + self.alpha * x_out


class ECABlock(nn.Module):
    """Ultra-lightweight Channel Attention (Costs < 5 parameters)"""

    def __init__(self, kernel_size=3):
        super(ECABlock, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, T, V)
        b, c, t, v = x.size()

        # Spatial-Temporal pooling to get channel weights: (B, C)
        y = x.mean(dim=(2, 3))

        # Apply 1D conv across channels: (B, 1, C)
        y = self.conv(y.view(b, 1, c))

        # Scale the original input
        y = self.sigmoid(y.view(b, c, 1, 1))
        return x * y.expand_as(x)

class DynamicASR(nn.Module):
    """Per-sample 6x6 routing from temporally pooled supernode tokens."""

    def __init__(self, channels=EMBEDDING_CHANNELS, num_nodes=6, attn_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.attn_dim = attn_dim
        self.query = nn.Linear(channels, attn_dim, bias=False)
        self.key = nn.Linear(channels, attn_dim, bias=False)
        self.scale = attn_dim ** -0.5

    def forward(self, x):
        # x: (B, C, T, V) with V = num_nodes
        tokens = x.mean(dim=2).transpose(1, 2)  # (B, V, C)
        q = self.query(tokens)  # (B, V, D)
        k = self.key(tokens)    # (B, V, D)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return torch.softmax(logits, dim=-1)  # (B, V, V)


class HierarchicalGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 # ADD THESE THREE FLAGS:
                 use_asr=True, 
                 use_tsm=True, 
                 use_eca=True,
                 use_dynamic_asr=False,
                 **kwargs):
        super().__init__()

        # --- SAVE ABLATION FLAGS ---
        self.use_asr = use_asr
        self.use_tsm = use_tsm
        self.use_eca = use_eca
        self.use_dynamic_asr = use_dynamic_asr

        print(
            'HierarchicalGCN initialized with ASR:', self.use_asr,
            'DynamicASR:', self.use_dynamic_asr,
            'TSM:', self.use_tsm,
            'ECA:', self.use_eca,
        )


        # Load Graph (which now contains part_indices and super_A)
        if graph is None:
            raise ValueError()
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person

        # --- Preprocessing & Embedding ---
        # The original BlockGCN uses a large BN layer. We keep the basic structure.
        self.data_bn = nn.BatchNorm1d(num_person * EMBEDDING_CHANNELS * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, EMBEDDING_CHANNELS)  # Reduced from 128 to 64 for local
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, EMBEDDING_CHANNELS))

        # --- 1. Local Part Encoders (6 Parallel Branches) ---
        self.part_encoders = nn.ModuleList()
        self.part_attentions = nn.ModuleList()  # NEW: Attention for each part
        local_channels = 64

        for i, sub_A_np in enumerate(self.graph.parts_A):
            # sub_A is (3, V_part, V_part). We create a tensor for the GCN unit.
            # sub_A = torch.tensor(sub_A_np, dtype=torch.float32, requires_grad=False)

            # 4-block Part Encoder with constant 64 channels (no temporal downsampling)
            encoder = nn.Sequential(
                TCN_GCN_unit(local_channels, local_channels, sub_A_np, stride=1),
                TCN_GCN_unit(local_channels, local_channels, sub_A_np, stride=1),
                TCN_GCN_unit(local_channels, local_channels, sub_A_np, stride=1),
                TCN_GCN_unit(local_channels, local_channels, sub_A_np, stride=1),
            )
            self.part_encoders.append(encoder)




        # --- 2. Global Supernode Encoder ---
        # super_A = torch.tensor(self.graph.super_A, dtype=torch.float32, requires_grad=False)
        super_A = self.graph.super_A
        if torch.is_tensor(super_A):
            super_A_np = super_A.numpy()
        else:
            super_A_np = super_A

        # NEW: Adaptive Cross-Part Graph (Learnable relations between the 6 parts)
        # Initialized to zeros so it slowly learns to add new connections over time
        self.adaptive_super_A = nn.Parameter(torch.zeros(6, 6))

        global_in_channels = local_channels  # 64

        # 5-block Global Network (includes two temporal downsamples)
        self.global_net = nn.Sequential(
            # Block 1: Channels: 64 -> 128, Stride: 2 (64 -> 32 frames)
            TCN_GCN_unit(global_in_channels, 128, super_A_np, stride=2),
            # Block 2: Channels: 128 -> 128, Stride: 1
            TCN_GCN_unit(128, 128, super_A_np, stride=1),
            # Block 3: Channels: 128 -> 256, Stride: 2 (32 -> 16 frames)
            TCN_GCN_unit(128, 256, super_A_np, stride=2),
            # Block 4: Channels: 256 -> 256, Stride: 1
            TCN_GCN_unit(256, 256, super_A_np, stride=1),
            # # Block 5: Channels: 256 -> 256, Stride: 1
            # TCN_GCN_unit(256, 256, super_A_np, stride=1),
            # Block 6: Channels: 256 -> 256, Stride: 1
            TCN_GCN_unit(256, 256, super_A_np, stride=1)
        )

        # --- 3. Output Head ---
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        # self.ctr_super = CTR_SuperGraph(in_channels=64, num_nodes=6)
        # Create a 6x6 learnable adjacency matrix for the Supergraph
        # Initialize with small random weights so it learns slowly
        # Conditionally initialize ASR matrix
        if self.use_asr:
            self.super_learnable_adj = nn.Parameter(torch.randn(6, 6) * 0.01)
        if self.use_dynamic_asr:
            # Requires static ASR residual path; keep a shared A_asr when use_asr is on.
            if not self.use_asr:
                self.super_learnable_adj = nn.Parameter(torch.randn(6, 6) * 0.01)
            self.dynamic_asr = DynamicASR(
                channels=local_channels, num_nodes=6, attn_dim=32
            )

        # Conditionally initialize ECA Block
        if self.use_eca:
            self.final_attention = ECABlock(kernel_size=3)
        # self.cross_part_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

    def forward(self, x, y, joint):
        # x is the data: (N, C, T, V, M). joint is the same tensor in original BlockGCN
        N, C, T, V, M = x.size()

        # --- Preprocessing ---
        # Reshape to (N*M*T) x V x C for embedding, then (N*M) x (V*C) x T for BN
        x_flat = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x_embed = self.to_joint_embedding(x_flat)  # (N*M*T) x V x 64
        x_embed += self.pos_embedding[:, :self.num_point]

        # Reshape for BN: (N*M) x (V*C) x T
        x_bn = rearrange(x_embed, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x_bn = self.data_bn(x_bn)

        # Final reshape to GCN input format: (N*M, C, T, V)
        x_gcn_in = x_bn.view(N, M, V, EMBEDDING_CHANNELS, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, EMBEDDING_CHANNELS, T, V)

        # --- 1. Local Part Processing ---
        # --- 1. Local Part Processing ---
        part_features = []
        B, C_in, T_in, V_full = x_gcn_in.size()  # B = N*M

        for i, indices in enumerate(self.graph.part_indices):
            # Slice the data to get the nodes for the current part
            x_part = x_gcn_in[:, :, :, indices]

            # --- STANDARD SEQUENTIAL PASS (No Dense Skips) ---
            # The entire 4-block TCN_GCN_unit sequence runs sequentially
            feat = self.part_encoders[i](x_part)

            # --- STANDARD AVERAGE POOLING (No Learnable/Max Pool) ---
            # Condense the local nodes into a single supernode feature
            feat_pooled = F.avg_pool2d(feat, kernel_size=(1, feat.size(3)))

            part_features.append(feat_pooled)

        # 1. Concatenate the 6 parts: (B, C, T, V_super=6)
        x_global = torch.cat(part_features, dim=3)

        # 2. Learnable Edges (Spatial Routing across limbs)
        if self.use_dynamic_asr:
            # Static A_asr + per-sample A_dyn; both applied as residual mixes.
            a_dyn = self.dynamic_asr(x_global)  # (B, 6, 6)
            a = self.super_learnable_adj.unsqueeze(0) + a_dyn
            x_global = x_global + torch.einsum('bctv,bvw->bctw', x_global, a)
        elif self.use_asr:
            x_global = x_global + torch.einsum('bctv,vw->bctw', x_global, self.super_learnable_adj)

        # 3. Temporal Shift Module (TSM - Temporal Routing)
        if self.use_tsm:
            B, C, T, V_super = x_global.size()
            fold = C // 8
            x_shifted = torch.zeros_like(x_global)
            x_shifted[:, :fold, :-1, :] = x_global[:, :fold, 1:, :]
            x_shifted[:, fold:2 * fold, 1:, :] = x_global[:, fold:2 * fold, :-1, :]
            x_shifted[:, 2 * fold:, :, :] = x_global[:, 2 * fold:, :, :]
            x_global = x_shifted  # Reassign back to x_global to maintain the pipeline

        # 4. Global Network
        x_global = self.global_net(x_global)

        # 5. Bottleneck Attention (Noise Filtering)
        if self.use_eca:
            x_global = self.final_attention(x_global)

        # 6. Final Classification
        g_h, g_w = x_global.size(2), x_global.size(3)
        x_out = F.avg_pool2d(x_global, kernel_size=(g_h, g_w), stride=(g_h, g_w))
        x_out = x_out.view(N, M, -1).mean(dim=1)

        return self.fc(x_out), y
