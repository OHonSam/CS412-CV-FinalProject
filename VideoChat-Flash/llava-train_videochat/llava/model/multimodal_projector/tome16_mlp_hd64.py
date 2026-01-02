# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import torch.nn as nn
from typing import Callable, Tuple
import torch.nn.functional as F



# def bipartite_soft_matching(
#     metric: torch.Tensor,
#     r: int, 
# ) -> Tuple[Callable, Callable]:
#     """
#     Applies ToMe with a balanced matching set (50%, 50%).

#     Input size is [batch, tokens, channels].
#     r indicates the number of tokens to remove (max 50% of tokens).
#     """
#     protected = 0

#     t = metric.shape[1]
#     r = min(r, (t - protected) // 2)

#     assert r > 0, r

#     with torch.no_grad():
#         metric = metric / metric.norm(dim=-1, keepdim=True)
#         a, b = metric[..., ::2, :], metric[..., 1::2, :]
#         scores = a @ b.transpose(-1, -2)

#         node_max, node_idx = scores.max(dim=-1)
#         edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

#         unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
#         src_idx = edge_idx[..., :r, :]  # Merged Tokens
#         dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = x[..., ::2, :], x[..., 1::2, :]
#         n, t1, c = src.shape
#         unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
#         src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
#         dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)

#         return torch.cat([unm, dst], dim=1)

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         unm_len = unm_idx.shape[1]
#         unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
#         n, _, c = unm.shape

#         src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

#         out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

#         out[..., 1::2, :] = dst
#         out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
#         out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

#         return out

#     return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    query_embedding: torch.Tensor = None,
    relevance_weight: float = 0.3,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # Original similarity scores
        scores = a @ b.transpose(-1, -2)

        # ============== Query-Aware Relevance ==============
        if query_embedding is not None:
            batch_size = a.shape[0]
            
            # Ensure query_embedding has correct shape
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)  # (1, C)
            
            # Expand query to match batch size: (1, C) -> (B, C)
            if query_embedding.shape[0] == 1 and batch_size > 1:
                query_embedding = query_embedding.expand(batch_size, -1)  # (B, C)
            
            # Normalize query
            query_embedding = query_embedding / (query_embedding.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Handle dimension mismatch between query and metric
            query_dim = query_embedding.shape[-1]
            metric_dim = a.shape[-1]
            
            if query_dim != metric_dim:
                # Project query to metric dimension by truncating or padding
                if query_dim > metric_dim:
                    query_embedding = query_embedding[..., :metric_dim]
                else:
                    # Pad with zeros if query is smaller
                    padding = torch.zeros(
                        query_embedding.shape[0], metric_dim - query_dim,
                        device=query_embedding.device, dtype=query_embedding.dtype
                    )
                    query_embedding = torch.cat([query_embedding, padding], dim=-1)
                
                # Re-normalize after adjustment
                query_embedding = query_embedding / (query_embedding.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Compute relevance: (B, N/2, C) @ (B, C, 1) -> (B, N/2, 1)
            query_embedding = query_embedding.unsqueeze(-1)  # (B, C, 1)
            
            relevance_a = torch.bmm(a, query_embedding).squeeze(-1)  # (B, N/2)
            relevance_b = torch.bmm(b, query_embedding).squeeze(-1)  # (B, N/2)
            
            # Normalize relevance to [0, 1]
            relevance_a_min = relevance_a.min(dim=-1, keepdim=True)[0]
            relevance_a_max = relevance_a.max(dim=-1, keepdim=True)[0]
            relevance_a = (relevance_a - relevance_a_min) / (relevance_a_max - relevance_a_min + 1e-6)
            
            relevance_b_min = relevance_b.min(dim=-1, keepdim=True)[0]
            relevance_b_max = relevance_b.max(dim=-1, keepdim=True)[0]
            relevance_b = (relevance_b - relevance_b_min) / (relevance_b_max - relevance_b_min + 1e-6)
            
            # Create relevance penalty matrix: (B, N/2, N/2)
            relevance_penalty = relevance_a.unsqueeze(-1) + relevance_b.unsqueeze(-2)
            
            # Normalize penalty
            penalty_max = relevance_penalty.amax(dim=(-1, -2), keepdim=True)
            relevance_penalty = relevance_penalty / (penalty_max + 1e-6)
            
            # Apply penalty: reduce merge scores for relevant token pairs
            scores = scores - relevance_weight * relevance_penalty
        # =========================================================

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

# def merge_wavg(
#     merge: Callable, x: torch.Tensor, size: torch.Tensor = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Applies the merge function by taking a weighted average based on token size.
#     Returns the merged tensor and the new token sizes.
#     """
#     if size is None:
#         size = torch.ones_like(x[..., 0, None])

#     x = merge(x * size, mode="sum")
#     size = merge(size, mode="sum")

#     x = x / size
#     return x, size




# class ToMe16_mlp_hd64(nn.Module):
#     def __init__(self, config, vision_cfg):
#         super().__init__()
#         self._config = config
#         self.mm_hidden_size = config.mm_hidden_size
#         self.hw = vision_cfg.image_size // vision_cfg.patch_size
#         self.num_attention_heads = vision_cfg.num_attention_heads
#         self.mlp = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
#                     nn.GELU(),
#                     nn.Linear(config.hidden_size, config.hidden_size))
#         self.max_pos_hw = self.hw
#         self.max_pos_num_frames = config.mm_pos_num_frames
#         # self._set_3d_pos_cache(max_grid_size=self.max_pos_hw, max_t_size=self.max_pos_num_frames)
#         self.num_image_patches_per_side = 8
#         self.num_frame_patches_per_side = 4
        
#     def merge_tokens(self, x, target_num_token):
#         r"""
#         x = torch.randn(10, 2560, c)
#         x = merge_tokens(x, r_merge_list=[1280])
#         """
#         size = None
#         b, p, c = x.shape
#         tmp_p = p
#         r_merge_list = []
#         assert tmp_p > target_num_token, f"{tmp_p} should greater than {target_num_token}"
#         while tmp_p != target_num_token:
#             if tmp_p - target_num_token <= (tmp_p // 2):
#                 r_merge_list.append(tmp_p - target_num_token)
#                 break
#             else:
#                 r_merge_list.append(tmp_p // 2)
#                 tmp_p = tmp_p - (tmp_p // 2)
                
        
#         head = self.num_attention_heads

#         dim = c // head
#         for r in r_merge_list:
#             metric = x.reshape(b, p, head, dim).mean(2) # [b, p, c//head]
#             merge, _ = bipartite_soft_matching(
#                 metric, 
#                 r
#             )
#             x, size = merge_wavg(merge, x, size)
#             _, p, _ = x.shape
#         # x = x.reshape(-1, c)  # 300, 1024
#         return x




#     def forward(self, x, compress=False, local_num_frames=-1):

#         height = width = self.hw
#         assert height * width == x.shape[1]
#         dtype = x.dtype
#         device = x.device

#         if local_num_frames != -1 and local_num_frames != 1:
#             assert compress is True
#         if compress:
#             if local_num_frames != -1:
#                 num_frames = local_num_frames
#                 x = x.reshape(x.shape[0] // local_num_frames, -1, x.shape[-1])
#             else:
#                 num_frames = x.shape[0]
#                 x = x.reshape(1, -1, x.shape[-1])
#             num_tome_tokens = 16 * num_frames
#         else:
#             num_tome_tokens = 64
        
#         x = self.merge_tokens(x, target_num_token=num_tome_tokens)
#         x = self.mlp(x)

#         return x

#     @property
#     def config(self):
#         return {"mm_projector_type": "tome16_mlp_hd64"}

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class ToMe16_mlp_hd64(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.mm_hidden_size = config.mm_hidden_size
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.num_attention_heads = vision_cfg.num_attention_heads

        # MLP projection layers
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        
        # NEW: Optional projection layer for query embedding
        # This projects query from LLM space to vision space
        self.query_proj = nn.Linear(config.hidden_size, config.mm_hidden_size, bias=False)
        
        # NEW: Relevance weight (can be tuned)
        self.relevance_weight = getattr(config, 'tome_relevance_weight', 0.3)
        
    def merge_tokens(self, x, target_num_token, query_embedding=None):
        """
        Iteratively merge tokens until reaching target count.
        
        Args:
            x: (B, N, C) token features
            target_num_token: target number of tokens
            query_embedding: (C,) or (B, C) optional query for relevance weighting
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        
        # Calculate merge schedule
        r_merge_list = []
        assert tmp_p > target_num_token, f"tmp_p={tmp_p}, target={target_num_token}"
        
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)
        
        # Apply merges iteratively
        head = self.num_attention_heads
        dim = c // head
        
        for r in r_merge_list:
            # Compute metric using attention head averaging
            metric = x.reshape(b, p, head, dim).mean(2)
            
            # Get merge function with query-aware matching
            merge, _ = bipartite_soft_matching(
                metric, 
                r,
                query_embedding=query_embedding,      # NEW
                relevance_weight=self.relevance_weight  # NEW
            )
            
            # Apply weighted average merge
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        
        return x

    def forward(self, x, compress=False, query_embedding=None, local_num_frames=-1):
        """
        Forward pass with optional query-aware token merging.
        
        Args:
            x: (B, H*W, C) visual features, H*W = 256 typically
            compress: whether to apply token merging (True for video)
            query_embedding: (hidden_size,) query for relevance weighting
            local_num_frames: number of frames to process together
        
        Returns:
            (B, N_tokens, hidden_size) projected features
        """
        height = width = self.hw
        assert height * width == x.shape[1], f"hw={height*width}, x.shape[1]={x.shape[1]}"
        
        if local_num_frames != -1 and local_num_frames != 1:
            assert compress is True
        
        # Process query embedding if provided
        projected_query = None
        if query_embedding is not None and compress:
            # Project query from LLM space to vision space
            with torch.no_grad():
                if query_embedding.dim() == 1:
                    query_embedding = query_embedding.unsqueeze(0)
                projected_query = self.query_proj(query_embedding)  # (1, mm_hidden_size)
                projected_query = projected_query.squeeze(0)  # (mm_hidden_size,)
        
        if compress:
            # VIDEO MODE: merge tokens across frames
            if local_num_frames != -1:
                num_frames = local_num_frames
                # Reshape: (B, 256, C) → (B/local, local*256, C)
                x = x.reshape(x.shape[0] // local_num_frames, -1, x.shape[-1])
            else:
                num_frames = x.shape[0]
                x = x.reshape(1, -1, x.shape[-1])
            
            # Target: 16 tokens per frame
            num_tome_tokens = 16 * num_frames
            
            # Apply query-aware token merging
            x = self.merge_tokens(x, target_num_token=num_tome_tokens, query_embedding=projected_query)
        else:
            # IMAGE MODE: 64 tokens total, no query-aware merging needed
            num_tome_tokens = 64
            if x.shape[1] > num_tome_tokens:
                x = self.merge_tokens(x, target_num_token=num_tome_tokens, query_embedding=projected_query)
        
        # Project to LLM dimension
        x = self.mlp(x)
        
        return x

    @property
    def config(self):
        return self._config