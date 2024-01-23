from typing import Optional

import torch
import torch.nn as nn

from misc import (
    argmax_one_hot,
    get_init_weight,
    stable_softmax_temperature
)


class MixtureOfTokensLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            n_experts: int = 4,
            group_size: int = 32,
            experts: int = 512,
            sparsity_dim: int = 0,
            temperature: float = 1.0,
            init_type: str = "truncated_normal",
            init_scale: float = 1.0,
            expert_size: Optional[int] = None,
            flop_matched: bool = False,
            emit_softmax_over_experts: bool = False,
            use_discrete_routing: bool = False

    ):
        super().__init__()

        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.n_experts: int = n_experts

        self.group_size: int = group_size
        self.experts: int = experts

        self.sparsity_dim: int = sparsity_dim

        self.temperature: float = temperature

        self.init_type: str = init_type
        self.init_scale: float = init_scale

        self.flop_matched: bool = flop_matched
        self.emit_softmax_over_experts: bool = emit_softmax_over_experts
        self.use_discrete_routing: bool = use_discrete_routing

        if flop_matched:
            assert d_ff == 4 * d_model, f"dff = {self.dff} is not equal to 4 * dm = {4 * self.dm} as in vanilla transformer"
            self.dff *= self.group_size

        if expert_size is None:
            assert d_ff % n_experts == 0, f"dff = {self.dff} is not divisible by n_experts = {self.n_experts}"
            self.expert_size = d_ff // n_experts

        self.lin1 = nn.Parameter(
            get_init_weight(
                (self.n_experts, self.d_model, self.expert_size),
                fan_in=self.d_model,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        self.lin2 = nn.Parameter(
            get_init_weight(
                (self.n_experts, self.expert_size, self.d_model),
                fan_in=self.expert_size,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

        self.controller = nn.Parameter(
            get_init_weight(
                (self.d_model, self.n_experts),
                fan_in=self.d_model,
                init_type=self.init_type,
                scale=self.init_scale,
            )
        )

    def forward(self, x):
        x = self.group_tokens(x)
        merge_weights, emit_weights = self.calculate_mixed_tokens_with_weights(x)
        x = self.merge_map_emit(x, merge_weights, emit_weights)
        x = self.redistribute_tokens(x)
        return x

    def group_tokens(self, x):
        """
        Reshape code so the axis to split into groups is on position 1, and then group over said axis.
        e.g.:
         - if we group tokens from different sequences in a batch (sparsity = 0), we need to put the batch dimension to position 1.
         - if we group tokens within one sequence, the dimension to split into groups is already on position 1, hence we leave it as is.

        free_dimension is the dimension on position 0 after reshape
        split_dimension is the dimension on position 1 - the one to split into groups

        :param x: normal input tensor of shape (batch, seq_len, dmodel)
        :return: x of shape (free_dimension, split_dimension // group_size, group_size , dmodel)
        """
        assert len(x.shape) == 3, "incorrect shape of a tensor, expected a 3D tensor"
        assert x.size(-1) == self.d_model, f"expected the last dimension of input tensor to be d_model = {self.d_model}"

        if self.sparsity_dim == 0:
            x = x.transpose(0, 1)
        elif self.sparsity_dim != 1:
            raise NotImplementedError

        free_dimension = x.size(1)
        assert free_dimension % self.group_size == 0, f"free dimension = {free_dimension} should be divisible by group size = {self.group_size}"

        x = x.view(x.size(0), -1, self.group_size, self.d_model)
        return x

    def redistribute_tokens(self, x):
        """
        An inverse operation to group_tokens.
        """
        assert len(x.shape) == 4, "incorrect shape of a tensor, expected a 4D tensor"

        x = x.view(x.size(0), -1, self.d_model)
        if self.sparsity_dim == 0:
            x = x.transpose(0, 1)
        elif self.sparsity_dim != 1:
            raise NotImplementedError

        return x

    def calculate_mixed_tokens_with_weights(self, x):
        """
        This function calculates merge and emit weights based on the input tensor, using a controller matrix.
        The merge weights determine the aggregation of tokens within a group, and emit weights govern the redistribution
        of the aggregated token back to the original tokens. Temperature scaling is applied to the logits, and optional
        discrete routing can be used to obtain one-hot representations of the weights.
        """
        # shape of x is (free_dimension, split_dimension // group_size, group_size, dmodel)
        merge_logits = torch.matmul(x, self.controller)
        # self.update_cache_for_logging("merge_logits", merge_logits)

        # shape of merge_logits is (free_dimension, aggr_dimension // group_size, group_size, n_experts)
        temp_merge = self.temperature
        temp_emit = self.temperature

        merge_softmax_dim = -2
        emit_softmax_dim = -1 if self.emit_softmax_over_experts else -2

        merge_weights = stable_softmax_temperature(
            merge_logits, temp_merge, dim=merge_softmax_dim
        )

        # by default we use the same weights for emitting and merging, but if the temperature is learnable or we want to take softmax over experts for emitting, we will use different weights
        if isinstance(temp_merge, torch.nn.Parameter) or self.emit_softmax_over_experts:
            emit_weights = stable_softmax_temperature(
                merge_logits, temp_emit, dim=emit_softmax_dim
            )
        else:
            emit_weights = merge_weights

        if self.use_discrete_routing:
            merge_weights = argmax_one_hot(merge_weights, dim=merge_softmax_dim)
            emit_weights = argmax_one_hot(emit_weights, dim=emit_softmax_dim)
        return merge_weights, emit_weights

    def merge_map_emit(self, x, merge_weights, emit_weights):
        """
        :param x: input reshaped to (free_dimension, split_dimension // group_size, group_size, dmodel)
        :param merge_weights: weights for merging tokens within a group, shape (free_dimension, split_dimension // group_size, group_size, n_experts)
        :param emit_weights: weights for emitting tokens within a group, shape (free_dimension, split_dimension // group_size, group_size, n_experts)
        :return: tensor of token updates of shape (free_dimension, split_dimension // group_size, group_size, dmodel)
        """
        x = torch.matmul(
            merge_weights.transpose(-1, -2),
            x,
        )
        # x shape is (free_dimension, split_dimension // group_size, n_experts, dmodel) ||| lin1 shape is (n_experts, dmodel, expert_size)
        x = torch.bmm(x.view(-1, self.n_experts, self.d_model).transpose(0, 1), self.lin1)
        x = torch.relu_(x)
        # x shape is (n_experts, free_dimension * aggr_dimension // group_size, expert_size) ||| lin2 shape is (n_experts, expert_size, dmodel)
        x = torch.bmm(x, self.lin2)
        # x shape is (n_experts, free_dimension * aggr_dimension // group_size, dmodel)

        # merge_weights shape is (free_dimension, aggr_dimension // group_size, group_size, n_experts)
        # view x to be (n_experts, free_dimension, aggr_dimension // group_size, dmodel)
        # permute it to be (free_dimension, aggr_dimension // group_size, n_experts, dmodel)
        x = torch.matmul(
            emit_weights,
            x.view(x.size(0), emit_weights.size(0), -1, self.d_model).permute(1, 2, 0, 3),
        )

        return x
