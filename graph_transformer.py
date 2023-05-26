import jax
from jaxtyping import Array, Float, Int, Bool
from jax import numpy as np
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from typing import Callable

from einops import rearrange, repeat


class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        x = nn.LayerNorm()(x)
        return self.fn(x, *args, **kwargs)


class Residual(nn.Module):
    @nn.compact
    def __call__(self, x: Array, res: Array) -> Array:
        return x + res


class GatedResidual(nn.Module):
    @nn.compact
    def __call__(self, x: Array, res: Array) -> Array:
        gated_input = np.concatenate((x, res, x - res), axis=-1)
        gate = nn.sigmoid(nn.Dense(1, use_bias=False)(gated_input))
        return x * gate + res * (1 - gate)


class Attention(nn.Module):
    dim_head: int = 64
    heads: int = 8

    @nn.compact
    def __call__(
        self,
        nodes: Float[Array, "b n ne"],
        edges: Float[Array, "b n n ee"],
        mask: Bool[Array, "b k"],
    ) -> Array:
        h = self.heads
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head**-0.5
        to_q = nn.Dense(inner_dim)
        to_kv = nn.Dense(inner_dim * 2)
        edges_to_kv = nn.Dense(inner_dim)
        to_out = nn.Dense(nodes.shape[-1])

        q = to_q(nodes)
        k, v = np.split(to_kv(nodes), 2, axis=-1)

        e_kv = edges_to_kv(edges)

        q, k, v, e_kv = map(
            lambda t: rearrange(t, "b ... (h d) -> (b h) ... d", h=h), (q, k, v, e_kv)
        )

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, "b j d -> b () j d"), (k, v))

        k += ek
        v += ev

        sim = np.einsum("b i d, b i j d -> b i j", q, k) * scale

        mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
        mask = repeat(mask, "b i j -> (b h) i j", h=h)
        max_neg_value = -np.finfo(sim.dtype).max
        # sim.masked_fill_(~mask, max_neg_value)
        sim = np.where(~mask, max_neg_value, sim)

        attn = nn.softmax(sim, axis=-1)
        out = np.einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return to_out(out)


def feedforward(dim: int, ff_mult: int = 4) -> nn.Module:
    return nn.Sequential(
        (
            nn.Dense(dim * ff_mult),
            nn.gelu,
            nn.Dense(dim),
        )
    )


class NodeEdgeLayerPair(nn.Module):
    dim_head: int
    heads: int
    with_feedforward: bool = True

    @nn.compact
    def __call__(
        self, nodes, edges, mask
    ):  # , node_edges_mask: tuple[Array, Array, Array]):
        # nodes, edges, mask = node_edges_mask
        attn = PreNorm(
            Attention(
                dim_head=self.dim_head,
                heads=self.heads,
            ),
        )
        attn_residual = GatedResidual()

        nodes = attn_residual(attn(nodes, edges, mask), nodes)

        if self.with_feedforward:
            ff = PreNorm(feedforward(self.dim))
            ff_residual = GatedResidual()
            nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges, mask


class GraphTransformer(nn.Module):
    depth: int
    edge_dim: int = -1
    dim_head: int = 64
    heads: int = 8
    gate_residual: bool = False
    with_feedforward: bool = False
    norm_edges: bool = False

    @nn.compact
    def __call__(self, nodes: Array, edges: Array, mask: Array) -> tuple[Array, Array]:
        nodes, edges, _ = nn.Sequential(
            [
                NodeEdgeLayerPair(
                    dim_head=self.dim_head,
                    heads=self.heads,
                    with_feedforward=self.with_feedforward,
                )
                for _ in range(self.depth)
            ]
        )(nodes, edges, mask)
        return nodes, edges

    @classmethod
    def initialize(
        cls,
        key: Array,
        number_of_nodes: int,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int = -1,
        out_edge_features: int = -1,
        num_layers: int = 3,
    ) -> tuple[nn.Module, FrozenDict]:
        out_node_features = (
            out_node_features if out_node_features > 0 else in_node_features
        )
        out_edge_features = (
            out_edge_features if out_edge_features > 0 else in_edge_features
        )
        model = cls(
            depth=num_layers,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (2, number_of_nodes, in_node_features)
        edges_shape = (2, number_of_nodes, number_of_nodes, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        mask = np.ones((2, number_of_nodes), dtype=bool)
        print(f"Init {nodes.shape=}")
        print(f"Init {edges.shape=}")
        params = model.init(
            key,
            nodes,
            edges,
            mask,
        )
        return model, params
