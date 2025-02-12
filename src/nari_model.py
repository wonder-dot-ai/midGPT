from dataclasses import dataclass
import math
import typing as tp
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jrandom
from jax import vmap
import jax.tree_util as jtu

from .layers import (
    Linear,
    Embedding,
    RMSNorm,
    fixed_pos_embedding,
    apply_rotary_pos_emb,
)

Array = jax.Array
KeyArray = tp.Any
P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding
Mesh = jax.sharding.Mesh
with_sharding_constraint = jax.lax.with_sharding_constraint

# =============================================================================
# Original Modules (GPT, Block, etc.)
# =============================================================================


class MLP(eqx.Module):
    c_fc: Linear
    c_proj: Linear
    dropout: eqx.nn.Dropout

    def __init__(self, n_embd, dropout, key):
        key1, key2 = jrandom.split(key)
        self.c_fc = Linear(n_embd, 4 * n_embd, key=key1)
        self.c_proj = Linear(4 * n_embd, n_embd, key=key2)
        self.dropout = eqx.nn.Dropout(dropout)

    @jax.named_scope("mlp")
    def __call__(self, x_D, inference=False, key=None):
        x_D = jax.nn.gelu(self.c_fc(x_D))
        return self.dropout(self.c_proj(x_D), inference=inference, key=key)


class CausalSelfAttention(eqx.Module):
    n_head: int
    n_embd: int
    c_attn: Linear
    c_proj: Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    q_ln: eqx.nn.LayerNorm
    k_ln: eqx.nn.LayerNorm

    def __init__(self, n_embd, n_head, dropout, key):
        key1, key2 = jrandom.split(key)
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = Linear(n_embd, 3 * n_embd, key=key1)
        self.c_proj = Linear(n_embd, n_embd, key=key2)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.q_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )
        self.k_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )

    @jax.named_scope("causal_sa")
    def __call__(self, x_TxD, inference=False, key=None):
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
        T, D = x_TxD.shape
        Q_TxD, K_TxD, V_TxD = jnp.split(vmap(self.c_attn)(x_TxD), 3, axis=-1)
        C = self.n_embd // self.n_head
        Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T, self.n_head, C)), (1, 0, 2))
        K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T, self.n_head, C)), (1, 0, 2))
        # QK LayerNorm
        Q_HxTxC = vmap(vmap(self.q_ln))(Q_HxTxC)
        K_HxTxC = vmap(vmap(self.k_ln))(K_HxTxC)
        # Rotary embeddings
        sin_TxCp, cos_TxCp = fixed_pos_embedding(C, T)  # Cp = C//2
        Q_HxTxC = apply_rotary_pos_emb(Q_HxTxC, sin_TxCp, cos_TxCp)
        K_HxTxC = apply_rotary_pos_emb(K_HxTxC, sin_TxCp, cos_TxCp)
        V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T, self.n_head, C)), (1, 0, 2))
        A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
        causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
        A_HxTxT = jnp.where(causal_mask, float("-inf"), A_HxTxT)
        # Softmax should be in full precision.
        orig_dtype = A_HxTxT.dtype
        A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
        A_HxTxT = A_HxTxT.astype(orig_dtype)
        A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
        out_TxD = jnp.reshape(jnp.transpose(A_HxTxT @ V_HxTxC, (1, 0, 2)), (T, D))
        out_TxD = self.resid_dropout(
            vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key
        )
        return out_TxD


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    ln1: RMSNorm
    ln2: RMSNorm

    def __init__(self, n_embd, n_head, dropout, key):
        key1, key2 = jrandom.split(key)
        self.attn = CausalSelfAttention(
            n_embd=n_embd, n_head=n_head, dropout=dropout, key=key1
        )
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, key=key2)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    @jax.named_scope("block")
    def __call__(self, x_TxD, inference=False, key=None):
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jrandom.split(key)
            mlp_key = jrandom.split(mlp_key, x_TxD.shape[0])
        x_TxD = x_TxD + self.attn(
            vmap(self.ln1)(x_TxD), inference=inference, key=attn_key
        )
        mlp = vmap(self.mlp, in_axes=(0, None, 0))
        return x_TxD + mlp(vmap(self.ln2)(x_TxD), inference, mlp_key)


@dataclass
class GPTConfig:
    block_size: int  # Max sequence length
    vocab_size: int  # No. of tokens
    n_layer: int  # No. of transformer blocks
    n_head: int  # No. attention heads
    n_embd: int  # Hidden dimension
    dropout: float


class GPT(eqx.Module):
    wte: Embedding
    drop: eqx.nn.Dropout
    blocks: tp.List[Block]
    ln_f: RMSNorm
    lm_head: Linear
    n_layer: int

    def __init__(self, config, key):
        self.n_layer = config.n_layer
        block_key, head_key = jrandom.split(key)
        self.drop = eqx.nn.Dropout(config.dropout)

        def make_block(_key):
            return Block(config.n_embd, config.n_head, config.dropout, _key)

        self.blocks = eqx.filter_vmap(make_block)(
            jrandom.split(block_key, config.n_layer)
        )
        self.ln_f = RMSNorm(config.n_embd, eps=1e-5)
        embed_std = 1 / math.sqrt(config.n_embd)
        wte_wt = embed_std * jrandom.normal(
            head_key, (config.vocab_size, config.n_embd)
        )
        self.wte = Embedding(config.vocab_size, config.n_embd, weight=wte_wt)
        # Share first and last layer parameters.
        self.lm_head = Linear(config.n_embd, config.vocab_size, weight=wte_wt)

    @jax.named_scope("gpt")
    def __call__(self, x_T, inference=False, key=None):
        # Either (inference=False and key) or (inference=True and key=None)
        drop_key, block_keys = None, None
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, self.n_layer)
        x_TxD = self.drop(self.wte(x_T), inference=inference, key=drop_key)
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def block_fn(
            _x_TxD: Array, block_and_key: tp.Tuple[GPT, tp.Optional[KeyArray]]
        ):
            _dynamic_block, _key = block_and_key
            block = eqx.combine(_dynamic_block, static_blocks)
            return block(_x_TxD, inference=inference, key=_key), None

        # Set unroll=self.n_layer for better speed (but slower compile).
        x_TxD, _ = jax.lax.scan(block_fn, x_TxD, (dynamic_blocks, block_keys), unroll=1)
        x_TxD = vmap(self.ln_f)(x_TxD)
        logits_TxV = vmap(self.lm_head)(x_TxD)
        return logits_TxV


def count_params(model: GPT) -> int:
    dupe = jnp.size(model.lm_head.weight_MxN)  # embedding and final layer are shared.
    tot = sum([jnp.size(x) for x in jtu.tree_leaves(model) if isinstance(x, jax.Array)])
    return tot - dupe  # non-embedding only.


def shard_gpt(
    model: GPT, mesh: Mesh, shard_model: bool, sharding_fn=with_sharding_constraint
) -> eqx.Module:
    """Shard model parameters over devices (TPUs or GPUs)."""

    def sharding_map(x: Array) -> NamedSharding:
        axes: tuple[tp.Any, ...] = (None,) * x.ndim
        if x.size > 2**18 and shard_model:
            axes = (None,) * (x.ndim - 1) + ("data",)
        return NamedSharding(mesh, P(*axes))

    dynamic_model, static_model = eqx.partition(model, eqx.is_array)
    dynamic_model = jtu.tree_map(
        lambda x: sharding_fn(x, sharding_map(x)), dynamic_model
    )
    return eqx.combine(dynamic_model, static_model)


# =============================================================================
# New Modules for Encoder–Decoder Transformer
# =============================================================================


# --- 5. A configuration dataclass for the encoder–decoder transformer.
@dataclass
class TransformerConfig:
    enc_block_size: int  # Max sequence length for encoder
    dec_block_size: int  # Max sequence length for decoder
    src_vocab_size: int  # Vocabulary size for encoder
    tgt_vocab_size: int  # Vocabulary size for decoder
    n_enc_layer: int = 12  # Number of encoder layers
    n_dec_layer: int = 32  # Number of decoder layers
    n_enc_embd: int = 1536  # Encoder embedding dimension
    n_dec_embd: int = 2560  # Decoder embedding dimension
    n_enc_head: int = 16  # Encoder attention heads
    n_dec_gqa_query_head: int = 32  # Decoder GQA query heads
    n_dec_cross_query_head: int = 16 # Decoder cross-attention query heads
    n_dec_kv_head: int = 8  # Decoder key / value heads
    dropout: float = 0.1


# --- 1. A "full" (bidirectional) self-attention module (for the encoder)
class SelfAttention(eqx.Module):
    n_head: int
    n_embd: int
    c_attn: Linear
    c_proj: Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    q_ln: eqx.nn.LayerNorm
    k_ln: eqx.nn.LayerNorm

    def __init__(self, n_embd, n_head, dropout, key):
        key1, key2 = jrandom.split(key)
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.c_attn = Linear(n_embd, 3 * n_embd, key=key1)
        self.c_proj = Linear(n_embd, n_embd, key=key2)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.q_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )
        self.k_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )

    @jax.named_scope("self_sa")
    def __call__(self, x_TxD, inference=False, key=None):
        # Note: this module is identical to CausalSelfAttention except no causal mask.
        T, D = x_TxD.shape
        Q_TxD, K_TxD, V_TxD = jnp.split(vmap(self.c_attn)(x_TxD), 3, axis=-1)
        C = self.n_embd // self.n_head
        Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T, self.n_head, C)), (1, 0, 2))
        K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T, self.n_head, C)), (1, 0, 2))
        Q_HxTxC = vmap(vmap(self.q_ln))(Q_HxTxC)
        K_HxTxC = vmap(vmap(self.k_ln))(K_HxTxC)
        # Rotary embeddings
        sin_TxCp, cos_TxCp = fixed_pos_embedding(C, T)
        Q_HxTxC = apply_rotary_pos_emb(Q_HxTxC, sin_TxCp, cos_TxCp)
        K_HxTxC = apply_rotary_pos_emb(K_HxTxC, sin_TxCp, cos_TxCp)
        V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T, self.n_head, C)), (1, 0, 2))
        A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
        # (No causal masking is applied here.)
        orig_dtype = A_HxTxT.dtype
        A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
        A_HxTxT = A_HxTxT.astype(orig_dtype)
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
        A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
        out_TxD = jnp.reshape(jnp.transpose(A_HxTxT @ V_HxTxC, (1, 0, 2)), (T, D))
        out_TxD = self.resid_dropout(
            vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key
        )
        return out_TxD


# --- 2. An Encoder block (self-attention + MLP) using the above SelfAttention.
class EncoderBlock(eqx.Module):
    attn: SelfAttention
    mlp: MLP
    ln1: RMSNorm
    ln2: RMSNorm

    def __init__(self, n_embd, n_head, dropout, key):
        key1, key2 = jrandom.split(key)
        self.attn = SelfAttention(
            n_embd=n_embd, n_head=n_head, dropout=dropout, key=key1
        )
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, key=key2)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    @jax.named_scope("encoder_block")
    def __call__(self, x_TxD, inference=False, key=None):
        attn_key, mlp_key = (None, None)
        if key is not None:
            attn_key, mlp_key = jrandom.split(key)
            mlp_key = jrandom.split(mlp_key, x_TxD.shape[0])
        x_TxD = x_TxD + self.attn(
            vmap(self.ln1)(x_TxD), inference=inference, key=attn_key
        )
        mlp = vmap(self.mlp, in_axes=(0, None, 0))
        return x_TxD + mlp(vmap(self.ln2)(x_TxD), inference, mlp_key)


# --- 3. A Cross-Attention module for decoder blocks.
class CrossAttention(eqx.Module):
    n_head: int
    n_embd: int
    q_proj: Linear
    kv_proj: Linear
    out_proj: Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    q_ln: eqx.nn.LayerNorm
    k_ln: eqx.nn.LayerNorm

    def __init__(self, n_embd, n_head, dropout, key):
        key1, key2, key3 = jrandom.split(key, 3)
        self.n_head = n_head
        self.n_embd = n_embd
        self.q_proj = Linear(n_embd, n_embd, key=key1)
        self.kv_proj = Linear(n_embd, 2 * n_embd, key=key2)
        self.out_proj = Linear(n_embd, n_embd, key=key3)
        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.q_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )
        self.k_ln = eqx.nn.LayerNorm(
            n_embd // n_head, eps=1e-6, use_weight=True, use_bias=False
        )

    @jax.named_scope("cross_attention")
    def __call__(self, x_dec_TxD, x_enc_TxD, inference=False, key=None):
        # x_dec_TxD: decoder tokens; x_enc_TxD: encoder tokens
        T_dec, _ = x_dec_TxD.shape
        T_enc, _ = x_enc_TxD.shape
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)
        # Query from decoder.
        Q_TxD = vmap(self.q_proj)(x_dec_TxD)
        C = self.n_embd // self.n_head
        Q_HxTxC = jnp.transpose(jnp.reshape(Q_TxD, (T_dec, self.n_head, C)), (1, 0, 2))
        # Keys and values from encoder.
        KV_Tx2D = vmap(self.kv_proj)(x_enc_TxD)
        K_TxD, V_TxD = jnp.split(KV_Tx2D, 2, axis=-1)
        K_HxTxC = jnp.transpose(jnp.reshape(K_TxD, (T_enc, self.n_head, C)), (1, 0, 2))
        V_HxTxC = jnp.transpose(jnp.reshape(V_TxD, (T_enc, self.n_head, C)), (1, 0, 2))
        # Apply layer norm to Q and K.
        Q_HxTxC = vmap(vmap(self.q_ln))(Q_HxTxC)
        K_HxTxC = vmap(vmap(self.k_ln))(K_HxTxC)
        # Compute attention.
        A_HxTxT = Q_HxTxC @ jnp.transpose(K_HxTxC, (0, 2, 1))
        orig_dtype = A_HxTxT.dtype
        A_HxTxT = jax.nn.softmax(A_HxTxT.astype(jnp.float32) / jnp.sqrt(C), axis=-1)
        A_HxTxT = A_HxTxT.astype(orig_dtype)
        A_HxTxT = self.attn_dropout(A_HxTxT, inference=inference, key=adrop_key)
        out_dec_HxTxC = A_HxTxT @ V_HxTxC
        out_dec_TxD = jnp.reshape(
            jnp.transpose(out_dec_HxTxC, (1, 0, 2)), (T_dec, self.n_embd)
        )
        out_dec_TxD = self.resid_dropout(
            vmap(self.out_proj)(out_dec_TxD), inference=inference, key=pdrop_key
        )
        return out_dec_TxD


# --- 4. A Masked Grouped Query Attention (GQA) for self-attention.
class GroupedQueryAttention(eqx.Module):
    n_query_heads: int  # Number of query heads (32)
    n_kv_heads: int  # Number of key/value heads (8)
    n_embd: int
    c_attn: Linear  # Projects to queries
    c_kv: Linear  # Projects to keys and values
    c_proj: Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout
    q_ln: eqx.nn.LayerNorm
    k_ln: eqx.nn.LayerNorm

    def __init__(
        self,
        n_embd,
        n_query_heads,
        n_kv_heads,
        dropout,
        key,
    ):
        key1, key2, key3 = jrandom.split(key, 3)
        self.n_query_heads = n_query_heads  # 32
        self.n_kv_heads = n_kv_heads  # 8
        self.n_embd = n_embd  # 2560

        # Separate projections for Q and KV
        self.c_attn = Linear(n_embd, n_embd, key=key1)  # 2560, 2560
        self.c_kv = Linear(n_embd, n_embd // 2, key=key2)  # 2560, 1280
        self.c_proj = Linear(n_embd, n_embd, key=key3)  # 2560, 2560

        self.attn_dropout = eqx.nn.Dropout(dropout)
        self.resid_dropout = eqx.nn.Dropout(dropout)
        self.q_ln = eqx.nn.LayerNorm(
            n_embd // n_query_heads, eps=1e-6, use_weight=True, use_bias=False
        )
        self.k_ln = eqx.nn.LayerNorm(
            n_embd // n_kv_heads, eps=1e-6, use_weight=True, use_bias=False
        )

    @jax.named_scope("grouped_query_attn")
    def __call__(self, x_TxD, inference=False, key=None):
        T, D = x_TxD.shape  # (T, 2560)
        adrop_key, pdrop_key = jrandom.split(key) if key is not None else (None, None)

        # Generate Q, K, V with different numbers of heads
        Q_TxD = vmap(self.c_attn)(x_TxD)
        KV_TxD2 = vmap(self.c_kv)(x_TxD)  # (T, 1280)
        K_TxD4, V_TxD4 = jnp.split(KV_TxD2, 2, axis=-1)  # (T, 640), (T, 640)

        # Reshape with different numbers of heads
        Cq = self.n_embd // self.n_query_heads  # 80
        Ckv = self.n_embd // self.n_kv_heads  # 640

        Q_HqxTxCq = jnp.transpose(
            jnp.reshape(Q_TxD, (T, self.n_query_heads, Cq)), (1, 0, 2)  # (32, T, 80)
        )
        K_HkvxTxCkv = jnp.transpose(
            jnp.reshape(K_TxD4, (T, self.n_kv_heads, Ckv)), (1, 0, 2)  # (8, T, 640)
        )
        V_HkvxTxCkv = jnp.transpose(
            jnp.reshape(V_TxD4, (T, self.n_kv_heads, Ckv)), (1, 0, 2)  # (8, T, 640)
        )

        # Repeat KV heads to match query heads
        repeat_factor = self.n_query_heads // self.n_kv_heads  # 32 // 8 = 4
        K_HqxTxCkv = jnp.repeat(K_HkvxTxCkv, repeat_factor, axis=0)  # (32, T, 640)
        V_HqxTxCkv = jnp.repeat(V_HkvxTxCkv, repeat_factor, axis=0)  # (32, T, 640)

        # Apply layer norm
        Q_HqxTxCq = vmap(vmap(self.q_ln))(Q_HqxTxCq)  # (32, T, 80)
        K_HqxTxCkv = vmap(vmap(self.k_ln))(K_HqxTxCkv)  # (32, T, 640)

        # Apply rotary embeddings
        sin_TxCp, cos_TxCp = fixed_pos_embedding(min(Cq, Ckv), T)
        Q_HqxTxCq = apply_rotary_pos_emb(Q_HqxTxCq, sin_TxCp, cos_TxCp)  # (32, T, 80)
        K_HqxTxCkv = apply_rotary_pos_emb(
            K_HqxTxCkv, sin_TxCp, cos_TxCp
        )  # (32, T, 640)

        # Apply causal mask
        A_HqxTxT = Q_HqxTxCq @ jnp.transpose(K_HqxTxCkv, (0, 2, 1))  # (32, T, T)
        causal_mask = jnp.tril(jnp.ones((1, T, T))) == 0
        A_HqxTxT = jnp.where(causal_mask, float("-inf"), A_HqxTxT)  # (32, T, T)

        orig_dtype = A_HqxTxT.dtype
        A_HqxTxT = jax.nn.softmax(
            A_HqxTxT.astype(jnp.float32) / jnp.sqrt(Ckv), axis=-1
        )  # (32, T, T)
        A_HqxTxT = A_HqxTxT.astype(orig_dtype)  # (32, T, T)
        A_HqxTxT = self.attn_dropout(
            A_HqxTxT, inference=inference, key=adrop_key
        )  # (32, T, T)

        out_TxD = jnp.reshape(
            jnp.transpose(A_HqxTxT @ V_HqxTxCkv, (1, 0, 2)), (T, D)
        )  # (T, 2560)
        out_TxD = self.resid_dropout(
            vmap(self.c_proj)(out_TxD), inference=inference, key=pdrop_key
        )  # (T, 2560)
        return out_TxD


# --- 4. A Decoder block that combines masked self-attention, cross-attention, and MLP.
class DecoderBlock(eqx.Module):
    self_attn: GroupedQueryAttention
    cross_attn: CrossAttention
    mlp: MLP
    ln1: RMSNorm
    ln2: RMSNorm
    ln3: RMSNorm

    def __init__(self, config: TransformerConfig, key):
        key1, key2, key3 = jrandom.split(key, 3)
        self.self_attn = GroupedQueryAttention(
            n_embd=config.n_dec_embd,
            n_query_heads=config.n_dec_gqa_query_head,
            n_kv_heads=config.n_dec_kv_head,
            dropout=config.dropout,
            key=key1,
        )
        self.cross_attn = CrossAttention(
            config.n_dec_embd, config.n_dec_cross_query_head, config.dropout, key=key2
        )
        self.mlp = MLP(config.n_dec_embd, config.dropout, key=key3)
        self.ln1 = RMSNorm(config.n_dec_embd)
        self.ln2 = RMSNorm(config.n_dec_embd)
        self.ln3 = RMSNorm(config.n_dec_embd)

    @jax.named_scope("decoder_block")
    def __call__(self, x_TxD, encoder_out, inference=False, key=None):
        if key is not None:
            key_sa, key_ca, key_mlp = jrandom.split(key, 3)
        else:
            key_sa = key_ca = key_mlp = None
        # Masked self-attention.
        x_TxD = x_TxD + self.self_attn(
            vmap(self.ln1)(x_TxD), inference=inference, key=key_sa
        )
        # Cross-attention: attend to encoder outputs.
        x_TxD = x_TxD + self.cross_attn(
            vmap(self.ln2)(x_TxD), encoder_out, inference=inference, key=key_ca
        )
        # Feed-forward network.
        mlp_fn = vmap(self.mlp, in_axes=(0, None, 0))
        mlp_keys = (
            jrandom.split(key_mlp, x_TxD.shape[0]) if key_mlp is not None else None
        )
        x_TxD = x_TxD + mlp_fn(vmap(self.ln3)(x_TxD), inference, mlp_keys)
        return x_TxD


# --- 6. The encoder module.
class TransformerEncoder(eqx.Module):
    wte: Embedding
    drop: eqx.nn.Dropout
    blocks: tp.List[EncoderBlock]
    ln_f: RMSNorm

    def __init__(self, config: TransformerConfig, key):
        self.drop = eqx.nn.Dropout(config.dropout)
        block_key, embed_key = jrandom.split(key)

        def make_block(k):
            return EncoderBlock(
                config.n_enc_embd, config.n_enc_head, config.dropout, key=k
            )

        self.blocks = eqx.filter_vmap(make_block)(
            jrandom.split(block_key, config.n_enc_layer)
        )
        self.ln_f = RMSNorm(config.n_enc_embd, eps=1e-5)
        embed_std = 1 / math.sqrt(config.n_enc_embd)
        wte_wt = embed_std * jrandom.normal(
            embed_key, (config.src_vocab_size, config.n_enc_embd)
        )
        self.wte = Embedding(config.src_vocab_size, config.n_enc_embd, weight=wte_wt)

    @jax.named_scope("encoder")
    def __call__(self, x_T, inference=False, key=None):
        drop_key, block_keys = None, None
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, len(self.blocks))
        x_TxD = self.drop(self.wte(x_T), inference=inference, key=drop_key)
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def block_fn(x, block_and_key):
            block, k = block_and_key
            block_full = eqx.combine(block, static_blocks)
            return block_full(x, inference=inference, key=k), None

        x_TxD, _ = jax.lax.scan(block_fn, x_TxD, (dynamic_blocks, block_keys), unroll=1)
        x_TxD = vmap(self.ln_f)(x_TxD)
        return x_TxD


# --- 7. The decoder module.
class TransformerDecoder(eqx.Module):
    wte: Embedding
    drop: eqx.nn.Dropout
    blocks: tp.List[DecoderBlock]
    ln_f: RMSNorm
    lm_head: Linear
    n_layer: int

    def __init__(self, config: TransformerConfig, key):
        self.n_layer = config.n_dec_layer
        block_key, head_key = jrandom.split(key)
        self.drop = eqx.nn.Dropout(config.dropout)

        def make_block(k):
            return DecoderBlock(config, k)

        self.blocks = eqx.filter_vmap(make_block)(
            jrandom.split(block_key, config.n_dec_layer)
        )
        self.ln_f = RMSNorm(config.n_dec_embd, eps=1e-5)
        embed_std = 1 / math.sqrt(config.n_dec_embd)
        init_embed = embed_std * jrandom.normal(
            head_key, (config.tgt_vocab_size, config.n_dec_embd)
        )
        self.wte = Embedding(
            config.tgt_vocab_size, config.n_dec_embd, weight=init_embed
        )
        # Transpose for the linear layer.
        self.lm_head = Linear(
            config.n_dec_embd,
            config.tgt_vocab_size,
            weight=jnp.transpose(init_embed, (1, 0)),
        )

    @jax.named_scope("decoder")
    def __call__(self, x_T, encoder_out, inference=False, key=None):
        drop_key, block_keys = None, None
        if key is not None:
            drop_key, block_keys = jrandom.split(key)
            block_keys = jrandom.split(block_keys, self.n_layer)
        x_TxD = self.drop(self.wte(x_T), inference=inference, key=drop_key)
        dynamic_blocks, static_blocks = eqx.partition(self.blocks, eqx.is_array)

        @jax.checkpoint
        def block_fn(x, block_and_key):
            block, k = block_and_key
            block_full = eqx.combine(block, static_blocks)
            return block_full(x, encoder_out, inference=inference, key=k), None

        x_TxD, _ = jax.lax.scan(block_fn, x_TxD, (dynamic_blocks, block_keys), unroll=1)
        x_TxD = vmap(self.ln_f)(x_TxD)
        logits_TxV = vmap(self.lm_head)(x_TxD)
        return logits_TxV


# --- 8. The full encoder-decoder Transformer.
class Nari(eqx.Module):
    encoder: TransformerEncoder
    decoder: TransformerDecoder

    def __init__(self, config: TransformerConfig, key):
        enc_key, dec_key = jrandom.split(key)
        self.encoder = TransformerEncoder(config, enc_key)
        self.decoder = TransformerDecoder(config, dec_key)

    @jax.named_scope("transformer")
    def __call__(self, src_T, tgt_T, inference=False, key=None):
        enc_key, dec_key = (None, None)
        if key is not None:
            enc_key, dec_key = jrandom.split(key)
        encoder_out = self.encoder(src_T, inference=inference, key=enc_key)
        logits = self.decoder(tgt_T, encoder_out, inference=inference, key=dec_key)
        return logits


def shard_nari(
    model: Nari, mesh: Mesh, shard_model: bool, sharding_fn=with_sharding_constraint
) -> eqx.Module:
    """Shard model parameters over devices (TPUs or GPUs)."""

    def sharding_map(x: Array) -> NamedSharding:
        axes: tuple[tp.Any, ...] = (None,) * x.ndim
        if x.size > 2**18 and shard_model:
            axes = (None,) * (x.ndim - 1) + ("data",)
        return NamedSharding(mesh, P(*axes))

    dynamic_model, static_model = eqx.partition(model, eqx.is_array)
    dynamic_model = jtu.tree_map(
        lambda x: sharding_fn(x, sharding_map(x)), dynamic_model
    )
    return eqx.combine(dynamic_model, static_model)
