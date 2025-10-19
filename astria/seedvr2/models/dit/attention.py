# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List, Optional

import torch
import torch.nn.functional as F

# from flash_attn import flash_attn_varlen_func

from torch import nn

def _sdpa_with_scale(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=softmax_scale,
        )
    except TypeError:
        pass

    if softmax_scale is not None:
        d = query.shape[-1]
        query = query * (softmax_scale * (d ** 0.5))
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


@torch.no_grad()  # remove if you need grads
def _flash_varlen_fallback(
    q: torch.Tensor,                  # (total_q, H, D)
    k: torch.Tensor,                  # (total_k, H, D)
    v: torch.Tensor,                  # (total_k, H, D)
    cu_seqlens_q: torch.Tensor,       # (B+1,)
    cu_seqlens_k: torch.Tensor,       # (B+1,)
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    device = q.device
    outs: List[torch.Tensor] = []

    sq = cu_seqlens_q.to("cpu").tolist()
    sk = cu_seqlens_k.to("cpu").tolist()
    B = len(sq) - 1

    for i in range(B):
        qs, qe = sq[i], sq[i + 1]
        ks, ke = sk[i], sk[i + 1]
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)  # (1,H,Lq,D)
        k_i = k[ks:ke].transpose(0, 1).unsqueeze(0)  # (1,H,Lk,D)
        v_i = v[ks:ke].transpose(0, 1).unsqueeze(0)  # (1,H,Lk,D)

        attn_mask = None
        is_causal = False
        if causal:
            if q_i.size(2) == k_i.size(2):
                is_causal = True
            else:
                Lq, Lk = q_i.size(2), k_i.size(2)
                attn_mask = torch.ones((Lq, Lk), dtype=torch.bool, device=device).triu(1)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,Lq,Lk)

        out_i = _sdpa_with_scale(
            q_i, k_i, v_i,
            attn_mask=attn_mask,
            dropout_p=dropout_p if torch.is_grad_enabled() and dropout_p > 0 else 0.0,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
        )
        outs.append(out_i.squeeze(0).transpose(0, 1).contiguous())  # -> (Lq,H,D)

    return torch.cat(outs, dim=0)  # (total_q,H,D)


def _first_defined(*vals):
    """Return the first item that is not None (no boolean coercion)."""
    for v in vals:
        if v is not None:
            return v
    return None


class TorchAttention(nn.Module):
    """
    Supports:
      • Standard SDPA: (query,key,value, attn_mask=?, dropout_p=?, is_causal|causal=?, softmax_scale|scale=?)
      • FlashAttention varlen: (q,k,v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=?, dropout_p=?, softmax_scale=?)
    """

    def tflops(self, args, kwargs, output) -> float:
        if "cu_seqlens_q" in kwargs and "cu_seqlens_k" in kwargs:
            H = output.shape[1] if output.dim() == 3 else 1
            cuq = kwargs["cu_seqlens_q"]
            cuk = kwargs["cu_seqlens_k"]
            seqlens_q = (cuq[1:] - cuq[:-1]).to(torch.float32) / 1e6
            seqlens_k = (cuk[1:] - cuk[:-1]).to(torch.float32) / 1e6
            q_like = _first_defined(kwargs.get("q"), kwargs.get("query"), args[0] if len(args) > 0 else None)
            d = q_like.shape[-1]
            return float(H * (4.0 * d * (seqlens_q * seqlens_k).sum()).item())

        q = _first_defined(kwargs.get("query"), kwargs.get("q"), args[0] if len(args) > 0 else None)
        k = _first_defined(kwargs.get("key"), kwargs.get("k"),   args[1] if len(args) > 1 else None)
        assert q is not None and k is not None, "query and key must be provided"
        if q.dim() == 4:
            b, h, sq, d = q.shape
            sk = k.shape[-2]
            return b * h * (4.0 * d * (sq / 1e6) * (sk / 1e6))
        else:
            sq = q.shape[-2]; sk = k.shape[-2]; d = q.shape[-1]
            return 4.0 * d * (sq / 1e6) * (sk / 1e6)

    def forward(self, *args, **kwargs):
        # Varlen (flash-attn style) call?
        if "cu_seqlens_q" in kwargs and "cu_seqlens_k" in kwargs:
            q = _first_defined(kwargs.get("q"), kwargs.get("query"), args[0] if len(args) > 0 else None)
            k = _first_defined(kwargs.get("k"), kwargs.get("key"),   args[1] if len(args) > 1 else None)
            v = _first_defined(kwargs.get("v"), kwargs.get("value"), args[2] if len(args) > 2 else None)
            if q is None or k is None or v is None:
                raise TypeError("Varlen attention expects q/k/v (or query/key/value) tensors")
            return _flash_varlen_fallback(
                q, k, v,
                kwargs["cu_seqlens_q"], kwargs["cu_seqlens_k"],
                kwargs.get("max_seqlen_q", q.shape[-2]),
                kwargs.get("max_seqlen_k", k.shape[-2]),
                dropout_p=kwargs.get("dropout_p", 0.0),
                softmax_scale=kwargs.get("softmax_scale", kwargs.get("scale", None)),
                causal=kwargs.get("causal", kwargs.get("is_causal", False)),
            )

        # Standard SDPA path
        query = _first_defined(kwargs.get("query"), kwargs.get("q"), args[0] if len(args) > 0 else None)
        key   = _first_defined(kwargs.get("key"),   kwargs.get("k"), args[1] if len(args) > 1 else None)
        value = _first_defined(kwargs.get("value"), kwargs.get("v"), args[2] if len(args) > 2 else None)
        if query is None or key is None or value is None:
            raise TypeError("scaled_dot_product_attention() requires query, key, value")

        attn_mask     = kwargs.get("attn_mask", None)
        dropout_p     = kwargs.get("dropout_p", 0.0)
        is_causal     = kwargs.get("is_causal", kwargs.get("causal", False))
        softmax_scale = kwargs.get("softmax_scale", kwargs.get("scale", None))

        return _sdpa_with_scale(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
        )


class SageAttention(TorchAttention):
    """
    SageAttention wrapper with graceful fallbacks.

    Supports:
      • Standard SDPA: (query,key,value, attn_mask=?, dropout_p=?, is_causal|causal=?, softmax_scale|scale=?)
      • Varlen: (q,k,v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=?, dropout_p=?, softmax_scale=?)

    Uses Sage when:
      • CUDA tensor
      • dtype is fp16/bf16
      • attn_mask is None OR attention is causal (no arbitrary masks)
    Otherwise falls back to your PyTorch SDPA / varlen fallback.
    """

    def __init__(
        self,
        *,
        tensor_layout: str = "HND",
        qk_quant_gran: str = "per_thread",
        pv_accum_dtype: str = "fp32+fp32",
        smooth_k: bool = True,
        prefer_sage_varlen: bool = True,
    ):
        super().__init__()
        self.tensor_layout = tensor_layout  # "HND" (B,H,N,D) or "NHD" (B,N,H,D) accepted; we convert as needed
        self.qk_quant_gran = qk_quant_gran
        self.pv_accum_dtype = pv_accum_dtype
        self.smooth_k = smooth_k
        self.prefer_sage_varlen = prefer_sage_varlen

        # Optional imports
        self._have_sage = False
        self._have_sage_varlen = False
        try:
            from sageattention import sageattn as _sageattn  # type: ignore
            self._sageattn = _sageattn
            self._have_sage = True
        except Exception:
            self._sageattn = None  # type: ignore

        try:
            from sageattention import sageattn_varlen as _sageattn_varlen  # type: ignore
            self._sageattn_varlen = _sageattn_varlen
            self._have_sage_varlen = True
        except Exception:
            self._sageattn_varlen = None  # type: ignore

    # ---------- helpers ----------
    @staticmethod
    def _first_defined(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    @staticmethod
    def _is_half_like(t: torch.Tensor) -> bool:
        return t.dtype in (torch.float16, torch.bfloat16)

    @staticmethod
    def _to_hnd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ensure (B,H,N,D) for Sage dense kernel. Supports input (B,H,N,D) or (B,N,H,D).
        """
        if q.dim() != 4:
            raise ValueError("Sage dense path expects 4D tensors")
        # If already (B,H,N,D) assume matching for k,v
        if q.size(1) == k.size(1) == v.size(1):
            return q, k, v
        # Otherwise treat as (B,N,H,D) and transpose to (B,H,N,D)
        return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    def _eligible_dense(self, q: torch.Tensor, attn_mask, is_causal: bool) -> bool:
        return (
            self._have_sage
            and q.is_cuda
            and self._is_half_like(q)
            and (attn_mask is None or is_causal)
            and q.dim() == 4
        )

    def _eligible_varlen(self, q: torch.Tensor, causal: bool) -> bool:
        # Varlen Sage requires CUDA and fp16/bf16; masking limited to causal/none.
        return self._have_sage_varlen and q.is_cuda and self._is_half_like(q)

    # ---------- forward ----------
    def forward(self, *args, **kwargs):
        # VARLEN SIGNATURE?
        if "cu_seqlens_q" in kwargs and "cu_seqlens_k" in kwargs:
            return self._forward_varlen(*args, **kwargs)
        # STANDARD SDPA
        return self._forward_dense(*args, **kwargs)

    # --- dense path (standard SDPA signature) ---
    def _forward_dense(self, *args, **kwargs):
        query = self._first_defined(kwargs.get("query"), kwargs.get("q"), args[0] if len(args) > 0 else None)
        key   = self._first_defined(kwargs.get("key"),   kwargs.get("k"), args[1] if len(args) > 1 else None)
        value = self._first_defined(kwargs.get("value"), kwargs.get("v"), args[2] if len(args) > 2 else None)
        if query is None or key is None or value is None:
            raise TypeError("scaled_dot_product_attention() requires query, key, value")

        attn_mask     = kwargs.get("attn_mask", None)
        dropout_p     = kwargs.get("dropout_p", 0.0)
        is_causal     = kwargs.get("is_causal", kwargs.get("causal", False))
        softmax_scale = kwargs.get("softmax_scale", kwargs.get("scale", None))

        if self._eligible_dense(query, attn_mask, is_causal):
            try:
                q, k, v = self._to_hnd(query, key, value)
                out = self._sageattn(  # type: ignore[attr-defined]
                    q, k, v,
                    tensor_layout="HND",
                    is_causal=is_causal,
                    sm_scale=softmax_scale,
                    qk_quant_gran=self.qk_quant_gran,
                    pv_accum_dtype=self.pv_accum_dtype,
                    smooth_k=self.smooth_k,
                )
                return out
            except Exception:
                # Fall through to PyTorch SDPA on any runtime issue
                pass

        # Fallback (keeps your scale handling + mask semantics)
        return _sdpa_with_scale(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
        )

    # --- varlen path (flash-attn style signature) ---
    @torch.no_grad()  # align with your fallback’s default
    def _forward_varlen(self, *args, **kwargs):
        q = self._first_defined(kwargs.get("q"), kwargs.get("query"), args[0] if len(args) > 0 else None)
        k = self._first_defined(kwargs.get("k"), kwargs.get("key"),   args[1] if len(args) > 1 else None)
        v = self._first_defined(kwargs.get("v"), kwargs.get("value"), args[2] if len(args) > 2 else None)
        if q is None or k is None or v is None:
            raise TypeError("Varlen attention expects q/k/v (or query/key/value) tensors")

        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        max_seqlen_q = kwargs.get("max_seqlen_q", q.shape[-2])
        max_seqlen_k = kwargs.get("max_seqlen_k", k.shape[-2])
        dropout_p    = kwargs.get("dropout_p", 0.0)
        softmax_scale = kwargs.get("softmax_scale", kwargs.get("scale", None))
        causal       = kwargs.get("causal", kwargs.get("is_causal", False))

        # Preferred: single-kernel Sage varlen
        if self.prefer_sage_varlen and self._eligible_varlen(q, causal):
            try:
                out = self._sageattn_varlen(  # type: ignore[attr-defined]
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    is_causal=causal,
                    sm_scale=softmax_scale,
                )
                return out
            except Exception:
                pass  # fall through

        # Secondary: per-sequence loop accelerated with Sage if available
        if self._have_sage and q.is_cuda and self._is_half_like(q):
            device = q.device
            outs: list[torch.Tensor] = []
            sq = cu_seqlens_q.to("cpu").tolist()
            sk = cu_seqlens_k.to("cpu").tolist()
            B = len(sq) - 1

            for i in range(B):
                qs, qe = sq[i], sq[i + 1]
                ks, ke = sk[i], sk[i + 1]
                # Build (1,H,L, D) for Sage dense
                q_i = q[qs:qe].transpose(0, 1).unsqueeze(0).contiguous()
                k_i = k[ks:ke].transpose(0, 1).unsqueeze(0).contiguous()
                v_i = v[ks:ke].transpose(0, 1).unsqueeze(0).contiguous()

                try:
                    out_i = self._sageattn(  # type: ignore[attr-defined]
                        q_i, k_i, v_i,
                        tensor_layout="HND",
                        is_causal=(causal and (q_i.size(2) == k_i.size(2))),
                        sm_scale=softmax_scale,
                        qk_quant_gran=self.qk_quant_gran,
                        pv_accum_dtype=self.pv_accum_dtype,
                        smooth_k=self.smooth_k,
                    )
                except Exception:
                    # If Sage failed for a sub-batch, fall back to SDPA just for this slice
                    attn_mask = None
                    is_causal_slice = False
                    if causal:
                        if q_i.size(2) == k_i.size(2):
                            is_causal_slice = True
                        else:
                            Lq, Lk = q_i.size(2), k_i.size(2)
                            attn_mask = torch.ones((Lq, Lk), dtype=torch.bool, device=device).triu(1)
                            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                    out_i = _sdpa_with_scale(
                        q_i, k_i, v_i,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p if torch.is_grad_enabled() and dropout_p > 0 else 0.0,
                        is_causal=is_causal_slice,
                        softmax_scale=softmax_scale,
                    )

                outs.append(out_i.squeeze(0).transpose(0, 1).contiguous())

            return torch.cat(outs, dim=0)

        # Final fallback: your original SDPA-based varlen implementation
        return _flash_varlen_fallback(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )



# class FlashAttentionVarlen(nn.Module):
#     def tflops(self, args, kwargs, output) -> float:
#         cu_seqlens_q = kwargs["cu_seqlens_q"]
#         cu_seqlens_k = kwargs["cu_seqlens_k"]
#         _, h, d = output.shape
#         seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
#         seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
#         return h * (4 * d * (seqlens_q * seqlens_k).sum())

#     def forward(self, *args, **kwargs):
#         kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
#         return flash_attn_varlen_func(*args, **kwargs)