"""
TDN (Temporal Difference Network, CVPR'21) ResNet-50 pretrained on
Something-Something V2, adapted for the 33-class "What Happens Next?" challenge.

Why this expert
---------------
SSv2 is motion-dominated *by design* (appearance is deliberately uninformative —
same banal objects across classes). V-JEPA2 / VideoMAE are RGB-appearance models
and collapse exactly on the real-vs-pretended pairs (e.g. "Picking up" 12% vs
"Pretending to pick up" 45%): the only cue is whether the object actually leaves
the surface. TDN models *explicit temporal differences* (a short-term local-motion
TDM at the stem + a long-term cross-segment TDM across stages). Different input
modality => genuinely decorrelated errors, which is what the XGBoost stack needs
(it only gains +1% today because the experts are correlated). TDN solo will land
BELOW V-JEPA2; that is fine — the goal is decorrelation, not a higher solo score.

Vendored official code (plain PyTorch, no mmcv/mmaction): src/models/tdn_vendor/.
The released SSv2 R50-8f checkpoint (top-1 64.0%) was trained as
nn.DataParallel(TSN(...)). We rebuild the exact TSN(num_class=174,
num_segments=8, resnet50, dropout=0.5) so the state_dict matches verbatim,
assert a strict load (a vendoring-fidelity check — it fails loudly if the
vendored arch drifts from the checkpoint), then do head surgery: new_fc ->
Identity (segment consensus then averages 2048-d features) + a fresh 33-class
MLP head, the same pattern as slowfast_r50.py.

Input adaptation (done IN forward; the shared VideoFrameDataset is untouched)
---------------------------------------------------------------------------
The dataset yields the standard (B, T, C, H, W). TDN needs `num_segments` local
windows of 5 consecutive frames -> (B*num_segments, 15, H, W). We pick
`num_segments` anchor frames spread across T and take a +/-2 neighbourhood per
anchor (clamped to the clip). With a dense first-60% pool (num_frames ~16-19)
the short-term TDM gets real local motion; with very few frames the windows
collapse, the short-term differences -> 0 and the model degrades gracefully to
the long-term (cross-segment) TDM. So this stays usable even if the test set is
restricted to a handful of frames.

Input shape  : (B, T, C, H, W)
Output shape : (B, num_classes)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# The vendored TDN code uses absolute `from ops.X import Y` imports. Make that
# package resolvable WITHOUT editing the vendored files (keeps them byte-faithful
# to the upstream commit, which is what guarantees the checkpoint keys match).
# Scoped to this module; `ops` is not used anywhere else in this repo.
_VENDOR_ROOT = Path(__file__).resolve().parent / "tdn_vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from ops.models import TSN  # noqa: E402  (vendored TDN; path injected just above)

# Official model-zoo link for the pretrained weights, surfaced in the error
# message so the download step is self-documenting.
_SSV2_R50_8F_DRIVE = (
    "https://drive.google.com/drive/folders/14pZ1W_Mh8nR4e1ziEz7AFpgbz-79HpIC"
)


class TDNSSv2Finetune(nn.Module):

    SSV2_NUM_CLASSES = 174

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        pretrained_ckpt: str | None = None,
        num_segments: int = 8,
        frames_per_segment: int = 5,
        dropout: float = 0.5,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_segments = int(num_segments)
        self.frames_per_segment = int(frames_per_segment)
        if self.frames_per_segment != 5:
            raise ValueError(
                "TDN's short-term TDM is hard-wired for exactly 5 frames/segment "
                f"(got {self.frames_per_segment}). Change the architecture, not this arg."
            )

        # Rebuild the exact network the SSv2 checkpoint was saved from so the
        # state_dict matches 1:1. dropout>0 is what makes TSN create `new_fc`
        # (the real 174-class classifier); the exact p has no params so it does
        # not affect key matching. partial_bn=False -> no surprise BN freezing.
        self.tsn = TSN(
            self.SSV2_NUM_CLASSES,
            self.num_segments,
            "RGB",
            base_model="resnet50",
            consensus_type="avg",
            before_softmax=True,
            dropout=dropout,
            partial_bn=False,
            print_spec=False,
            pretrain="imagenet",
        )

        if pretrained:
            self._load_ssv2_checkpoint(pretrained_ckpt)

        # Pre-hook on layer2_bak: syncs CUDA before layer2 reads `x`, which is the
        # result of F.interpolate + element-wise add in tdn_net.py (async race on
        # PyTorch 2.11+/CUDA 12.8). Fires inside tsn.forward, no vendored code touched.
        def _cuda_sync_hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        self.tsn.base_model.layer2_bak.register_forward_pre_hook(_cuda_sync_hook)

        feature_dim = self.tsn.new_fc.in_features  # 2048 for R50
        # Head surgery (slowfast_r50.py pattern): drop the 174-class classifier;
        # consensus then averages 2048-d features over segments, our MLP maps 33.
        self.tsn.new_fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _load_ssv2_checkpoint(self, ckpt_path: str | None) -> None:
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"TDN SSv2 pretrained checkpoint not found: {ckpt_path!r}\n"
                f"Download R50-8f (top-1 64.0%) from the official model zoo:\n"
                f"  gdown --folder {_SSV2_R50_8F_DRIVE}\n"
                f"then set model.pretrained_ckpt to the .pth path "
                f"(store under /Data/marius.truquin/Model_checkpoints)."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Trained under nn.DataParallel -> strip the "module." prefix.
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        # strict=True is intentional: a key/shape mismatch means the vendored
        # architecture drifted from the released checkpoint -> fail loudly here
        # rather than silently training a half-random net.
        self.tsn.load_state_dict(state, strict=True)

    def _build_segments(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B*num_segments, frames_per_segment*C, H, W).

        num_segments anchors spread across the clip; each segment is the anchor
        +/- 2 neighbours (clamped). Channel order per segment is
        [f_{a-2}, f_{a-1}, f_a, f_{a+1}, f_{a+2}] so TDN_Net.forward's
        x[:,0:3]..x[:,12:15] line up and the main path uses the centre frame.
        Batch-major / segment-minor ordering matches TSN's consensus reshape.
        """
        B, T, C, H, W = x.shape
        half = self.frames_per_segment // 2
        anchors = torch.linspace(0, T - 1, self.num_segments, device=x.device)
        anchors = anchors.round().long()
        offsets = torch.arange(-half, half + 1, device=x.device)
        idx = (anchors[:, None] + offsets[None, :]).clamp_(0, T - 1)  # (S, 5)
        seg = x[:, idx, :, :, :]  # (B, S, 5, C, H, W)
        seg = seg.reshape(B, self.num_segments, self.frames_per_segment * C, H, W)
        return seg.reshape(B * self.num_segments, self.frames_per_segment * C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg = self._build_segments(x)        # (B*S, 15, H, W)
        feat = self.tsn(seg)                 # (B, 2048) after segment consensus
        return self.head(feat)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_lr = base_lr * backbone_lr_factor
        head_params = list(self.head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.tsn.parameters()
            if p.requires_grad and id(p) not in head_ids
        ]
        return [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
