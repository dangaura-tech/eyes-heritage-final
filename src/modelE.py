"""Model E: siamese U-Net change network. ConvNeXt-DINOv3 encoder (shared), per-scale fusion
[f_b, f_a, |f_a - f_b|], U-Net decoder to stride 4, dense 4-class change map, Gaussian centre pooling -> point logits.
Training/augmentation/pair sampling are inherited from Model C."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from modelC import ModelC, crop_pair, head_probs
from pairs import valid_dates, infer_pairs, aggregate
from embed import dihedral

ENC = "hf-hub:timm/convnext_small.dinov3_lvd1689m"


class _Chs:
    def __init__(self, c): self.c = c
    def channels(self): return self.c


class SatlasSwinV2B(nn.Module):
    """SatlasPretrain Swin-v2-B (NAIP aerial RGB, Apache-2.0) as a 4-level feature pyramid (strides 4/8/16/32)."""
    def __init__(self):
        super().__init__()
        import torchvision
        from huggingface_hub import hf_hub_download
        m = torchvision.models.swin_v2_b()
        m.load_state_dict(torch.load(hf_hub_download("torchgeo/swin_v2_b_naip_rgb_satlas",
                                                     "swin_v2_b_naip_rgb_satlas-685f45bd.pth"), map_location="cpu"))
        self.features = m.features
        self.feature_info = _Chs([128, 256, 512, 1024])

    def forward(self, x):
        out = []; f = x
        for i, layer in enumerate(self.features):
            f = layer(f)
            if i in (1, 3, 5, 7):
                out.append(f.permute(0, 3, 1, 2).contiguous())
        return out


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                                  nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SiameseUNet(nn.Module):
    full_ckpt = True

    def __init__(self, enc=ENC, dec=(256, 128, 64), drop=0.1, sigma_px=32.0):
        super().__init__()
        if enc == "satlas-swinv2b":
            self.bb = SatlasSwinV2B()
        else:
            import timm
            self.bb = timm.create_model(enc, pretrained=True, features_only=True)
        chs = self.bb.feature_info.channels()                     # strides 4, 8, 16, 32
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * c, c, 1), nn.GroupNorm(8, c), nn.GELU()) for c in chs])
        self.head = nn.ModuleDict(dict(
            up3=Up(chs[3], chs[2], dec[0]), up2=Up(dec[0], chs[1], dec[1]), up1=Up(dec[1], chs[0], dec[2]),
            cls=nn.Sequential(nn.Dropout2d(drop), nn.Conv2d(dec[2], 4, 1)),
            att=nn.Conv2d(dec[2], 1, 1)))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_px / 4)))   # in stride-4 cells
        self.head["fuse"] = self.fuse

    def dense(self, xb, xa):
        fb = self.bb(xb); fa = self.bb(xa)
        f = [m(torch.cat([b, a, (a - b).abs()], 1)) for m, b, a in zip(self.fuse, fb, fa)]
        x = self.head["up3"](f[3], f[2]); x = self.head["up2"](x, f[1]); x = self.head["up1"](x, f[0])
        return self.head["cls"](x), self.head["att"](x)

    def forward(self, xb, xa, m, k=None):
        if k is not None:
            xb = torch.stack([dihedral(xb[n], int(k[n])) for n in range(len(xb))])
            xa = torch.stack([dihedral(xa[n], int(k[n])) for n in range(len(xa))])
        logit, att = self.dense(xb, xa)                           # (B,4,H,W), (B,1,H,W); centre is invariant to k
        B, _, H, W = logit.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=logit.device), torch.arange(W, device=logit.device), indexing="ij")
        d2 = ((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2).float()
        s = att[:, 0] - d2[None] / (2 * torch.exp(self.log_sigma) ** 2)
        mm = F.interpolate(m.view(B, 1, int(m.shape[1] ** 0.5), -1), size=(H, W), mode="nearest")[:, 0] > 0.5
        if k is not None:
            mm = torch.stack([dihedral(mm[n], int(k[n])) for n in range(B)])
        mm = mm | (~mm.flatten(1).any(1))[:, None, None]
        s = s.masked_fill(~mm, -1e4)
        w = torch.softmax(s.flatten(1), 1).view(B, 1, H, W)
        return (logit * w).sum((2, 3))


class SiameseAdapt(nn.Module):
    """AdaptFormer pretrained on LEVIR-CD (building change, 0.5 m). The change head is widened from 2 to 2 + 4 output
    channels (pretrained change channels kept). The pretrained change evidence (logit 1 - logit 0) drives attention
    pooling around the point; the 4 new channels give the class logits. Fully fine-tuned."""
    full_ckpt = True

    def __init__(self, sigma_px=32.0):
        super().__init__()
        import json as _json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from third_party.adaptformer.configuration_adaptformer import AdaptFormerConfig
        from third_party.adaptformer.modeling_adaptformer import AdaptFormerForChangeDetection
        repo = "deepang/adaptformer-LEVIR-CD"
        cfg = AdaptFormerConfig(**{k: v for k, v in _json.load(open(hf_hub_download(repo, "config.json"))).items()
                                   if k not in ("auto_map", "architectures", "model_type", "torch_dtype", "transformers_version")})
        self.af = AdaptFormerForChangeDetection(cfg)
        self.af.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=True)
        old = self.af.decoder.head.out
        new = nn.Conv2d(old.in_channels, 6, 3, 1, 1)
        with torch.no_grad():
            new.weight[:2] = old.weight; new.bias[:2] = old.bias
        self.af.decoder.head.out = new
        self.bb = self.af           # param_groups: whole AdaptFormer at lr_lora
        self.head = nn.Module()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_px / 4)))
        self.att_w = nn.Parameter(torch.tensor(1.0))

    def forward(self, xb, xa, m, k=None):
        if k is not None:
            xb = torch.stack([dihedral(xb[n], int(k[n])) for n in range(len(xb))])
            xa = torch.stack([dihedral(xa[n], int(k[n])) for n in range(len(xa))])
        out = self.af(pixel_valuesA=xb, pixel_valuesB=xa, return_dict=True).logits
        logit = out[:, 2:]; ch = out[:, 1] - out[:, 0]
        B, _, H, W = logit.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=logit.device), torch.arange(W, device=logit.device), indexing="ij")
        d2 = ((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2).float()
        s = self.att_w * ch - d2[None] / (2 * (torch.exp(self.log_sigma) * H / 64) ** 2)
        mm = F.interpolate(m.view(B, 1, int(m.shape[1] ** 0.5), -1), size=(H, W), mode="nearest")[:, 0] > 0.5
        if k is not None:
            mm = torch.stack([dihedral(mm[n], int(k[n])) for n in range(B)])
        mm = mm | (~mm.flatten(1).any(1))[:, None, None]
        s = s.masked_fill(~mm, -1e4)
        w = torch.softmax(s.flatten(1), 1).view(B, 1, H, W)
        return (logit * w).sum((2, 3))


class ModelE(ModelC):
    @property
    def side(self):
        return self.cfg.get("side", 256)

    def make_net(self):
        if self.cfg.get("enc") == "adaptformer":
            return SiameseAdapt()
        return SiameseUNet(enc=self.cfg.get("enc", ENC), drop=self.cfg.get("drop", 0.1))

    @torch.no_grad()
    def _predict1(self, idx):
        net = self.net; net.eval(); tta = self.cfg.get("tta", 8)
        Ps, owner = [], []
        rng = np.random.RandomState(0)
        for n, kk in enumerate(idx):
            pid = self.pts.point_id.iloc[kk]; years, bad, q = self.meta[pid]
            vi, _ = valid_dates(bad, q); pr = infer_pairs(pid, vi, years)
            T = self.tiles[pid]; X, V = {}, {}
            for t in sorted({t for p in pr for t in p}):
                xb, vb, _, _ = crop_pair(T[t], T[t], rng, False, self.side, 0.0, 1.0, self.crop); X[t] = xb; V[t] = vb
            xb = torch.stack([X[i] for i, _ in pr]).to(self.dev); xa = torch.stack([X[j] for _, j in pr]).to(self.dev)
            m = torch.stack([F.adaptive_avg_pool2d(torch.minimum(V[i], V[j])[None, None], 16).flatten() for i, j in pr]).to(self.dev)
            acc = 0; cb = self.cfg.get("pred_chunk", 8)
            for v in range(tta):
                outs = []
                for c0 in range(0, len(pr), cb):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = net(xb[c0:c0 + cb], xa[c0:c0 + cb], m[c0:c0 + cb], torch.full((len(pr[c0:c0 + cb]),), v)).float()
                    outs.append(head_probs(z, self.cfg.get("two_head", False)))
                acc = acc + torch.cat(outs)
            Ps.append((acc / tta).cpu().numpy()); owner += [n] * len(pr)
            if not hasattr(self, "last_pairs"): self.last_pairs = {}
            self.last_pairs[pid] = (pr, Ps[-1])
        return aggregate(np.concatenate(Ps), np.array(owner), len(idx), mode=self.cfg.get("agg", "top2"), k=2)
