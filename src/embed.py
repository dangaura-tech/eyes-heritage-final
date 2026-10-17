"""Frozen ViT feature extraction with 8x dihedral views. Token grids are mapped back to the canonical frame
so that views can be mixed independently between dates without breaking spatial correspondence."""
import numpy as np, torch, torch.nn.functional as F, cv2

BACKBONES = {
    "dinov2-l": dict(hf="facebook/dinov2-large", patch=14, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "dinov2-g": dict(hf="facebook/dinov2-giant", patch=14, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "dinov3-sat-l": dict(hf="facebook/dinov3-vitl16-pretrain-sat493m", patch=16,
                         mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    "dinov3-web-l": dict(hf="facebook/dinov3-vitl16-pretrain-lvd1689m", patch=16,
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # DINOv3 via the public timm re-uploads (DINOv3 licence; the facebook/* repos are gated)
    "dinov3s-l": dict(timm="hf-hub:timm/vit_large_patch16_dinov3.sat493m", patch=16,
                      mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    "dinov3w-l": dict(timm="hf-hub:timm/vit_large_patch16_dinov3.lvd1689m", patch=16,
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    "dinov3s-7b": dict(timm="hf-hub:timm/vit_7b_patch16_dinov3.sat493m", patch=16,
                       mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    # RemoteCLIP (remote-sensing CLIP, ViT-L/14 at a fixed 224 px input) via open_clip
    "remoteclip-l": dict(openclip=("ViT-L-14", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"), patch=14,
                         mean=(0.481, 0.458, 0.408), std=(0.269, 0.261, 0.276)),
}
# input configs: crop side (px of 0.5 m) -> model grid tokens; output stored at GRID x GRID
CONFIGS = {"B1": dict(crop=256, grid_in=32, grid_out=16), "B2": dict(crop=128, grid_in=16, grid_out=16),
           "B1c": dict(crop=256, grid_in=16, grid_out=16)}     # 128 m at a fixed 224 px input (CLIP-style ViTs)


class OpenClipVisual(torch.nn.Module):
    """Wraps an open_clip ViT so that encode() can treat it like the timm path."""
    def __init__(self, visual):
        super().__init__(); self.visual = visual
        self.blocks = visual.transformer.resblocks; self.embed_dim = visual.transformer.width

    def forward_intermediates(self, x, indices, **kw):
        out = self.visual.forward_intermediates(x, indices=indices, output_fmt="NCHW", intermediates_only=True,
                                                output_extra_tokens=True)
        sp, ex = out["image_intermediates"], out.get("image_intermediates_prefix")
        return [(sp[k], ex[k] if ex is not None else sp[k].flatten(2).mean(-1)[:, None]) for k in range(len(sp))]


def load_backbone(name, device="cuda"):
    if "openclip" in BACKBONES[name]:
        import open_clip
        from huggingface_hub import hf_hub_download
        arch, repo, fn = BACKBONES[name]["openclip"]
        m = open_clip.create_model(arch, pretrained=None)
        sd = torch.load(hf_hub_download(repo, fn), map_location="cpu")
        m.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
        return OpenClipVisual(m.visual).to(device).half().eval()
    if "timm" in BACKBONES[name]:
        import timm
        m = timm.create_model(BACKBONES[name]["timm"], pretrained=True, num_classes=0)
        return m.to(device).half().eval()
    from transformers import AutoModel
    m = AutoModel.from_pretrained(BACKBONES[name]["hf"], torch_dtype=torch.float16).to(device).eval()
    return m


def is_timm(m):
    return hasattr(m, "blocks") and hasattr(m, "forward_intermediates")


def n_layers(m):
    return len(m.blocks) if is_timm(m) else m.config.num_hidden_layers


def hidden(m):
    return m.embed_dim if is_timm(m) else m.config.hidden_size


def dihedral(x, k):
    """x: (..., H, W). k in 0..7: rot90 k%4 then optional flip."""
    y = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if k >= 4 else y


def inv_dihedral(x, k):
    y = torch.flip(x, dims=(-1,)) if k >= 4 else x
    return torch.rot90(y, -(k % 4), dims=(-2, -1))


def prep(rgba_list, crop, side, mean, std, standardise=True):
    """Centre crop, per-crop per-channel standardisation over valid pixels, resize to `side`.
    Returns (B,3,side,side) float tensor and (B,side,side) validity."""
    xs, ms = [], []
    for a in rgba_list:
        o = (a.shape[0] - crop) // 2
        c = a[o:o + crop, o:o + crop]
        v = c[..., 3] > 0
        f = c[..., :3].astype(np.float32) / 255.0
        if standardise and v.sum() > 50:
            mu = f[v].mean(0); sd = f[v].std(0) + 1e-3
            f = (f - mu) / sd
        else:
            f = (f - np.array(mean, np.float32)) / np.array(std, np.float32)
        f[~v] = 0.0
        f = cv2.resize(f, (side, side), interpolation=cv2.INTER_CUBIC)
        vv = cv2.resize(v.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA)
        xs.append(f.transpose(2, 0, 1)); ms.append(vv)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))


@torch.no_grad()
def encode(model, x, patch, grid_in, grid_out, mid_layer):
    """x (B,3,S,S) on device fp16. Returns tokens (B, grid_out^2, 2D) [last | mid] and cls (B, D)."""
    if is_timm(model):
        L = len(model.blocks)
        outs = model.forward_intermediates(x, indices=[mid_layer - 1, L - 1], norm=True, output_fmt="NCHW",
                                           intermediates_only=True, return_prefix_tokens=True)
        (g_mid, _), (g_last, pre) = outs
        pre = pre if pre.dim() == 3 else pre[:, None]
        def pool(t):
            return F.adaptive_avg_pool2d(t.float(), grid_out).to(t.dtype) if grid_out != grid_in else t
        return torch.cat([pool(g_last), pool(g_mid)], 1), pre[:, 0]
    out = model(pixel_values=x, output_hidden_states=True)
    hs = out.hidden_states
    n_reg = getattr(model.config, "num_register_tokens", 0) or 0
    def grid(h):
        t = h[:, 1 + n_reg:, :]
        B, N, D = t.shape
        t = t.transpose(1, 2).reshape(B, D, grid_in, grid_in)
        if grid_out != grid_in:
            t = F.adaptive_avg_pool2d(t.float(), grid_out).to(h.dtype)
        return t
    last = out.last_hidden_state
    g_last = grid(last); g_mid = grid(hs[mid_layer])
    cls = last[:, 0, :]
    return torch.cat([g_last, g_mid], 1), cls      # (B, 2D, g, g)


def token_validity(valid, grid):
    """valid (B,S,S) -> fraction valid per token (B, grid, grid)."""
    return F.adaptive_avg_pool2d(valid[:, None].float(), grid)[:, 0]


def extract_to_dir(index, out, bb, cfgn, variant="tiles", model=None, log=print):
    """index: DataFrame(point_id, site, t) in tile order. Writes tokens/cls/valid/index/meta into `out`."""
    import json, time
    import pandas as pd
    from pathlib import Path
    from common import tile_path
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    B = BACKBONES[bb]; C = CONFIGS[cfgn]
    model = model or load_backbone(bb)
    L = n_layers(model); mid = int(round(2 * L / 3))
    D = hidden(model); G = C["grid_out"]; side = C["grid_in"] * B["patch"]
    N = len(index)
    log(f"{bb} {cfgn} layers {L} mid {mid} D {D} side {side} N {N}")
    tok = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float16, shape=(N, 8, G * G, 2 * D))
    cls = np.lib.format.open_memmap(out / "cls.npy", mode="w+", dtype=np.float16, shape=(N, 8, D))
    val = np.zeros((N, G * G), np.float16)
    bs = 32 if side <= 256 else 16
    if "-g" in bb: bs //= 2
    t0 = time.time(); cur = None; curz = None
    rows = list(index.itertuples())
    for s in range(0, N, bs):
        batch = rows[s:s + bs]; imgs = []
        for r in batch:
            if cur != r.point_id:
                curz = np.load(tile_path(r.site, r.point_id), allow_pickle=True)[variant]; cur = r.point_id
            imgs.append(curz[r.t])
        x, v = prep(imgs, C["crop"], side, B["mean"], B["std"])
        val[s:s + len(batch)] = token_validity(v, G).reshape(len(batch), -1).numpy()
        x = x.cuda().half()
        tb = np.zeros((len(batch), 8, G * G, 2 * D), np.float16); cb = np.zeros((len(batch), 8, D), np.float16)
        for k in range(8):
            g, c = encode(model, dihedral(x, k), B["patch"], C["grid_in"], G, mid)
            g = inv_dihedral(g, k)
            tb[:, k] = g.flatten(2).transpose(1, 2).cpu().numpy(); cb[:, k] = c.cpu().numpy()
        tok[s:s + len(batch)] = tb; cls[s:s + len(batch)] = cb
        if s % (bs * 40) == 0:
            log(f"{s}/{N} {time.time() - t0:.0f}s")
    tok.flush(); cls.flush()
    np.save(out / "valid.npy", val)
    index.reset_index(drop=True).to_parquet(out / "index.parquet")
    json.dump(dict(backbone=bb, hf=B.get("hf", B.get("timm", str(B.get("openclip")))), config=cfgn, mid_layer=mid, layers=L, D=D, grid=G, side=side,
                   variant=variant), open(out / "meta.json", "w"))
    (out / "done").touch()
    return model


def center_pool(d):
    """Centre-pooled (~16 m radius) token embeddings per tile x view -> d/center.npy"""
    import json
    from pathlib import Path
    d = Path(d)
    meta = json.load(open(d / "meta.json")); G = meta["grid"]
    tok = np.load(d / "tokens.npy", mmap_mode="r"); val = np.load(d / "valid.npy").astype(np.float32)
    tok_m = 128 / G if meta["config"].startswith("B1") else 64 / G
    r = max(1, int(round(16 / tok_m)))
    yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij"); c = (G - 1) / 2
    cm = ((np.abs(yy - c) <= r) & (np.abs(xx - c) <= r)).flatten()
    N = tok.shape[0]
    out = np.zeros((N, 8, tok.shape[-1]), np.float16)
    for s in range(0, N, 64):
        t = np.asarray(tok[s:s + 64][:, :, cm], np.float32)
        w = val[s:s + 64][:, cm][:, None, :, None] + 1e-3
        out[s:s + 64] = ((t * w).sum(2) / w.sum(2)).astype(np.float16)
    np.save(d / "center.npy", out)
    return out


def dino_pair_feats(d):
    """Cosine / L2 distances between dates (CLS and centre-pooled) for all i<j pairs of every point."""
    import pandas as pd
    from pathlib import Path
    d = Path(d)
    cls = np.load(d / "cls.npy", mmap_mode="r"); cen = np.load(d / "center.npy").astype(np.float32)
    idx = pd.read_parquet(d / "index.parquet"); D = cen.shape[-1] // 2
    row = {(p, t): k for k, (p, t) in enumerate(zip(idx.point_id, idx.t))}
    cosd = lambda a, b: 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-6)
    clsf = np.asarray(cls, np.float32).mean(1); cm = cen.mean(1)
    rows = []
    for pid, g in idx.groupby("point_id", sort=False):
        ts = sorted(g.t)
        for a, i in enumerate(ts):
            for j in ts[a + 1:]:
                ki, kj = row[(pid, i)], row[(pid, j)]
                rows.append(dict(point_id=pid, i=i, j=j, dino_cls_cos=float(cosd(clsf[ki], clsf[kj])),
                                 dino_cen_cos=float(cosd(cm[ki, :D], cm[kj, :D])),
                                 dino_cen_mid_cos=float(cosd(cm[ki, D:], cm[kj, D:])),
                                 dino_cen_l2=float(np.linalg.norm(cm[ki, :D] - cm[kj, :D]) / np.sqrt(D))))
    return pd.DataFrame(rows)
