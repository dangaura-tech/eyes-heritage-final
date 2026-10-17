"""AnyChange-style SAM features (Zheng et al., NeurIPS 2024): point-prompted masks per date + bitemporal latent
matching of SAM image embeddings inside each date's mask. Produces symmetric pair features for Model A."""
import numpy as np, torch, torch.nn.functional as F, pandas as pd

SAM_ID = "facebook/sam-vit-huge"
CROP = 256          # 128 m around the point
GRID = 64           # SAM embedding grid (2 m per cell for a 128 m crop)


def load_sam(device="cuda"):
    from transformers import SamModel, SamProcessor
    return SamModel.from_pretrained(SAM_ID, dtype=torch.float16).to(device).eval(), SamProcessor.from_pretrained(SAM_ID)


@torch.no_grad()
def sam_tiles(model, proc, tiles, device="cuda", bs=8):
    """tiles: (T,384,384,4) uint8. Returns emb (T,256,64,64) fp16, mask (T,64,64) bool, score (T,), valid (T,64,64)."""
    o = (tiles.shape[1] - CROP) // 2
    crops = [t[o:o + CROP, o:o + CROP] for t in tiles]
    E, M, S, V = [], [], [], []
    for s in range(0, len(crops), bs):
        cb = crops[s:s + bs]
        imgs = [c[..., :3] for c in cb]
        pts = [[[CROP / 2, CROP / 2]] for _ in cb]
        inp = proc(images=imgs, input_points=[[p] for p in pts], return_tensors="pt").to(device)
        emb = model.get_image_embeddings(inp["pixel_values"].half())
        out = model(image_embeddings=emb, input_points=inp["input_points"].half(), multimask_output=True)
        masks = proc.image_processor.post_process_masks(out.pred_masks.float().cpu(), inp["original_sizes"].cpu(),
                                                        inp["reshaped_input_sizes"].cpu())
        sc = out.iou_scores.float().cpu()[:, 0]                       # (b,3)
        for b in range(len(cb)):
            k = int(sc[b].argmax()); m = masks[b][0, k].float()       # (CROP,CROP)
            M.append(F.adaptive_avg_pool2d(m[None, None], GRID)[0, 0].numpy() > 0.5)
            V.append(F.adaptive_avg_pool2d(torch.from_numpy((cb[b][..., 3] > 0).astype(np.float32))[None, None], GRID)[0, 0].numpy())
            S.append(float(sc[b, k]))
        E.append(emb.cpu())
    return torch.cat(E).numpy(), np.stack(M), np.array(S), np.stack(V)


def _mean_in(emb, m):
    if m.sum() < 1:
        m = np.zeros_like(m); m[GRID // 2 - 1:GRID // 2 + 1, GRID // 2 - 1:GRID // 2 + 1] = True
    return emb[:, m].astype(np.float32).mean(1)


def _cosd(a, b):
    return float(1 - (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def pair_feats(pid, emb, mask, score, valid):
    rows = []
    c = GRID // 2
    cen = np.zeros((GRID, GRID), bool); cen[c - 2:c + 2, c - 2:c + 2] = True
    T = len(emb)
    for i in range(T):
        for j in range(i + 1, T):
            mi, mj = mask[i], mask[j]
            inter = (mi & mj).sum(); uni = (mi | mj).sum()
            vv = (valid[i] > 0.5) & (valid[j] > 0.5)
            ai, aj = mi.sum() / GRID ** 2, mj.sum() / GRID ** 2
            d_i = _cosd(_mean_in(emb[i], mi), _mean_in(emb[j], mi))     # date-i object, seen at date j
            d_j = _cosd(_mean_in(emb[i], mj), _mean_in(emb[j], mj))
            d_c = _cosd(_mean_in(emb[i], cen), _mean_in(emb[j], cen))
            ei = emb[i].astype(np.float32).reshape(emb.shape[1], -1); ej = emb[j].astype(np.float32).reshape(emb.shape[1], -1)
            cosmap = (ei * ej).sum(0) / (np.linalg.norm(ei, axis=0) * np.linalg.norm(ej, axis=0) + 1e-6)
            cosmap = cosmap.reshape(GRID, GRID)
            rows.append(dict(point_id=pid, i=i, j=j, sam_iou=float(inter / max(uni, 1)),
                             sam_area_min=float(min(ai, aj)), sam_area_max=float(max(ai, aj)),
                             sam_area_absdiff=float(abs(ai - aj)), sam_score_min=float(min(score[i], score[j])),
                             sam_lat_max=max(d_i, d_j), sam_lat_mean=(d_i + d_j) / 2, sam_lat_cen=d_c,
                             sam_cos_cen16=float(1 - cosmap[c - 8:c + 8, c - 8:c + 8][vv[c - 8:c + 8, c - 8:c + 8]].mean())
                             if vv[c - 8:c + 8, c - 8:c + 8].any() else 0.0,
                             sam_cos_p90=float(np.percentile(1 - cosmap[vv], 90)) if vv.any() else 0.0))
    return rows


def sam_pair_table(pts, tile_loader, log=print):
    model, proc = load_sam()
    rows = []
    for n, (pid, site) in enumerate(zip(pts.point_id, pts.site)):
        tiles = tile_loader(site, pid)
        emb, mask, score, valid = sam_tiles(model, proc, tiles)
        rows += pair_feats(pid, emb, mask, score, valid)
        if n % 50 == 0:
            log(f"sam {n}/{len(pts)}")
    return pd.DataFrame(rows)
