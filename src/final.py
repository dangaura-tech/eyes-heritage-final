"""Final training on all 776 points and member-wise prediction for predict.py.

A member spec is a dict: {"name": ..., "kind": "A"|"Apt"|"B"|"D", "cfg": {...}, "budget": n_iter/epochs}.
Each trained member is stored in models/final/<name>/ as cfg.json + model.pt (data references stripped)."""
import json, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
import common
from common import CACHE, MODELS, seed_all, load_points
from pairs import CORE_YEAR


def _tables_for(kind, cfg, pts, unary, pairs, emb_dir):
    from modelA import PairTable
    extra = None
    if cfg.get("dino"):
        from embed import dino_pair_feats
        extra = dino_pair_feats(Path(emb_dir[cfg["dino"]]))
    return PairTable(pts, unary, pairs, extra)


def train_member(spec, seeds, out_root=MODELS / "final", train_idx=None):
    pts = load_points(); kind, cfg = spec["kind"], spec["cfg"]
    out = Path(out_root) / spec["name"]; out.mkdir(parents=True, exist_ok=True)
    allidx = np.arange(len(pts)) if train_idx is None else np.asarray(train_idx)
    if kind in ("A", "Apt"):
        from modelA import PairTable, ModelA
        from modelApt import PointTable, ModelApt
        U = pd.read_parquet(CACHE / f"featA_{cfg.get('variant', 'tiles')}_unary.parquet")
        P = pd.read_parquet(CACHE / f"featA_{cfg.get('variant', 'tiles')}_pairs.parquet")
        ex = pd.read_parquet(CACHE / f"dinopair_{cfg['dino']}.parquet") if cfg.get("dino") else None
        if cfg.get("sam"):
            sp = pd.read_parquet(CACHE / "sampair.parquet")
            ex = sp if ex is None else ex.merge(sp, on=["point_id", "i", "j"], how="left")
        T = PairTable(pts, U, P, ex)
        objs = []
        for s in seeds:
            if kind == "A":
                m = ModelA(T, cfg); m.fit_final(allidx, s, spec["budget"]); objs.append(m.models)
            else:
                m = ModelApt(PointTable(T), cfg); m.fit_final(allidx, s, spec["budget"]); objs.append(m.m)
        torch.save(dict(kind=kind, cfg=cfg, models=objs, cols=T.names), out / "model.pt")
    elif kind in ("B", "D"):
        nets = []
        if kind == "B":
            from modelB import EmbStore, ModelB
            S = EmbStore(CACHE / "emb" / cfg["emb"]); m = ModelB(pts, S, cfg)
        else:
            from modelD import SeqStore, ModelD
            S = SeqStore(CACHE / "emb" / cfg["emb"], pts, [CACHE / "emb" / x for x in cfg.get("extra_emb", [])]); m = ModelD(pts, S, cfg)
        for s in seeds:
            m.fit_final(allidx, s, spec["budget"])
        nets = [n.cpu().state_dict() for n in m.nets]
        arch = dict(G=getattr(S, "G", None), D2=getattr(S, "D2", None),
                    d_in=(m._dims()[0] if kind == "D" else None), n_date=(m._dims()[1] if kind == "D" else None), emb_meta=json.load(open(CACHE / "emb" / cfg["emb"] / "meta.json")))
        torch.save(dict(kind=kind, cfg=cfg, nets=nets, arch=arch), out / "model.pt")
    elif kind == "C":
        from modelC import ModelC
        from modelE import ModelE
        m = (ModelE if cfg.get("arch") == "unet" else ModelC)(pts, cfg); files = []
        for s in seeds:
            m.fit_final(allidx, s, spec["budget"])
            net = m.nets.pop()
            sd = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
                  for k, v in net.state_dict().items()
                  if cfg.get("full_ft") or cfg.get("arch") == "unet" or "lora" in k or k.startswith("head")}
            fn = f"net_seed{s}.pt"; torch.save(sd, out / fn); files.append(fn)   # one file per seed, saved immediately
            del net, sd; torch.cuda.empty_cache()
        torch.save(dict(kind=kind, cfg=cfg, nets=files), out / "model.pt")
    json.dump(spec, open(out / "cfg.json", "w"), indent=1, default=str)
    return out


def _ensure_embeddings(pts, work, needed, log=print):
    """Extract token caches for prediction points for every backbone/config in `needed`."""
    from embed import extract_to_dir, center_pool, load_backbone
    index = []
    for pid, site in zip(pts.point_id, pts.site):
        z = np.load(common.tile_path(site, pid), allow_pickle=True)
        index += [(pid, site, t) for t in range(len(z["dates"]))]
    index = pd.DataFrame(index, columns=["point_id", "site", "t"])
    dirs = {}
    models = {}
    for name in sorted(needed):
        bb, cfgn = name.split("_")[0], name.split("_")[1]
        d = Path(work) / "emb" / name
        if not (d / "done").exists():
            if bb not in models:
                models.clear(); torch.cuda.empty_cache(); models[bb] = load_backbone(bb)
            extract_to_dir(index, d, bb, cfgn, model=models[bb], log=log)
        if not (d / "center.npy").exists():
            center_pool(d)
        dirs[name] = d
    return dirs


def predict_members(pts, ens, mdir, work, use_year=False, tta=8, log=print):
    from modelA import PairTable, ModelA
    from modelApt import PointTable, ModelApt
    from modelB import EmbStore, ModelB, Head
    from modelD import SeqStore, ModelD, SeqNet
    mdir = Path(mdir); work = Path(work)
    CORE_YEAR.clear()
    if use_year and "year" in pts and pts.year.notna().any():
        CORE_YEAR.update({p: float(y) for p, y in zip(pts.point_id, pts.year) if pd.notna(y)})
    members = [m for m, w in ens["weights"].items() if w > 0]
    specs = {m: torch.load(mdir / m / "model.pt", weights_only=False) for m in members}
    needed = set()
    for s in specs.values():
        if s["kind"] in ("B", "D"): needed.add(s["cfg"]["emb"]); needed.update(s["cfg"].get("extra_emb", []))
        if s["kind"] == "E": needed.add(s["cfg"]["emb"]); needed.update(s["cfg"].get("extra_emb", []))
        if s["cfg"].get("dino"): needed.add(s["cfg"]["dino"])
    emb_dirs = _ensure_embeddings(pts, work, needed, log=log)
    featA = None
    out = {}
    idx = np.arange(len(pts))
    for name, s in specs.items():
        kind, cfg = s["kind"], s["cfg"]
        if kind in ("A", "Apt"):
            if featA is None:
                import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
                from p3_featA import run as featrun
                pre = work / "featA"
                if not Path(f"{pre}_pairs.parquet").exists():
                    featrun([(p, common.tile_path(si, p)) for p, si in zip(pts.point_id, pts.site)], str(pre))
                featA = (pd.read_parquet(f"{pre}_unary.parquet"), pd.read_parquet(f"{pre}_pairs.parquet"))
            from embed import dino_pair_feats
            ex = dino_pair_feats(emb_dirs[cfg["dino"]]) if cfg.get("dino") else None
            if cfg.get("sam"):
                sfn = work / "sampair.parquet"
                if not sfn.exists():
                    from samfeat import sam_pair_table
                    sam_pair_table(pts, lambda si, p: np.load(common.tile_path(si, p), allow_pickle=True)["tiles"],
                                   log=log).to_parquet(sfn)
                sp = pd.read_parquet(sfn)
                ex = sp if ex is None else ex.merge(sp, on=["point_id", "i", "j"], how="left")
            T = PairTable(pts, featA[0], featA[1], ex)
            assert T.names == s["cols"], "feature columns differ from training"
            Ps = []
            for mods in s["models"]:
                if kind == "A":
                    m = ModelA(T, cfg); m.models = mods; Ps.append(m.predict(idx))
                else:
                    m = ModelApt(PointTable(T), cfg); m.m = mods; Ps.append(m.predict(idx))
            out[name] = np.mean(Ps, 0)
        elif kind == "B":
            S = EmbStore(emb_dirs[cfg["emb"]]); m = ModelB(pts, S, cfg)
            a = s["arch"]; G = a["G"]
            m.nets = []
            for sd in s["nets"]:
                net = Head(a["D2"], G, d=cfg.get("d", 256), layers=cfg.get("layers", 2),
                           drop=cfg.get("drop", 0.1), use_mid=cfg.get("use_mid", True))
                net.load_state_dict(sd); m.nets.append(net.cuda().eval())
            m.cfg = dict(cfg, tta=tta); out[name] = m.predict(idx)
        elif kind == "C":
            from modelC import ModelC, Siamese
            from modelE import ModelE
            unet = cfg.get("arch") == "unet"
            m = (ModelE if unet else ModelC)(pts, dict(cfg, tta=tta)); m.nets = []
            for sd in s["nets"]:
                if isinstance(sd, str):     # per-seed checkpoint file (fp16) -> float32
                    sd = {k: (v.float() if v.is_floating_point() else v)
                          for k, v in torch.load(mdir / name / sd, map_location="cpu").items()}
                net = m.make_net() if unet else Siamese(bb=cfg.get("bb", "dinov2-l"), rank=cfg.get("rank", 16),
                                                         last_k=cfg.get("last_k", 8), full_ft=cfg.get("full_ft", False),
                                                         grad_ckpt=False)
                net.load_state_dict(sd, strict=False); m.nets.append(net.cuda().eval())
            out[name] = m.predict(idx)
            del m; torch.cuda.empty_cache()
        elif kind == "D":
            S = SeqStore(emb_dirs[cfg["emb"]], pts, [emb_dirs[x] for x in cfg.get("extra_emb", [])]); m = ModelD(pts, S, cfg)
            m.nets = []
            for sd in s["nets"]:
                net = SeqNet(s["arch"]["d_in"], d=cfg.get("d", 256), layers=cfg.get("layers", 2), drop=cfg.get("drop", 0.1),
                             delta=cfg.get("delta", False), n_date=s["arch"].get("n_date") or 3)
                net.load_state_dict(sd); m.nets.append(net.cuda().eval())
            out[name] = m.predict(idx, tta=tta)
        log(f"member {name}: mean p_change {1 - out[name][:, 3].mean():.3f}")
    return out
