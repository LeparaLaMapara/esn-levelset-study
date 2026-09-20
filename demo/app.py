"""The demo service: run the level set live, and compare what each model predicts.

Two things are on the page, and the page says which is which:

* The Chan-Vese evolution runs **live** on the server for whatever parameters
  the visitor sets. It is the same code the study generated its data with
  (`levelset.chanvese`), on CPU here, which is fast enough at 64x64.
* The model predictions come from the **trained checkpoints** in `demo/models`,
  loaded once and run on CPU. They are the same weights the paper reports.

No GPU is needed to serve this, and nothing is faked: if a checkpoint is
missing, the model is absent from the page rather than replaced by a stand-in.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from levelset.chanvese import THESIS, evolve  # noqa: E402
from levelset.models import build  # noqa: E402

app = FastAPI(title="Learning the level set: reservoirs against trained recurrent networks")

SAMPLES = np.load(ROOT / "assets" / "samples.npz") if (ROOT / "assets" / "samples.npz").exists() else None
RESULTS = json.loads((ROOT / "assets" / "results.json").read_text()) if (ROOT / "assets" / "results.json").exists() else {}
MODELS: dict[str, torch.nn.Module] = {}
META = json.loads((ROOT / "models" / "manifest.json").read_text()) if (ROOT / "models" / "manifest.json").exists() else {}


def load_models() -> None:
    """Load every checkpoint named in the manifest, once, on CPU."""
    for name, spec in META.get("models", {}).items():
        kwargs = dict(spec.get("kwargs", {}))
        model = build(spec["arch"], **kwargs)
        state = torch.load(ROOT / "models" / spec["file"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        MODELS[name] = model


load_models()


def png(arr: np.ndarray, scale: int = 3) -> str:
    """A small array as a base64 PNG, nearest neighbour upscaled to stay crisp."""
    a = np.clip(arr, 0, 1)
    img = Image.fromarray((a * 255).astype(np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class EvolveRequest(BaseModel):
    sample: int = 0
    mu: float = THESIS["mu"]
    dt: float = THESIS["dt"]
    iterations: int = 100
    frames: int = 6


@app.get("/api/samples")
def samples() -> JSONResponse:
    if SAMPLES is None:
        return JSONResponse({"samples": []})
    imgs = SAMPLES["images"]
    return JSONResponse({"samples": [{"index": i, "image": png(imgs[i])} for i in range(len(imgs))]})


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest) -> JSONResponse:
    """Run the Chan-Vese level set live, and return a strip of iterates."""
    if SAMPLES is None:
        return JSONResponse({"error": "no samples bundled"}, status_code=404)
    image = torch.from_numpy(SAMPLES["images"][req.sample : req.sample + 1].astype("float32"))
    masks = evolve(image, iterations=int(req.iterations), mu=float(req.mu), dt=float(req.dt))[0]
    picks = np.linspace(0, masks.shape[0] - 1, int(req.frames)).astype(int)
    return JSONResponse({
        "image": png(SAMPLES["images"][req.sample]),
        "frames": [{"t": int(t) + 1, "mask": png(masks[t].numpy().astype("float32"))} for t in picks],
        "moved": float((masks[-1] != masks[0]).float().mean()),
    })


class PredictRequest(BaseModel):
    sample: int = 0
    t: int = 20
    horizon: int = 10


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> JSONResponse:
    """What each trained model predicts, next to the true Chan-Vese iterate."""
    if SAMPLES is None or not MODELS:
        return JSONResponse({"error": "no models bundled"}, status_code=404)
    window = META.get("window", 4)
    masks = torch.from_numpy(SAMPLES["masks"][req.sample].astype("float32"))
    image = torch.from_numpy(SAMPLES["images"][req.sample].astype("float32"))
    t = max(window, min(int(req.t), masks.shape[0] - int(req.horizon) - 1))

    frames = masks[t - window : t].unsqueeze(0)
    mean, std = META.get("mean", 0.45), META.get("std", 0.25)
    x = torch.stack([((image - mean) / std).expand(1, window, 64, 64), frames], dim=2)
    target = masks[t + req.horizon - 1]
    previous = masks[t - 1]
    change = (target != previous).float()

    out = {"image": png(image.numpy()), "previous": png(previous.numpy()),
           "target": png(target.numpy()), "change": png(change.numpy()),
           "t": t, "horizon": int(req.horizon), "predictions": []}
    with torch.no_grad():
        for name, model in MODELS.items():
            pred = (torch.sigmoid(model(x).float()) > 0.5).float()[0]
            inter = float((pred * target).sum())
            union = float(((pred + target) > 0).float().sum())
            c_inter = float((pred * target * change).sum())
            c_union = float((((pred + target) > 0).float() * change).sum())
            out["predictions"].append({
                "name": name,
                "label": META["models"][name].get("label", name),
                "mask": png(pred.numpy()),
                "iou": round(inter / max(union, 1), 4),
                "change_iou": round(c_inter / max(c_union, 1), 4),
                "note": META["models"][name].get("note", ""),
            })
    return JSONResponse(out)


@app.get("/api/results")
def api_results() -> JSONResponse:
    return JSONResponse(RESULTS)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "models": list(MODELS), "samples": 0 if SAMPLES is None else len(SAMPLES["images"])}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))
