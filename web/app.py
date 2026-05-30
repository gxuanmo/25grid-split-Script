#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Splitter v4 — Web UI

本地 FastAPI 服务：上传图片 → 自动检测网格 → 网页预览 → 切割 → 单张/打包下载。
直接复用项目根的 grid_splitter_v4.py，不做算法侵入式修改。
"""

import io
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 把项目根加到 sys.path，复用现有 grid_splitter_v4 算法
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_splitter_v4 import (  # noqa: E402  (intentional: must follow sys.path tweak)
    _imread_unicode,
    auto_detect_grid_dimensions,
    detect_outer_borders,
    find_grid_lines,
    split_grid_clean,
)

# ───────── 路径与常量 ─────────
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_ROOT = WEB_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB，覆盖项目里现有的 63MB 大图

app = FastAPI(title="Grid Splitter v4")


# ───────── Pydantic schemas ─────────
class Borders(BaseModel):
    top: int
    bot: int
    left: int
    right: int


class UploadResp(BaseModel):
    file_id: str
    original_name: str
    img_w: int
    img_h: int
    borders: Borders
    cropped_w: int
    cropped_h: int
    rows: int
    cols: int
    # 坐标已加回 borders 偏移，相对原图
    h_lines: list[int]
    v_lines: list[int]
    auto_detected: bool
    preview_url: str


class PreviewReq(BaseModel):
    file_id: str
    rows: int = Field(..., ge=1, le=20)
    cols: int = Field(..., ge=1, le=20)


class PreviewResp(BaseModel):
    h_lines: list[int]
    v_lines: list[int]
    borders: Borders


class SplitResp(BaseModel):
    count: int
    cell_w: int
    cell_h: int
    clips: list[dict]
    zip_url: str


# ───────── 工具函数 ─────────
def _session_dir(file_id: str) -> Path:
    # 防目录穿越
    if not file_id.isalnum() or len(file_id) > 64:
        raise HTTPException(400, "bad file_id")
    p = UPLOAD_ROOT / file_id
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, "session not found")
    return p


def _find_original(sess: Path) -> Path:
    for ext in ALLOWED_SUFFIXES:
        f = sess / f"original{ext}"
        if f.exists():
            return f
    raise HTTPException(404, "original image missing")


def _compute_lines(img, rows: int, cols: int):
    """检测外边框 → 在 crop 后图上找网格线 → 把坐标加回偏移，返回相对原图坐标。"""
    h, w = img.shape[:2]
    top, bot, left, right = detect_outer_borders(img)
    cropped = img[top:h - bot if bot else h, left:w - right if right else w]
    ch, cw = cropped.shape[:2]
    h_lines_local, v_lines_local = find_grid_lines(cropped, rows, cols)
    return {
        "borders": Borders(top=top, bot=bot, left=left, right=right),
        "cropped_w": cw,
        "cropped_h": ch,
        "h_lines": [y + top for y in h_lines_local],
        "v_lines": [x + left for x in v_lines_local],
    }


# ───────── 路由 ─────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/static/index.html")


@app.post("/api/upload", response_model=UploadResp)
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "missing filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported format: {suffix}")

    file_id = uuid.uuid4().hex[:12]
    sess = UPLOAD_ROOT / file_id
    sess.mkdir(parents=True, exist_ok=True)

    # 流式落盘 + 大小校验
    orig_path = sess / f"original{suffix}"
    size = 0
    oversize = False
    with open(orig_path, "wb") as f:
        while chunk := await file.read(1 << 20):  # 1MB chunks
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                oversize = True
                break
            f.write(chunk)
    # 退出 with 后 fd 已关闭，Windows 上 rmtree 才不会被锁
    if oversize:
        shutil.rmtree(sess, ignore_errors=True)
        raise HTTPException(413, f"file too large (>{MAX_UPLOAD_BYTES} bytes)")

    img = _imread_unicode(str(orig_path))
    if img is None:
        shutil.rmtree(sess, ignore_errors=True)
        raise HTTPException(400, "cannot decode image")
    h, w = img.shape[:2]

    # 检测外边框
    top, bot, left, right = detect_outer_borders(img)
    cropped = img[top:h - bot if bot else h, left:w - right if right else w]

    # 自动检测网格维度
    detected = auto_detect_grid_dimensions(cropped)
    if detected:
        rows, cols = detected
        auto = True
    else:
        rows, cols = 3, 3
        auto = False

    lines_local = find_grid_lines(cropped, rows, cols)
    h_lines = [y + top for y in lines_local[0]]
    v_lines = [x + left for x in lines_local[1]]

    return UploadResp(
        file_id=file_id,
        original_name=file.filename,
        img_w=w,
        img_h=h,
        borders=Borders(top=top, bot=bot, left=left, right=right),
        cropped_w=cropped.shape[1],
        cropped_h=cropped.shape[0],
        rows=rows,
        cols=cols,
        h_lines=h_lines,
        v_lines=v_lines,
        auto_detected=auto,
        preview_url=f"/files/{file_id}/{orig_path.name}",
    )


@app.post("/api/preview", response_model=PreviewResp)
def preview(req: PreviewReq):
    sess = _session_dir(req.file_id)
    img = _imread_unicode(str(_find_original(sess)))
    if img is None:
        raise HTTPException(500, "cannot reload original")
    info = _compute_lines(img, req.rows, req.cols)
    return PreviewResp(
        h_lines=info["h_lines"],
        v_lines=info["v_lines"],
        borders=info["borders"],
    )


@app.post("/api/split", response_model=SplitResp)
def split(req: PreviewReq):
    sess = _session_dir(req.file_id)
    orig = _find_original(sess)
    clips_dir = sess / "clips"
    # split_grid_clean 自己会清掉旧 clip_*.jpg，但保险起见目录先建
    clips_dir.mkdir(parents=True, exist_ok=True)

    n = split_grid_clean(
        image_path=str(orig),
        output_folder=str(clips_dir),
        rows=req.rows,
        cols=req.cols,
        auto_detect=True,
    )
    if n <= 0:
        raise HTTPException(500, "split produced 0 clips")

    clips = sorted(clips_dir.glob("clip_*.jpg"))
    # 时间戳防浏览器缓存（同一 file_id 多次切割时 clip 名相同）
    bust = int(time.time() * 1000)

    # 回填 cell 尺寸（仅读第一张即可，所有 cell 统一过 Lanczos）
    first = cv2.imdecode(np.fromfile(str(clips[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    cell_h, cell_w = first.shape[:2]

    return SplitResp(
        count=n,
        cell_w=cell_w,
        cell_h=cell_h,
        clips=[
            {
                "idx": i + 1,
                "name": c.name,
                "url": f"/files/{req.file_id}/clips/{c.name}?v={bust}",
            }
            for i, c in enumerate(clips)
        ],
        zip_url=f"/api/zip/{req.file_id}?v={bust}",
    )


@app.get("/api/zip/{file_id}")
def download_zip(file_id: str, v: Optional[str] = None):  # noqa: ARG001 — v 仅用于 URL 防缓存
    sess = _session_dir(file_id)
    clips_dir = sess / "clips"
    if not clips_dir.exists():
        raise HTTPException(404, "no clips yet")
    clips = sorted(clips_dir.glob("clip_*.jpg"))
    if not clips:
        raise HTTPException(404, "no clips yet")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for clip in clips:
            zf.write(clip, clip.name)
    buf.seek(0)

    fname = f"clips_{file_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.delete("/api/session/{file_id}")
def cleanup(file_id: str):
    sess = _session_dir(file_id)
    shutil.rmtree(sess, ignore_errors=True)
    return {"ok": True}


# 静态资源：前端单页 + 上传文件直读（原图预览 / 切片缩略图）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/files", StaticFiles(directory=str(UPLOAD_ROOT)), name="files")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8765, reload=False)
