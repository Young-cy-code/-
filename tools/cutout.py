#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把用户提供的狗狗图片抠图（去掉背景）并去掉水印，产出游戏直接可用的透明 PNG。

为什么自己写：
  素材背景很杂（纯白 / 黄色渐变 / 红色实底），水印还跟背景色差很大。做法分两步：
    1) 用很严的容差 + 区域生长，先框出「肯定是背景」的一块；
    2) 用这块背景拟合出一张平滑的「背景色地图」（往图里外推），
       再按「离背景色地图多远」来判定前景。
  这样渐变背景不会剩下脏块，而狗身上浅色的部分也不会被误吞（第一步的严容差保证了这点）。
  最后只保留最大的前景块 + 离它够近的碎块；离得远的小块（水印）直接扔掉。

用法：
  python tools/cutout.py                  # tools/src/*-raw.*  →  tools/src/cut/*.png
  python tools/cutout.py --preview        # 顺便导出 _cut_preview.png 对照表
  python tools/cutout.py --only 01,03 --debug
"""
import argparse
import os
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "tools", "src")
OUT_DIR = os.path.join(ROOT, "assets", "dogs")   # 直接写进游戏素材目录

CANVAS = 512          # 输出画布边长（和 game.js 的贴图约定一致）
FILL = 0.92           # 主体占画布长边的比例（= game.js 的 ASSET_FILL）
WORK = 480            # 算遮罩时的工作分辨率
SEED_THR = 12         # 第一步：认定「肯定是背景」的严容差
MODEL_THR = 20        # 第二步：跟背景色地图比，差多少算前景
GAP = 0.025           # 离狗多近的碎块算「同一只狗」（相对长边）
CLOSE = 0.012         # 形态学闭运算的半径（相对长边），把断开的线稿接起来
FILL_R = 0.034        # 填肚子用的闭运算半径：先让轮廓闭合，再灌成实心
DESPILL_THR = 44      # 判定「这是漏进来的背景色」的容差

# 源文件顺序 = 用户给的「从小到大」顺序；这里同时定了游戏里的等级顺序
SLUGS = ["01-pup", "02-drop", "03-roll", "04-sit", "05-stand",
         "06-long", "07-bib", "08-puff", "09-big", "10-god"]

# 某张图里如果还有区域生长吃不掉的水印，在这里手工补一刀（原图像素坐标）：
MANUAL_ERASE = {}


# ---------------------------------------------------------------- 基础工具
def load_work(path, work=WORK):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, work / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
    return im


def _box1(a, r, axis):
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    pad = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], 0)
    c = np.cumsum(pad, axis=0)
    z = np.zeros((1,) + a.shape[1:], c.dtype)
    c = np.concatenate([z, c], 0)
    out = (c[2 * r + 1:] - c[:-(2 * r + 1)]) / (2 * r + 1)
    return np.moveaxis(out, 0, axis)


def box_blur(a, r):
    """三次盒糊 ≈ 高斯；不依赖任何第三方图像库。"""
    if r < 1:
        return a.astype(np.float32)
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box1(out, r, 0)
        out = _box1(out, r, 1)
    return out


def grow_background(rgb, thr):
    """从四条边往里长：每个像素只跟「已经是背景的邻居」比，渐变背景也能顺着吃进去。"""
    h, w, _ = rgb.shape
    a = rgb.astype(np.int16)
    vis = np.zeros((h, w), bool)
    dq = deque()

    def seed(y, x):
        if not vis[y, x]:
            vis[y, x] = True
            dq.append((y, x))

    for x in range(w):
        seed(0, x)
        seed(h - 1, x)
    for y in range(h):
        seed(y, 0)
        seed(y, w - 1)

    while dq:
        y, x = dq.popleft()
        c = a[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not vis[ny, nx]:
                if int(np.abs(a[ny, nx] - c).max()) <= thr:
                    vis[ny, nx] = True
                    dq.append((ny, nx))
    return vis


def smooth_model(rgb, seed):
    """用确定是背景的像素，拟合一张平滑的背景色地图（会外推进狗的身体里）。"""
    h, w = seed.shape
    r = max(3, int(round(0.05 * max(h, w))))
    m = seed.astype(np.float32)
    num = np.empty((h, w, 3), np.float32)
    for c in range(3):
        num[..., c] = box_blur(rgb[..., c].astype(np.float32) * m, r)
    den = box_blur(m, r)
    model = np.empty((h, w, 3), np.float32)
    safe = np.maximum(den, 1e-3)
    for c in range(3):
        model[..., c] = num[..., c] / safe
    glob = rgb[seed].mean(axis=0) if seed.any() else np.array([255.0, 255.0, 255.0])
    weak = den < 0.04
    model[weak] = glob
    return model


def components(mask):
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    sizes = [0]
    ys, xs = np.nonzero(mask)
    n = 0
    for k in range(len(ys)):
        y, x = int(ys[k]), int(xs[k])
        if lab[y, x]:
            continue
        n += 1
        lab[y, x] = n
        dq = deque([(y, x)])
        size = 0
        while dq:
            cy, cx = dq.popleft()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    dq.append((ny, nx))
        sizes.append(size)
    return lab, sizes


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fill_holes(comp):
    x0, y0, x1, y1 = bbox_of(comp)
    sub = comp[y0:y1, x0:x1]
    h, w = sub.shape
    outside = np.zeros((h, w), bool)
    dq = deque()

    def seed(y, x):
        if not sub[y, x] and not outside[y, x]:
            outside[y, x] = True
            dq.append((y, x))

    for x in range(w):
        seed(0, x)
        seed(h - 1, x)
    for y in range(h):
        seed(y, 0)
        seed(y, w - 1)
    while dq:
        cy, cx = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not sub[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                dq.append((ny, nx))
    out = np.zeros_like(comp)
    out[y0:y1, x0:x1] = sub | ~outside
    return out


def pick_subject(lab, sizes):
    """留最大的一块，并把离它够近的碎块并进来；离得远的（水印）丢掉。"""
    main = 1 + int(np.argmax(sizes[1:]))
    keep = [main]
    for _ in range(len(sizes)):
        kx0, ky0, kx1, ky1 = bbox_of(np.isin(lab, keep))
        span = max(kx1 - kx0, ky1 - ky0)
        gap = max(4, int(round(GAP * span)))
        grew = False
        for i in range(1, len(sizes)):
            if i in keep or sizes[i] < 8:
                continue
            x0, y0, x1, y1 = bbox_of(lab == i)
            if x0 - gap < kx1 and x1 + gap > kx0 and y0 - gap < ky1 and y1 + gap > ky0:
                keep.append(i)
                grew = True
        if not grew:
            break
    dropped = [(sizes[i], i) for i in range(1, len(sizes)) if i not in keep and sizes[i] >= 8]
    return np.isin(lab, keep), sorted(dropped, reverse=True)


def close_mask(mask, r):
    """闭运算：先把附近的碎块粘起来再腐蚀回去，断开的细线稿就能接上。"""
    if r < 1:
        return mask
    size = 2 * r + 1
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(im) > 128


def subject_mask(rgb):
    seed = grow_background(rgb, SEED_THR)
    model = smooth_model(rgb, seed)
    d = np.abs(rgb.astype(np.float32) - model).max(axis=2)
    lab_c, _ = components((d <= MODEL_THR) | seed)
    ys, xs = np.nonzero(seed)
    if len(ys) == 0:
        raise RuntimeError("找不到背景")
    border = sorted(set(int(v) for v in np.unique(lab_c[ys, xs]) if v))
    bg = np.isin(lab_c, border) if border else seed
    fg = ~bg
    if not fg.any():
        raise RuntimeError("找不到前景")
    # 线稿容易断成好几段：先闭运算粘起来，再挑最大的一块（水印离得远，粘不上）
    span = max(rgb.shape[:2])
    r = max(1, int(round(CLOSE * span)))
    lab, sizes = components(close_mask(fg, r))
    subj, dropped = pick_subject(lab, sizes)
    # 线稿的肚子是「线之间漏出来的背景色」，会被判成背景：
    # 所以再把轮廓闭合一次、把里面灌成实心，狗才不是个空心线框。
    r2 = max(r, int(round(FILL_R * span)), min(16, int(round(0.06 * span))))
    solid = fill_holes(close_mask(subj, r2))
    return solid, dropped, model


def despill(rgb, alpha, model, thr=DESPILL_THR, fill=(255, 255, 255)):
    """轮廓内部还留着「局部的背景色」（线稿是镂空的）→ 刷成白色，跟其它贴图统一。

    判断依据是逐像素的背景色模型（不是全局调色板），所以狗身上本来就是这个
    颜色的地方才会被刷白，狗本身有颜色的地方一律不动。"""
    d = np.abs(rgb.astype(np.float32) - model).max(axis=2)
    hit = (alpha > 128) & (d <= thr)
    out = rgb.copy()
    out[hit] = fill
    return out, int(hit.sum())


# ---------------------------------------------------------------- 主流程
def build(path, slug):
    work = load_work(path)
    rgb_w = np.asarray(work)
    fg_main, dropped, model_w = subject_mask(rgb_w)
    seed = grow_background(rgb_w, SEED_THR)
    fg_main = fill_holes(fg_main)
    dropped = [(s, float(s) / fg_main.size) for s, _ in dropped]

    full = Image.open(path).convert("RGB")
    W, H = full.size

    if slug in MANUAL_ERASE:
        arr = np.asarray(full).copy()
        for (x0, y0, x1, y1) in MANUAL_ERASE[slug]:
            pad = 6
            ax0, ay0 = max(0, x0 - pad), max(0, y0 - pad)
            ax1, ay1 = min(W, x1 + pad), min(H, y1 + pad)
            ring = np.concatenate([
                arr[max(0, ay0 - 3):ay0, ax0:ax1].reshape(-1, 3),
                arr[ay1:ay1 + 3, ax0:ax1].reshape(-1, 3),
                arr[ay0:ay1, max(0, ax0 - 3):ax0].reshape(-1, 3),
                arr[ay0:ay1, ax1:ax1 + 3].reshape(-1, 3),
            ]) if ax1 > ax0 and ay1 > ay0 else np.zeros((0, 3))
            if len(ring):
                arr[ay0:ay1, ax0:ax1] = np.median(ring, axis=0)
        full = Image.fromarray(arr)

    m = Image.fromarray((fg_main * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    alpha = np.asarray(m).astype(np.float32)
    alpha = np.clip((alpha - 60) * (255.0 / 130.0), 0, 255).astype(np.uint8)

    rgb = np.asarray(full).copy()
    model = np.empty((H, W, 3), np.float32)
    for c in range(3):
        model[..., c] = np.asarray(
            Image.fromarray(model_w[..., c].astype(np.float32)).resize((W, H), Image.LANCZOS),
            dtype=np.float32)
    rgb2, nspill = despill(rgb, alpha, model)

    ys, xs = np.nonzero(alpha > 128)
    if len(ys) == 0:
        raise RuntimeError("alpha 全空：%s" % path)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    sub = np.dstack([rgb2, alpha])[y0:y1, x0:x1]
    im = Image.fromarray(sub, "RGBA")
    target = int(round(CANVAS * FILL))
    h, w = y1 - y0, x1 - x0
    s = target / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(im, ((CANVAS - nw) // 2, (CANVAS - nh) // 2), im)

    return {
        "slug": slug, "raw": os.path.basename(path), "src": (W, H), "srcbox": (x0, y0, x1, y1),
        "fg": round(100.0 * float(fg_main.mean()), 1),
        "fill": round(100.0 * float(fg_main.sum()) / max(1, (x1 - x0) * (y1 - y0)), 1),
        "dropped": dropped[:4], "spill": nspill, "canvas": canvas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    raws = sorted(f for f in os.listdir(SRC_DIR) if "-raw." in f)
    print("%-4s %-12s %-12s %-24s %7s %6s %6s  %s" %
          ("slug", "文件", "源尺寸", "主体在源图里的框", "主体占比", "框内填充", "刷白", "丢掉的小块"))
    print("-" * 112)
    outs = []
    for i, fn in enumerate(raws):
        slug = SLUGS[i] if i < len(SLUGS) else "%02d" % (i + 1)
        if args.only and slug not in args.only.split(","):
            continue
        info = build(os.path.join(SRC_DIR, fn), slug)
        dst = os.path.join(OUT_DIR, slug + ".png")
        info["canvas"].save(dst)
        drop = ", ".join("%.2f%%" % (p * 100) for _, p in info["dropped"]) or "-"
        print("%-4s %-12s %-12s %-24s %6.1f%% %6.1f%% %6d  %s" %
              (slug, fn, "%dx%d" % info["src"], str(info["srcbox"]),
               info["fg"], info["fill"], info["spill"], drop))
        outs.append((info, dst))

    if args.preview and outs:
        from PIL import ImageDraw, ImageFont
        n = len(outs)
        cell, pad = 190, 8
        sheet = Image.new("RGB", (pad + n * (cell + pad), 24 + (cell + 26) * 2), (240, 246, 243))
        d = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(size=15)
        except TypeError:
            font = ImageFont.load_default()
        d.text((pad, 6), "上排 = 原图（缩放）    下排 = 抠图结果（棋盘格 = 透明区域）", fill=(40, 60, 55), font=font)
        for i, (info, dst) in enumerate(outs):
            x = pad + i * (cell + pad)
            y = 24
            raw = Image.open(os.path.join(SRC_DIR, info["raw"]))
            raw = raw.convert("RGB")
            s2 = (cell - 12) / max(raw.size)
            raw = raw.resize((max(1, int(raw.width * s2)), max(1, int(raw.height * s2))), Image.LANCZOS)
            sheet.paste(raw, (x + (cell - raw.width) // 2, y + (cell - raw.height) // 2))
            d.text((x + 4, y + cell + 2), "src " + info["slug"], fill=(40, 60, 55), font=font)
            y2 = y + cell + 26
            tile = Image.new("RGB", (cell, cell), (255, 255, 255))
            for yy in range(0, cell, 16):
                for xx in range(0, cell, 16):
                    if (xx // 16 + yy // 16) % 2:
                        tile.paste((226, 232, 229), (xx, yy, min(cell, xx + 16), min(cell, yy + 16)))
            cut = Image.open(dst).resize((cell - 12, cell - 12), Image.LANCZOS)
            tile.paste(cut, (6, 6), cut)
            sheet.paste(tile, (x, y2))
            d.text((x + 4, y2 + cell + 2), "cut " + info["slug"], fill=(40, 60, 55), font=font)
        out = os.path.join(ROOT, "preview-cut.png")
        sheet.save(out)
        print("已导出对照表 %s" % out)


if __name__ == "__main__":
    main()
