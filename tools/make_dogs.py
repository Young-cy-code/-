#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线条小狗素材生成器
==================

11 级小狗全部用矢量路径画出来（超采样抗锯齿），不依赖任何外部美术素材。
输出约定和原版《合成大奶娃》完全一致，方便直接套用原来的渲染 / 碰撞管线：

  assets/dogs/NN-slug.png   512x512 PNG-32，纯透明底，主体占画布长边的 92%
                            （和 game.js 里的 ASSET_FILL 保持一致）
  _dogs_preview.png         --sheet 时导出的对照表（含"游戏内实际大小"一行）

两个关键做法：

1) 一笔画轮廓：不是"一个零件画一次边"——那样头/身/耳接缝处会糊一堆内轮廓线。
   而是先把所有零件整体铺一层墨（各自向外扩 lw），再整体压一层白填充，
   于是只剩一条干净的外轮廓，正是线条小狗那种一笔画的味道。

2) 描边粗细反推：屏幕上要看到 L 像素的线，取决于"这套造型最后占多少个设计单位"。
   造型宽高不一时 D 会变，所以先按 D=1 画一遍量出真实的 D，再按 D 重画一遍。
   这样 11 级在游戏里看着粗细一致，不会小的糊成一团、大的细得像没画。

坐标系：造型写在「归一化设计空间」[0,1]x[0,1] 里，y 轴向下。
光栅化时用 zoom 把设计空间缩到画布中间（两侧留白），允许造型伸到 [-0.3, 1.3]。
画完按 alpha 包围盒裁紧，等比缩放到 471px (=512*0.92) 居中贴到 512 画布上，
于是「贴图里的主体尺寸」和「物理半径 r」严格对应，不会看着大、撞着小。

跑法：
  python tools/make_dogs.py             # 生成 11 张 PNG
  python tools/make_dogs.py --sheet     # 顺便导出对照表
"""
import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "dogs")

WORK = 3200          # 光栅化画布
ZOOM = 1.60          # 设计空间 [0,1] 占画布的比例（留白给翅膀/光环这种外扩零件）
CANVAS = 512         # 成品画布边长
FILL = 0.92          # 主体占成品画布长边的比例（必须和 game.js 的 ASSET_FILL 一致）
CONTENT = int(round(CANVAS * FILL))   # 471

INK = (58, 52, 62, 255)
BODY = (255, 255, 255, 255)
EAR_IN = (250, 226, 231, 255)


def with_a(c, a):
    if len(c) == 4:
        return (c[0], c[1], c[2], a)
    return tuple(c) + (a,)


# ----------------------------------------------------------------------------
#  基础矢量工具
# ----------------------------------------------------------------------------

def bezier(p0, p1, p2, p3, steps=28):
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        b0, b1, b2, b3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        out.append((p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
                    p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3))
    return out


def path(*cmds):
    """('M', p) / ('L', p) / ('C', c1, c2, p) 拼成折线"""
    pts, cur = [], (0.0, 0.0)
    for c in cmds:
        if c[0] == "M":
            cur = c[1]
            pts.append(cur)
        elif c[0] == "L":
            cur = c[1]
            pts.append(cur)
        else:
            pts.extend(bezier(cur, c[1], c[2], c[3])[1:])
            cur = c[3]
    return pts


def superellipse(cx, cy, rx, ry, n=2.0, steps=260):
    """n=2 是椭圆；n 越大越接近圆角方"""
    pts = []
    e = 2.0 / n
    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        ct, st = math.cos(t), math.sin(t)
        pts.append((cx + rx * math.copysign(abs(ct) ** e, ct),
                    cy + ry * math.copysign(abs(st) ** e, st)))
    return pts


class _Draw:
    """图元统一吃归一化坐标；lw 一律是「看得见的描边粗细」"""

    size = WORK
    zoom = ZOOM

    def _pt(self, p):
        k = self.size / self.zoom
        return ((p[0] - 0.5) * k + self.size * 0.5, (p[1] - 0.5) * k + self.size * 0.5)

    def _pts(self, pts):
        return [self._pt(p) for p in pts]

    def _w(self, w):
        return max(1, int(round(w * self.size / self.zoom)))

    def shape(self, pts, fill=None, ink=None, lw=0.0):
        p = self._pts(pts)
        if fill is not None:
            self.d.polygon(p, fill=fill)
        if ink is not None and lw > 0:
            self.d.line(p + [p[0]], fill=ink, width=self._w(2.0 * lw), joint="curve")
        return self

    def outline(self, pts, color, w):
        p = self._pts(pts)
        self.d.line(p + [p[0]], fill=color, width=self._w(w), joint="curve")
        return self

    def ellipse(self, cx, cy, rx, ry, fill=None, ink=None, lw=0.0, n=2.0):
        return self.shape(superellipse(cx, cy, rx, ry, n), fill, ink, lw)

    def stroke(self, pts, color, w):
        p = self._pts(pts)
        pw = self._w(w)
        self.d.line(p, fill=color, width=pw, joint="curve")
        r = pw / 2.0
        for q in (p[0], p[-1]):
            self.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=color)
        return self

    def capsule(self, a, b, w, fill, ink=None, lw=0.0):
        pa, pb = self._pt(a), self._pt(b)
        if ink is not None and lw > 0:
            wi = self._w(w + 2.0 * lw)
            self.d.line([pa, pb], fill=ink, width=wi)
            r = wi / 2.0
            for q in (pa, pb):
                self.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=ink)
        if fill is not None:
            wf = self._w(w)
            self.d.line([pa, pb], fill=fill, width=wf)
            r = wf / 2.0
            for q in (pa, pb):
                self.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=fill)
        return self

    def ribbon(self, pts, w, fill, ink, lw):
        """带描边的曲线：先铺粗墨线，再压上细填充"""
        if ink is not None and lw > 0:
            self.stroke(pts, ink, w + 2.0 * lw)
        self.stroke(pts, fill, w)
        return self


class _Layer(_Draw):
    def __init__(self, pen):
        self.size = pen.size
        self.zoom = pen.zoom
        self.pen = pen
        self.img = Image.new("RGBA", pen.img.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pen.img = Image.alpha_composite(self.pen.img, self.img)
        self.pen.d = ImageDraw.Draw(self.pen.img)


class Pen(_Draw):
    def __init__(self, size=WORK, zoom=ZOOM):
        self.size = size
        self.zoom = zoom
        self.img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def layer(self):
        return _Layer(self)

    def span(self):
        """造型在「设计单位」里占多大（取长边），用来反推描边粗细"""
        box = self.img.getbbox()
        if not box:
            return 1.0
        return max(box[2] - box[0], box[3] - box[1]) * self.zoom / float(self.size)

    def finish(self):
        box = self.img.getbbox()
        if not box:
            return None
        crop = self.img.crop(box)
        w, h = crop.size
        k = CONTENT / float(max(w, h))
        nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
        crop = crop.resize((nw, nh), Image.LANCZOS)
        out = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
        out.paste(crop, ((CANVAS - nw) // 2, (CANVAS - nh) // 2), crop)
        return out


# ----------------------------------------------------------------------------
#  图元工厂：同一个零件既能「扩 g 铺墨」也能「原样填充」
# ----------------------------------------------------------------------------

def P_ell(cx, cy, rx, ry, n=2.0):
    return lambda p, g, c: p.ellipse(cx, cy, rx + g, ry + g, fill=c, n=n)


def P_cap(a, b, w):
    return lambda p, g, c: p.capsule(a, b, w + 2.0 * g, c, None)


def P_poly(pts):
    return lambda p, g, c: (p.outline(pts, c, 2.0 * g) if g > 0 else p.shape(pts, fill=c))


def P_line(pts, w):
    return lambda p, g, c: p.stroke(pts, c, w + 2.0 * g)


def silhouette(p, prims, lw):
    """先整体铺墨，再整体填白 —— 只留一条干净的外轮廓"""
    for pr in prims:
        pr(p, lw, INK)
    for pr in prims:
        pr(p, 0.0, BODY)


# ----------------------------------------------------------------------------
#  小狗零件
# ----------------------------------------------------------------------------

HEAD_RX, HEAD_RY, HEAD_N = 0.255, 0.236, 2.30


def ear_pts(s, cx, cy, rx, ry, k):
    """下垂耳轮廓（u 向外、v 向下，单位是 rx / ry）；根部会被脑袋压住"""
    ku, kv = k if isinstance(k, tuple) else (k, k)

    def m(u, v):
        return (cx + s * rx * u * ku, cy + ry * v * kv)

    return path(
        ("M", m(0.60, -0.74)),
        ("C", m(0.96, -0.86), m(1.22, -0.78), m(1.36, -0.52)),
        ("C", m(1.50, -0.24), m(1.50, 0.12), m(1.42, 0.42)),
        ("C", m(1.32, 0.78), m(1.12, 1.00), m(0.86, 1.02)),
        ("C", m(0.70, 1.02), m(0.62, 0.62), m(0.60, -0.74)),
    )


def ear_inner_pts(s, cx, cy, rx, ry, k):
    ku, kv = k if isinstance(k, tuple) else (k, k)

    def m(u, v):
        return (cx + s * rx * u * ku, cy + ry * v * kv)

    return path(
        ("M", m(0.96, -0.46)),
        ("C", m(1.20, -0.44), m(1.28, -0.10), m(1.26, 0.18)),
        ("C", m(1.24, 0.50), m(1.10, 0.68), m(0.94, 0.72)),
        ("C", m(0.86, 0.50), m(0.88, -0.10), m(0.96, -0.46)),
    )


def head_prims(cx, cy, k=1.0, ear_k=None):
    rxk, ryk = HEAD_RX * k, HEAD_RY * k
    ek = k if ear_k is None else ear_k
    prims = [P_poly(superellipse(cx, cy, rxk, ryk, HEAD_N))]
    for s in (-1, 1):
        prims.append(P_poly(ear_pts(s, cx, cy, rxk, ryk, ek)))
    return prims


def head_details(p, cx, cy, t, k=1.0, ear_k=None):
    rxk, ryk = HEAD_RX * k, HEAD_RY * k
    ek = k if ear_k is None else ear_k
    for s in (-1, 1):
        p.shape(ear_inner_pts(s, cx, cy, rxk, ryk, ek), fill=EAR_IN)
    face(p, cx, cy, rxk, ryk, t)


def face(p, cx, cy, rx, ry, t):
    """眼睛 / 鼻子 / ω 嘴 / 腮红（t 越小五官越大，小图上才看得见）"""
    lw = line_width(t)
    eye_r = max(0.026, 0.058 - 0.0032 * t)
    ex = rx * 0.42
    ey = cy - ry * 0.13
    for s in (-1, 1):
        p.ellipse(cx + s * ex, ey, eye_r * 0.86, eye_r * 1.14, fill=INK)
        if eye_r > 0.040:
            with p.layer() as L:
                L.ellipse(cx + s * ex - eye_r * 0.30, ey - eye_r * 0.44,
                          eye_r * 0.30, eye_r * 0.30, fill=(255, 255, 255, 235))

    nr = eye_r * 0.85
    ny = cy + ry * 0.26
    p.ellipse(cx, ny, nr, nr * 0.78, fill=INK, n=1.7)

    m0 = ny + nr * 0.72
    for s in (-1, 1):
        pts = path(("M", (cx, m0 - nr * 0.34)),
                   ("C", (cx + s * nr * 0.22, m0 + nr * 0.78),
                    (cx + s * nr * 1.00, m0 + nr * 0.72),
                    (cx + s * nr * 1.38, m0 + nr * 0.04)))
        p.stroke(pts, INK, lw * 0.9)

    if eye_r > 0.034:
        with p.layer() as L:
            for s in (-1, 1):
                L.ellipse(cx + s * rx * 0.72, cy + ry * 0.32, rx * 0.19, rx * 0.11,
                          fill=(255, 168, 184, 120))


def collar(p, cx, cy, w, lw, color):
    pts = path(("M", (cx - w * 0.5, cy)),
               ("C", (cx - w * 0.26, cy + w * 0.20), (cx + w * 0.26, cy + w * 0.20),
                (cx + w * 0.5, cy)))
    p.ribbon(pts, w * 0.150, color, INK, lw * 0.65)
    r = w * 0.070
    p.ellipse(cx, cy + w * 0.175, r, r, fill=color, ink=INK, lw=lw * 0.85)


def bone(p, cx, cy, w, h, lw, color):
    bar = ((cx - w * 0.5, cy), (cx + w * 0.5, cy))
    ends = [(cx + s * w * 0.5, cy + d * h * 0.30) for s in (-1, 1) for d in (-1, 1)]
    for e in ends:                       # 先整体铺墨，接缝里就不会有内轮廓
        p.ellipse(e[0], e[1], h * 0.46 + lw, h * 0.46 + lw, fill=INK)
    p.capsule(bar[0], bar[1], h * 0.54 + 2.0 * lw, INK)
    for e in ends:
        p.ellipse(e[0], e[1], h * 0.46, h * 0.46, fill=color)
    p.capsule(bar[0], bar[1], h * 0.54, color)


def bow(p, cx, cy, s, lw, color):
    for sg in (-1, 1):
        pts = path(("M", (cx + sg * s * 0.10, cy)),
                   ("C", (cx + sg * s * 0.45, cy - s * 0.72),
                    (cx + sg * s * 1.22, cy - s * 0.64),
                    (cx + sg * s * 1.12, cy + s * 0.02)),
                   ("C", (cx + sg * s * 1.04, cy + s * 0.64),
                    (cx + sg * s * 0.45, cy + s * 0.62),
                    (cx + sg * s * 0.10, cy)))
        p.shape(pts, fill=color, ink=INK, lw=lw)
    p.ellipse(cx, cy, s * 0.17, s * 0.17, fill=color, ink=INK, lw=lw)


def cap(p, cx, cy, rx, ry, lw, color):
    dome = [pt for pt in superellipse(cx, cy, rx, ry, 2.0, 200) if pt[1] <= cy + 1e-6]
    dome.append((cx - rx, cy))
    p.shape(dome, fill=color, ink=INK, lw=lw)
    brim = superellipse(cx + rx * 0.34, cy + ry * 0.10, rx * 1.24, ry * 0.30, 2.3)
    p.shape(brim, fill=color, ink=INK, lw=lw)
    p.ellipse(cx, cy - ry * 0.88, rx * 0.15, rx * 0.15, fill=color, ink=INK, lw=lw * 0.9)


def cape(p, cx, cy, w, h, lw, color):
    """斗篷：画在狗之前，只露出身体两侧和下面"""
    pts = path(("M", (cx - w * 0.44, cy)),
               ("C", (cx - w * 0.78, cy + h * 0.34), (cx - w * 0.76, cy + h * 0.80),
                (cx - w * 0.54, cy + h)),
               ("C", (cx - w * 0.16, cy + h * 0.86), (cx + w * 0.16, cy + h * 0.92),
                (cx + w * 0.46, cy + h)),
               ("C", (cx + w * 0.74, cy + h * 0.76), (cx + w * 0.80, cy + h * 0.32),
                (cx + w * 0.44, cy)))
    p.shape(pts, fill=color, ink=INK, lw=lw)


def wing(p, root, cy, side, size, lw, color, alpha=255):
    """翅膀：三片羽毛从 root 朝外上方张开（side: -1 左 / +1 右）"""
    for ang in (-1.30, -0.35, 0.60):
        tip = (root + side * size * 1.02 * math.cos(ang * 0.62),
               cy + size * ang * 0.92)
        pts = path(("M", (root, cy)),
                   ("C", (root + side * size * 0.42, cy + size * ang * 0.62 - size * 0.24),
                    (tip[0] - side * size * 0.26, tip[1] - size * 0.20), tip),
                   ("C", (tip[0] - side * size * 0.24, tip[1] + size * 0.20),
                    (root + side * size * 0.40, cy + size * ang * 0.60 + size * 0.24),
                    (root, cy)))
        with p.layer() as L:
            L.shape(pts, fill=with_a(color, alpha), ink=with_a(INK, alpha), lw=lw)


def halo(p, cx, cy, rx, ry, lw):
    p.shape(superellipse(cx, cy, rx, ry, 2.0, 200), fill=(255, 214, 92, 255), ink=INK, lw=lw)
    p.shape(superellipse(cx, cy, rx * 0.58, ry * 0.58, 2.0, 120), fill=(0, 0, 0, 0))


def crown(p, cx, cy, w, h, lw, color=(255, 205, 70, 255)):
    pts = path(("M", (cx - w * 0.5, cy)), ("L", (cx - w * 0.5, cy - h)),
               ("L", (cx - w * 0.24, cy - h * 0.42)), ("L", (cx, cy - h * 1.14)),
               ("L", (cx + w * 0.24, cy - h * 0.42)), ("L", (cx + w * 0.5, cy - h)),
               ("L", (cx + w * 0.5, cy)), ("L", (cx - w * 0.5, cy)))
    p.shape(pts, fill=color, ink=INK, lw=lw)
    for s in (-1, 0, 1):
        p.ellipse(cx + s * w * 0.30, cy - h * (1.06 if s == 0 else 0.96),
                  w * 0.058, w * 0.058, fill=(255, 255, 255, 255), ink=INK, lw=lw * 0.7)


# ----------------------------------------------------------------------------
#  四种姿势：先拼剪影，再补五官
# ----------------------------------------------------------------------------

def pose_head(p, lw, t, ear_k):
    prims = head_prims(0.500, 0.500, 1.0, ear_k)
    silhouette(p, prims, lw)
    head_details(p, 0.500, 0.500, t, 1.0, ear_k)


def pose_lying(p, lw, t, ear_k):
    prims = [
        P_ell(0.500, 0.762, 0.292, 0.150, 2.45),            # 扁身体
        P_ell(0.318, 0.852, 0.118, 0.070),                  # 前爪 x2
        P_ell(0.682, 0.852, 0.118, 0.070),
    ]
    prims += head_prims(0.500, 0.496, 1.02, ear_k)
    silhouette(p, prims, lw)
    head_details(p, 0.500, 0.496, t, 1.02, ear_k)


def pose_sitting(p, lw, t, ear_k):
    prims = [
        P_ell(0.290, 0.790, 0.140, 0.122),                  # 后臀 x2
        P_ell(0.710, 0.790, 0.140, 0.122),
        P_ell(0.500, 0.752, 0.232, 0.188, 2.40),            # 身体
        P_cap((0.415, 0.772), (0.405, 0.905), 0.092),       # 前腿 x2
        P_cap((0.585, 0.772), (0.595, 0.905), 0.092),
    ]
    prims += head_prims(0.500, 0.402, 1.0, ear_k)
    silhouette(p, prims, lw)
    head_details(p, 0.500, 0.402, t, 1.0, ear_k)


def pose_stand(p, lw, t, ear_k):
    tail = path(("M", (0.700, 0.795)),
                ("C", (0.790, 0.812), (0.878, 0.762), (0.870, 0.666)))
    prims = [
        P_line(tail, 0.074),                                # 尾巴
        P_cap((0.400, 0.800), (0.392, 0.912), 0.098),       # 后腿 x2
        P_cap((0.600, 0.800), (0.608, 0.912), 0.098),
        P_ell(0.500, 0.672, 0.220, 0.184, 2.35),            # 身体
    ]
    prims += head_prims(0.500, 0.366, 1.0, ear_k)
    silhouette(p, prims, lw)
    head_details(p, 0.500, 0.366, t, 1.0, ear_k)


POSES = {"head": pose_head, "lying": pose_lying, "sit": pose_sitting, "stand": pose_stand}

COLORS = {
    "collar": (134, 214, 200, 255),
    "collar2": (255, 176, 190, 255),
    "bone": (244, 222, 176, 255),
    "bow": (255, 106, 128, 255),
    "cap": (90, 169, 230, 255),
    "cape": (162, 130, 232, 255),
    "crown": (255, 205, 70, 255),
    "wing": (245, 250, 255),
}

TIERS = [
    dict(t=0,  r=17,  slug="01-pup",    name="小奶狗",   pose="head",  ears=(0.72, 0.90), extras=[]),
    dict(t=1,  r=23,  slug="02-floppy", name="垂耳狗",   pose="head",  ears=(0.98, 1.16), extras=[]),
    dict(t=2,  r=31,  slug="03-sploot", name="趴趴狗",   pose="lying", ears=(0.90, 1.06), extras=[]),
    dict(t=3,  r=39,  slug="04-sit",    name="坐坐狗",   pose="sit",   ears=(0.88, 1.04), extras=["collar"]),
    dict(t=4,  r=48,  slug="05-stand",  name="站站狗",   pose="stand", ears=(0.86, 1.00), extras=["collar2"]),
    dict(t=5,  r=58,  slug="06-bone",   name="骨头狗",   pose="stand", ears=(0.88, 1.02), extras=["bone"]),
    dict(t=6,  r=69,  slug="07-bow",    name="蝴蝶结狗", pose="sit",   ears=(0.92, 1.08), extras=["bow"]),
    dict(t=7,  r=81,  slug="08-cap",    name="鸭舌帽狗", pose="stand", ears=(0.88, 1.02), extras=["cap"]),
    dict(t=8,  r=94,  slug="09-cape",   name="披风狗",   pose="stand", ears=(0.90, 1.06), extras=["cape", "collar"]),
    dict(t=9,  r=108, slug="10-halo",   name="半犬神",   pose="stand", ears=(0.94, 1.10), extras=["wing", "halo"]),
    dict(t=10, r=124, slug="11-god",    name="犬神",     pose="stand", ears=(0.98, 1.14), extras=["wing_big", "halo", "crown"]),
]


def line_width(t):
    """按「主体占 1 个设计单位」给的标称值；真实值由 draw_dog 量出 D 后修正"""
    r = TIERS[t]["r"]
    return (1.10 + 0.008 * r) / (2.0 * r)


def render(spec, lw):
    t = spec["t"]
    p = Pen()
    ex = spec["extras"]
    if "cape" in ex:
        cape(p, 0.500, 0.606, 0.600, 0.318, lw * 0.95, COLORS["cape"])
    if "wing" in ex or "wing_big" in ex:
        big = "wing_big" in ex
        for sg in (-1, 1):
            wing(p, 0.500 + sg * 0.155, 0.640,
                 sg, 0.280 if big else 0.215, lw * 0.9, COLORS["wing"],
                 255 if big else 225)
    POSES[spec["pose"]](p, lw, t, spec["ears"])
    if "collar" in ex or "collar2" in ex:
        collar(p, 0.500, 0.576, 0.340, lw, COLORS["collar2"] if "collar2" in ex else COLORS["collar"])
    if "bone" in ex:
        bone(p, 0.500, 0.612, 0.360, 0.150, lw * 0.9, COLORS["bone"])
    if "bow" in ex:
        bow(p, 0.318, 0.212, 0.118, lw, COLORS["bow"])
    if "cap" in ex:
        cap(p, 0.500, 0.172, 0.222, 0.162, lw, COLORS["cap"])
    if "halo" in ex:
        halo(p, 0.500, 0.036 if "crown" in ex else 0.072, 0.138, 0.048, lw * 0.8)
    if "crown" in ex:
        crown(p, 0.500, 0.196, 0.235, 0.098, lw * 0.9, COLORS["crown"])
    return p


def draw_dog(spec):
    lw0 = line_width(spec["t"])
    d = render(spec, lw0).span()          # 第一遍：量出这套造型真实占多少个设计单位
    return render(spec, lw0 * d).finish()  # 第二遍：按 D 修正描边后再画


# ----------------------------------------------------------------------------
#  输出
# ----------------------------------------------------------------------------

def make_sheet(imgs, path_out):
    pad, cell = 16, 200
    W = pad + len(imgs) * (cell + pad)
    H = pad * 3 + cell * 2 + 46
    sheet = Image.new("RGB", (W, H), (214, 236, 226))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    d.text((pad, 10), "上排 = 放大后的造型；下排 = 游戏里看到的实际大小", fill=(40, 60, 55), font=font)
    for i, (im, spec) in enumerate(zip(imgs, TIERS)):
        x = pad + i * (cell + pad)
        tile = Image.new("RGBA", (cell, cell), (214, 236, 226, 255))
        big = im.resize((cell - 24, cell - 24), Image.LANCZOS)
        tile.paste(big, (12, 12), big)
        sheet.paste(tile.convert("RGB"), (x, 40))
        d.text((x + 4, 40 + cell + 2), "%s r=%d" % (spec["slug"], spec["r"]),
               fill=(40, 60, 55), font=font)
        size = max(8, int(round(spec["r"] * 2 * 0.62)))
        small = im.resize((size, size), Image.LANCZOS)
        bx = x + (cell - size) // 2
        by = 40 + cell + 30
        d.rectangle([bx - 3, by - 3, bx + size + 2, by + size + 2], outline=(150, 180, 170))
        sheet.paste(small, (bx, by), small)
    sheet.save(path_out)
    print("已导出对照表 %s" % path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="顺便导出 _dogs_preview.png")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = []
    print("%-4s %-12s %-10s %-8s %s" % ("tier", "slug", "name", "pose", "尺寸"))
    print("-" * 58)
    for spec in TIERS:
        im = draw_dog(spec)
        dst = os.path.join(OUT_DIR, spec["slug"] + ".png")
        im.save(dst)
        imgs.append(im)
        print("%-4d %-12s %-10s %-8s %s  %.1f KB" % (
            spec["t"], spec["slug"], spec["name"], spec["pose"],
            "x".join(str(v) for v in im.size), os.path.getsize(dst) / 1024.0))
    print("-" * 58)
    print("共 %d 张 -> %s" % (len(TIERS), OUT_DIR))
    if args.sheet:
        make_sheet(imgs, os.path.join(ROOT, "_dogs_preview.png"))


if __name__ == "__main__":
    main()
