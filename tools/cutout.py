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
SEED_THR = 5          # 第一步：认定「肯定是背景」的严容差（必须小，见 FLAT_STD 那段注释）
MODEL_THR = 20        # 第二步：跟背景色地图比，差多少算前景
GAP = 0.020           # 离狗多近的碎块算「同一只狗」（相对长边）
EDGE_GAP = 0.045      # 判定「贴边残片」离主体多远算远（相对长边）
CLOSE = 0.012         # 形态学闭运算的半径（相对长边），把断开的线稿接起来
FILL_R = 0.034        # 填肚子用的闭运算半径「上限」（真正用多大由 fill_radius 按图定）
FILL_JUMP = 0.04      # 判定「这一圈闭运算真的把狗肚子封住了」的增量阈值（相对主体面积）
EDGE_MARGIN = 3       # 离画面边缘这么多像素以内就算「贴边」，专治截图带进来的边角废块
MIN_FRAG = 32         # 剥毛边时被剪断的碎渣：小于这个像素数直接丢掉
DESPILL_THR = 20      # 判定「这是漏进来的背景色」的容差

# 背景是「一整块纯色」的图（四条边几乎 100% 同一个颜色）走「零容差 + 剥毛边」的分割：
#   先从四条边往里灌，只把「和背景色一模一样」的像素连成背景（这一步绝不啃到狗），
#   再一圈一圈往外剥最多 BG_LAYERS 层「跟背景色差一点点」的像素，吃掉 JPEG 噪点毛边。
# 为什么第一步容差必须为 0：
#   * 04 坐坐狗背景 254、狗身 255，只差一级，容差一大狗身就被吃掉（会裂一条缝）；
#   * 05 站站狗背景是红、狗身是棕，两者之间是渐变过渡，
#     容差一大会顺着渐变一路爬进狗肚子，狗身被当成背景后又被去溢色刷成一片白
#     （这就是「扣多了、里面不是白色」的根因）。
FLAT_STD = 0.6        # 判定「边缘够不够平」的标准差阈值
FLAT_FRAC = 0.95      # 判定「边缘够不够纯」的主色占比阈值
BG_TOL = 8            # 剥毛边时允许的色差：JPEG 噪点会让背景在 ±2 之间跳
BG_LAYERS = 4         # 最多往外剥几层（剥不动就停，狗的黑描边/肉色跟背景差得远）

# 04 坐坐狗是特例：背景 254、狗身 255，只差一级。
# 对它来说「差一点点的背景色」和「狗身」根本分不开，容差必须锁死为 0，
# 否则区域生长会顺着 255 的狗身一路啃进去（上一版的裂缝就是这么来的）。
BG_TOL_BY_SLUG = {"04-sit": (0, 0)}

# 源文件顺序 = 用户给的「从小到大」顺序；这里同时定了游戏里的等级顺序
SLUGS = ["01-pup", "02-drop", "03-roll", "04-sit", "05-stand",
         "06-long", "07-bib", "08-puff", "09-big", "10-god"]

# 某张图里如果还有区域生长吃不掉的水印，在这里手工补一刀（原图像素坐标）：
MANUAL_ERASE = {}


# ---------------------------------------------------------------- 基础工具
RESAMPLE_BOX = getattr(Image, "Resampling", Image).BOX


def load_work(path, work=WORK):
    """缩到工作分辨率。

    这里必须用「面积平均」（BOX）而不是 LANCZOS：
    LANCZOS 会在边缘外侧振铃，把纯色背景变成 248,114,102 这种差一级的颜色，
    而纯色底是靠「和背景色一模一样」来判定的（容差 0），振铃一出现就会被当成狗，
    成品上就是描边外面一圈背景色的毛边（05 站站狗的红边就是这么来的）。
    面积平均只在真正混色的像素上取平均，纯色区域原样保留，振铃为零。
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(1.0, work / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), RESAMPLE_BOX)
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


def near_px_for(mask):
    """「贴着主体」的判定半径（工作分辨率下的像素数）。"""
    x0, y0, x1, y1 = bbox_of(mask)
    span = max(x1 - x0, y1 - y0)
    return max(3, int(round(EDGE_GAP * span)))


def pick_subject(lab, sizes, near_px):
    """留最大的一块，并把「外框够近」的碎块并进来；离得远的（水印）丢掉。

    另外单独处理一种脏东西：贴着画面边缘、又离主体很远的孤立块。那是截图/裁剪时
    从旁边带进来的页边装饰——06 长腿狗右侧那两块绿色、黄色小贴纸就是这样：
    它们的外框和狗重叠（按老规矩会被并进来），实际离狗有 40 多像素。
    这种贴边残片一律丢掉；狗身上被截断但紧挨主体的部分（距离近）照样保留。
    判定「贴边」留 EDGE_MARGIN 像素的余量：贴纸边缘常常被削掉一两个像素，
    正好差一格碰不到最外圈，卡死「必须压在最外一行」的话就漏掉了。
    """
    h, w = lab.shape
    margin = max(1, min(int(EDGE_MARGIN), min(h, w) // 4))
    edge = np.zeros((h, w), bool)
    edge[:margin, :] = edge[-margin:, :] = True
    edge[:, :margin] = edge[:, -margin:] = True
    main = 1 + int(np.argmax(sizes[1:]))
    near = dilate4(lab == main)
    for _ in range(max(0, int(near_px) - 1)):
        near = dilate4(near)
    keep = [main]
    for _ in range(len(sizes)):
        kx0, ky0, kx1, ky1 = bbox_of(np.isin(lab, keep))
        span = max(kx1 - kx0, ky1 - ky0)
        gap = max(4, int(round(GAP * span)))
        grew = False
        for i in range(1, len(sizes)):
            if i in keep or sizes[i] < 8:
                continue
            m = lab == i
            x0, y0, x1, y1 = bbox_of(m)
            if not (x0 - gap < kx1 and x1 + gap > kx0 and y0 - gap < ky1 and y1 + gap > ky0):
                continue
            if (m & edge).any() and not (m & near).any():
                continue
            keep.append(i)
            grew = True
        if not grew:
            break
    dropped = [(sizes[i], i) for i in range(1, len(sizes)) if i not in keep and sizes[i] >= 8]
    return np.isin(lab, keep), sorted(dropped, reverse=True)


def close_mask(mask, r):
    """闭运算：先把附近的碎块粘起来再腐蚀回去，断开的细线稿就能接上。

    做之前先在四周补一圈「假背景」：MaxFilter/MinFilter 在画面边界上只看得到
    画面内的像素，腐蚀在边上会被削弱，结果就是「离画面边缘比 r 近的东西会被
    黏到边缘上」——06 长腿狗的头顶离上边 12 像素，闭运算半径 16，
    于是头顶上面凭空多出一条 12 像素高的白底色带。补一圈再裁掉就没这问题。
    """
    if r < 1:
        return mask
    size = 2 * r + 1
    h, w = mask.shape
    pad = np.zeros((h + 2 * r, w + 2 * r), bool)
    pad[r:r + h, r:r + w] = mask
    im = Image.fromarray((pad * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return (np.asarray(im) > 128)[r:r + h, r:r + w]


def fill_radius(subj):
    """按图算「够用的最小闭运算半径」——先让轮廓闭合、再灌成实心。

    大半径闭运算会把狗身两侧的凹口也一起封上（耳朵和身体之间、两条腿之间），
    灌成实心后成品上就是一块糊在凹口里的白斑：08 毛球狗耳朵下面那几块、
    09 大块头一侧的缺口都是这么来的。

    所以从 1 开始一圈一圈试，只在「再放大一圈会突然多封住一大片」时才采用那一圈
    ——那一下就是线稿真正的断口被接上了。试不到就退回 1（只补一两像素的小缝）。
    """
    span = max(subj.shape)
    r_max = max(1, int(round(FILL_R * span)))
    area = max(1, int(subj.sum()))
    prev = area
    for r in range(1, r_max + 1):
        cur = int(fill_holes(close_mask(subj, r)).sum())
        if cur - prev >= FILL_JUMP * area:
            return r
        prev = cur
    return 1


def drop_fragments(mask):
    """丢掉「剥毛边」剪下来的碎渣。

    纯色底是零容差判定的，狗周围那一圈抗锯齿像素本来就时有时无。剥毛边会顺着
    这些毛刺往里啃，把一根细毛刺从中间剪断，成品上就是描边旁边飘着的白点/白丝
    （02 垂耳狗身上有 30 来个小碎块就是这个）。碎渣都在主体旁边、面积很小，
    这里按面积一刀切掉；主体那一块永远保留。
    """
    lab, sizes = components(mask)
    if len(sizes) <= 2:
        return mask
    main = 1 + int(np.argmax(sizes[1:]))
    keep = [i for i in range(1, len(sizes)) if i == main or sizes[i] >= MIN_FRAG]
    return np.isin(lab, keep)


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
    grown = close_mask(fg, r)
    lab, sizes = components(grown)
    subj, dropped = pick_subject(lab, sizes, near_px_for(grown))
    # 线稿的肚子是「线之间漏出来的背景色」，会被判成背景：
    # 所以再把轮廓闭合一次、把里面灌成实心，狗才不是个空心线框。
    # 闭合半径按图算，刚好封住断口就行（大了会把凹口也封上，见 fill_radius）。
    solid = fill_holes(close_mask(subj, fill_radius(subj)))
    return solid, dropped, model


def border_stats(rgb):
    """看四条边：返回 (背景色, 主色占比, 边缘标准差, 边缘像素)。"""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, 3)
    cols, counts = np.unique(border, axis=0, return_counts=True)
    k = int(counts.argmax())
    return (cols[k].astype(np.int16),
            float(counts[k]) / float(len(border)),
            float(border.astype(np.float32).std(axis=0).mean()),
            border)


def dilate4(m):
    """4 邻域膨胀一圈。"""
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def flood_from_border(cand):
    """从四条边往里灌：只有 cand 为真的像素才连得通。"""
    h, w = cand.shape
    vis = np.zeros((h, w), bool)
    dq = deque()

    def seed(y, x):
        if cand[y, x] and not vis[y, x]:
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
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                seed(ny, nx)
    return vis


def is_flat_bg(rgb):
    """这张图是不是「一整块纯色底」。"""
    _bgcol, frac, std, _border = border_stats(rgb)
    return std < FLAT_STD and frac >= FLAT_FRAC


def background_layers(rgb, slug=None):
    """算背景。

    返回 (vis0, vis1)：
      vis0 —— 只认「和背景色一模一样」的像素，绝对稳，绝不会啃到狗；
      vis1 —— 在 vis0 基础上最多往外剥 BG_LAYERS 层「和背景色差一点点」的
              像素，用来吃掉 JPEG 噪点 / 缩放噪点在描边外留下的一圈毛边。
    剥不动就会自己停下：狗的黑描边和肉色跟背景差得远，永远剥不过去。
    """
    bgcol, frac, std, border = border_stats(rgb)
    if std < FLAT_STD and frac >= FLAT_FRAC:
        tol, layers = BG_TOL, BG_LAYERS
    else:
        d0 = np.abs(border.astype(np.int16) - bgcol).max(axis=1)
        tol = max(3, min(int(np.ceil(np.percentile(d0, 99.5))) + 2, 28))
        layers = 1
    if slug and slug in BG_TOL_BY_SLUG:
        tol, layers = BG_TOL_BY_SLUG[slug]
    d = np.abs(rgb.astype(np.int16) - bgcol).max(axis=2)
    vis0 = flood_from_border(d <= 0)
    vis = vis0
    if layers > 0:
        near = d <= tol
        for _ in range(layers):
            # 一圈一圈往外剥（一次一圈，别一次把整片「接近背景色」的区域吞掉）
            front = dilate4(vis) & near & ~vis
            if not front.any():
                break
            vis = vis | front
    return vis0, vis


def background_mask(rgb, slug=None):
    return background_layers(rgb, slug)[1]


def subject_mask_exact(rgb, slug=None):
    """纯色底版的取主体：不给 model，调用方据此跳过去溢色。"""
    vis0, vis1 = background_layers(rgb, slug)
    fg0 = ~vis0
    if not fg0.any():
        raise RuntimeError("纯色底模式下找不到前景")
    lab, sizes = components(fg0)
    subj, dropped = pick_subject(lab, sizes, near_px_for(fg0))
    solid = fill_holes(close_mask(subj, fill_radius(subj)))
    # 主体连通性以「零容差」那版为准，毛边只是最后再削掉一圈：
    # 这样剥毛边不会把细腿细尾巴切断，更不会让整只狗被拆成几块。
    solid = drop_fragments(solid & ~(vis1 & ~vis0))
    return solid, dropped, None


def despill(rgb, alpha, model, thr=DESPILL_THR, fill=(255, 255, 255)):
    """轮廓内部还留着「局部的背景色」（线稿是镂空的）→ 刷成白色，跟其它贴图统一。

    判断依据是逐像素的背景色模型（不是全局调色板），所以狗身上本来就是这个
    颜色的地方才会被刷白，狗本身有颜色的地方一律不动。
    只动「实心内部」：最外那圈半透明像素是抗锯齿过渡带，颜色本来就是
    「墨色混着背景色」，在这里刷白会变成贴到游戏里的一圈白边。
    容差要小（20）：狗身上本来就有浅色（09 大块头身上那层淡蓝、
    01 小奶狗身上的奶黄），它们跟背景只差二三十级；容差放宽到 44 就会
    把这些「狗自己的淡色」当成漏进来的背景一并刷白——这就是「扣多了」。"""
    d = np.abs(rgb.astype(np.float32) - model).max(axis=2)
    hit = (alpha > 235) & (d <= thr)
    out = rgb.copy()
    out[hit] = fill
    return out, int(hit.sum())


def bleed_color(rgb, alpha, steps=3):
    """把不透明区域的颜色往透明那边抹几圈。

    浏览器放大/缩小贴图时是按像素做插值的：贴图外围的透明像素如果是纯黑或纯白，
    插值就会把那个颜色带进狗的边缘，看上去就是脏边。先把最外面几圈透明像素
    染成紧挨着它的颜色，缩放就没有杂色了。"""
    h, w = alpha.shape
    done = alpha > 8
    out = rgb.astype(np.float32).copy()
    for _ in range(max(1, int(steps))):
        acc = np.zeros_like(out)
        cnt = np.zeros((h, w), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            color = np.roll(out, (dy, dx), (0, 1))
            nb = np.roll(done, (dy, dx), (0, 1))
            if dy > 0:
                nb[0, :] = False
            elif dy < 0:
                nb[-1, :] = False
            if dx > 0:
                nb[:, 0] = False
            elif dx < 0:
                nb[:, -1] = False
            acc += color * nb[..., None]
            cnt += nb
        fill = (~done) & (cnt > 0)
        if not fill.any():
            break
        out[fill] = acc[fill] / cnt[fill][..., None]
        done = done | fill
    return out


# ---------------------------------------------------------------- 主流程
def build(path, slug):
    work = load_work(path)
    rgb_w = np.asarray(work)
    bgcol, _bfrac, _bstd, _bborder = border_stats(rgb_w)
    if is_flat_bg(rgb_w):
        # 纯色底：取主体时已经灌过实心，也不用去溢色
        fg_main, dropped, model_w = subject_mask_exact(rgb_w, slug)
    else:
        fg_main, dropped, model_w = subject_mask(rgb_w)
        fg_main = fill_holes(fg_main)
    # 两条路都再筛一遍：贴图里只留「主体 + 够大的块」，孤零零的小碎渣一律不要
    fg_main = drop_fragments(fg_main)
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

    # 遮罩 → 抗锯齿的 alpha：先糊一点点做出过渡带，再重映射到 0..255。
    # 两条路都糊：以前只有「精确背景色」那条路糊，渐变底那条路直接给 0/255 硬边，
    # 放大之后边缘都是台阶（06 长腿狗的锯齿边就这么来的）。
    m = Image.fromarray((fg_main * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    solid = np.asarray(m).astype(np.float32)
    soft = np.asarray(Image.fromarray(solid.astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32)
    alpha = np.clip((soft - 60) * (255.0 / 130.0), 0, 255)

    # 每个像素「原本压着的背景色」：纯色底就是那个纯色，渐变底用拟合出来的背景色地图
    rgb = np.asarray(full).astype(np.float32)
    if model_w is None:
        bgmap = np.empty((H, W, 3), np.float32)
        bgmap[:] = bgcol.astype(np.float32)
        nspill = 0
    else:
        bgmap = np.empty((H, W, 3), np.float32)
        for c in range(3):
            bgmap[..., c] = np.asarray(
                Image.fromarray(model_w[..., c].astype(np.float32)).resize((W, H), Image.LANCZOS),
                dtype=np.float32)
        rgb, nspill = despill(rgb, alpha, bgmap)

    ys, xs = np.nonzero(alpha > 128)
    if len(ys) == 0:
        raise RuntimeError("alpha 全空：%s" % path)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # 边缘像素的颜色是「墨色 + 背景色」按 alpha 混出来的：黑描边压在白底上，
    # 半透明的边就发灰。照搬这个灰去贴游戏背景，狗周围就糊出一圈灰白脏边。
    # 先按 a·真色 = 原色 − (1−a)·背景色 反解出「预乘颜色」，缩放用预乘值算，
    # 最后再除回 alpha —— 等价于「先把边抠干净再缩放」，灰边就没了。
    a = alpha[..., None] / 255.0
    premul = np.clip(rgb - (1.0 - a) * bgmap, 0, 255)

    sub = np.dstack([premul, alpha])[y0:y1, x0:x1].astype(np.uint8)
    im = Image.fromarray(sub, "RGBA")
    target = int(round(CANVAS * FILL))
    h, w = y1 - y0, x1 - x0
    s = target / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = im.resize((nw, nh), Image.LANCZOS)

    ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    plane = np.asarray(im).astype(np.float32)
    canvas = np.zeros((CANVAS, CANVAS, 4), np.float32)
    canvas[oy:oy + nh, ox:ox + nw, :3] = plane[..., :3]
    canvas[oy:oy + nh, ox:ox + nw, 3] = plane[..., 3]

    a_out = canvas[..., 3:4] / 255.0
    true = canvas[..., :3] / np.maximum(a_out, 1.0 / 255.0)
    true[a_out[..., 0] < 0.03] = bgcol.astype(np.float32)
    canvas[..., :3] = bleed_color(np.clip(true, 0, 255), canvas[..., 3])
    out = Image.fromarray(np.clip(np.rint(canvas), 0, 255).astype(np.uint8), "RGBA")

    return {
        "slug": slug, "raw": os.path.basename(path), "src": (W, H), "srcbox": (x0, y0, x1, y1),
        "fg": round(100.0 * float(fg_main.mean()), 1),
        "fill": round(100.0 * float(fg_main.sum()) / max(1, (x1 - x0) * (y1 - y0)), 1),
        "dropped": dropped[:4], "spill": nspill, "canvas": out,
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
