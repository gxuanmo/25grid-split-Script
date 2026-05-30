#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宫格图片切割工具 v4
基于 v3，做以下改进：
  1. 修复 _cluster_positions 末尾元素比较 bug
  2. 改进选择逻辑：用 min_found_per_dim 替代宽高比作为分接器
  3. 修复 glob 在 Windows 大小写不敏感导致的重复处理
  4. 清理输出目录避免旧文件残留
  5. 切割后自动裁剪网格线残留（白边/黑边）
"""

import cv2
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

# Windows 控制台默认 GBK，会把中文打成乱码；统一切到 UTF-8
if sys.platform == 'win32':
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

# 中文路径下 cv2.imread 一律失败并打印 [WARN:0] 噪音，统一只走 imdecode
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def _imread_unicode(path: str):
    """Windows + 中文路径安全的图片读取。失败返回 None。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, IOError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _compute_projections(gray: np.ndarray, smooth_k: int = 15):
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    h_proj = np.mean(np.abs(sobel_y), axis=1)
    v_proj = np.mean(np.abs(sobel_x), axis=0)
    kernel = np.ones(smooth_k) / smooth_k
    h_smooth = np.convolve(h_proj, kernel, mode='same')
    v_smooth = np.convolve(v_proj, kernel, mode='same')
    return h_smooth, v_smooth


def _cluster_positions(positions: list, tolerance: int) -> list:
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for p in positions[1:]:
        # v4 fix: compare with LAST element, not first
        if p - clusters[-1][-1] <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(round(sum(c) / len(c))) for c in clusters]


def detect_outer_borders(img: np.ndarray, max_check: int | None = None) -> tuple:
    gray = _to_gray(img)
    h, w = gray.shape
    # 自适应：大图 80px 远不够，按短边 5% 兜底
    if max_check is None:
        max_check = max(80, min(h, w) // 20)

    def _is_border_line(line: np.ndarray) -> bool:
        mean = np.mean(line)
        p5 = np.percentile(line, 5)
        p95 = np.percentile(line, 95)
        if mean < 30 and p95 < 60:
            return True
        if mean > 220 and p5 > 180:
            return True
        return False

    borders = [0, 0, 0, 0]
    for i in range(min(max_check, h)):
        if _is_border_line(gray[i, :]):
            borders[0] = i + 1
        else:
            break
    for i in range(min(max_check, h)):
        if _is_border_line(gray[h - 1 - i, :]):
            borders[1] = i + 1
        else:
            break
    for i in range(min(max_check, w)):
        if _is_border_line(gray[:, i]):
            borders[2] = i + 1
        else:
            break
    for i in range(min(max_check, w)):
        if _is_border_line(gray[:, w - 1 - i]):
            borders[3] = i + 1
        else:
            break
    return tuple(borders)


def verify_grid_config(h_smooth: np.ndarray, v_smooth: np.ndarray,
                       img_h: int, img_w: int,
                       rows: int, cols: int,
                       snr_threshold: float = 2.0) -> tuple:
    bg_h = np.median(h_smooth)
    bg_v = np.median(v_smooth)

    expected_h = [int(img_h * i / rows) for i in range(1, rows)]
    expected_v = [int(img_w * i / cols) for i in range(1, cols)]

    def _score_lines(expected, projection, bg):
        scores = []
        base_search = max(12, int(len(projection) * 0.04))
        # v4 fix: 自适应搜索范围——预期线条越少，窗口越窄
        # 防止稀疏配置（如 2x4 只需 1+3=4 条线）因搜索范围过宽
        # 而从内容边缘获得虚假满分
        n = len(expected)
        search = max(8, int(base_search * min(1.0, n / 3)))
        for exp in expected:
            lo = max(0, exp - search)
            hi = min(len(projection), exp + search)
            peak = projection[lo:hi].max()
            scores.append(peak / (bg + 1e-6))
        return scores

    h_scores = _score_lines(expected_h, h_smooth, bg_h)
    v_scores = _score_lines(expected_v, v_smooth, bg_v)

    h_found = sum(1 for s in h_scores if s > snr_threshold)
    v_found = sum(1 for s in v_scores if s > snr_threshold)

    h_ratio = h_found / len(expected_h) if expected_h else 1.0
    v_ratio = v_found / len(expected_v) if expected_v else 1.0

    score = (h_ratio + v_ratio) / 2

    return score, h_found, v_found


def count_grid_lines_hough(img: np.ndarray) -> tuple:
    gray = _to_gray(img)
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    threshold = int(min(h, w) * 0.2)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return 0, 0
    h_pos, v_pos = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 8 or theta > np.pi - np.pi / 8:
            v_pos.append(int(rho * np.cos(theta)))
        elif np.pi / 2 - np.pi / 8 < theta < np.pi / 2 + np.pi / 8:
            h_pos.append(int(rho * np.sin(theta)))
    tol = int(min(h, w) * 0.02)
    return len(_cluster_positions(h_pos, tol)), len(_cluster_positions(v_pos, tol))


def auto_detect_grid_dimensions(img: np.ndarray,
                                min_cells: int = 4,
                                max_cells: int = 100,
                                min_score: float = 0.5) -> tuple | None:
    gray = _to_gray(img)
    img_h, img_w = gray.shape
    h_smooth, v_smooth = _compute_projections(gray)

    aspect_ratio = img_w / img_h
    img_aspect = aspect_ratio

    all_configs = [
        (2, 2), (2, 3), (3, 2),
        (3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (3, 6), (6, 3),
        (4, 4), (4, 5), (5, 4), (4, 6), (6, 4),
        (5, 5), (5, 6), (6, 5), (5, 12), (12, 5),
        (6, 6), (6, 10), (10, 6), (7, 7),
        (2, 4), (4, 2), (2, 6), (6, 2), (2, 8), (8, 2), (2, 10), (10, 2),
    ]

    def get_config_category(rows, cols):
        if cols > rows * 1.3:
            return 'horizontal'
        elif rows > cols * 1.3:
            return 'vertical'
        return 'square'

    if aspect_ratio < 0.7:
        priority_category = 'vertical'
    elif aspect_ratio > 1.5:
        priority_category = 'horizontal'
    else:
        priority_category = 'square'

    priority_configs = [c for c in all_configs if get_config_category(c[0], c[1]) == priority_category]
    other_configs = [c for c in all_configs if get_config_category(c[0], c[1]) != priority_category]
    common_configs = priority_configs + other_configs

    # ──── 阶段一：评分和选择 ────
    # v4 改进：用 min_found_per_dim 替代宽高比作为分接器
    # 阈值收紧为 0.08（v3 用 0.1）
    TIE_THRESHOLD = 0.08
    results = []   # (rows, cols, score, h_found, v_found, lines)
    best_config = None
    best_score = 0.0
    best_min_found = 0      # min(h_found, v_found) 最小维检出数
    best_total_found = 0    # h_found + v_found

    for rows, cols in common_configs:
        total = rows * cols
        if not (min_cells <= total <= max_cells):
            continue

        score, h_found, v_found = verify_grid_config(
            h_smooth, v_smooth, img_h, img_w, rows, cols
        )
        lines = h_found + v_found
        min_found = min(h_found, v_found)

        print(f"      {rows}x{cols}: score={score:.2f}, "
              f"H={h_found}/{rows-1}, V={v_found}/{cols-1}, "
              f"min_dim={min_found}")

        results.append((rows, cols, score, h_found, v_found, lines))

        # 选择逻辑：
        # 1. score 明显更高（差距 > THRESHOLD）→ 直接选
        # 2. score 相近（差距 <= THRESHOLD）→ 按 min_found → score → total_found 选
        if score > best_score + TIE_THRESHOLD:
            best_score, best_config = score, (rows, cols)
            best_min_found, best_total_found = min_found, lines
        elif abs(score - best_score) <= TIE_THRESHOLD:
            # 优先：最小维检出数更多 → 对两个维度都有信心
            if min_found > best_min_found:
                best_score, best_config = score, (rows, cols)
                best_min_found, best_total_found = min_found, lines
            elif min_found == best_min_found:
                # 次优先：score 更高 → 检出率更好
                if score > best_score:
                    best_score, best_config = score, (rows, cols)
                    best_min_found, best_total_found = min_found, lines
                elif score == best_score:
                    # 再次：总检出线条数更多 → 更多证据
                    if lines > best_total_found:
                        best_config = (rows, cols)
                        best_total_found = lines

    if best_config and best_score >= min_score:
        r, c = best_config
        s = next(x[2] for x in results if x[0] == r and x[1] == c)
        print(f"      -> {r}x{c} (score={s:.2f})")
        return best_config

    # 霍夫变换兜底
    print("      Hough fallback...")
    h_cnt, v_cnt = count_grid_lines_hough(img)
    rows_h, cols_h = h_cnt + 1, v_cnt + 1
    total_h = rows_h * cols_h
    print(f"      Hough: {h_cnt}H + {v_cnt}V -> {rows_h}x{cols_h}")

    if min_cells <= total_h <= max_cells and rows_h >= 2 and cols_h >= 2:
        grid_aspect = cols_h / rows_h
        aspect_diff = abs(grid_aspect - img_w / img_h)
        if aspect_diff < 0.5:
            print(f"      Hough accepted: {rows_h}x{cols_h}")
            return rows_h, cols_h
        else:
            print(f"      Hough rejected: ar_diff={aspect_diff:.3f}")

    print("      [FAIL] use -r/-c manually")
    return None


# ──────────────────────────────────────────────────────
# 以下代码与 v3 完全一致
# ──────────────────────────────────────────────────────

def find_grid_lines(img: np.ndarray, rows: int, cols: int,
                    search_range: int = 25) -> tuple:
    gray = _to_gray(img)
    img_h, img_w = gray.shape
    h_smooth, v_smooth = _compute_projections(gray, smooth_k=10)

    def _find_peaks(expected_positions, projection, length):
        peaks = []
        n = len(expected_positions)
        if n == 0:
            return peaks
        # 期望相邻间距（cell 高/宽）—— 自适应搜索窗 + 强制单调
        cell_span = length / (n + 1)
        # 搜索窗最多 cell_span 的 1/3，避免相邻峰互相抢
        adaptive_sr = max(8, min(search_range, int(cell_span / 3)))
        # 相邻峰至少要隔 cell_span 的 1/3，否则视为重复
        min_gap = max(2, int(cell_span / 3))
        for idx, exp in enumerate(expected_positions):
            lo = max(0, exp - adaptive_sr)
            hi = min(length, exp + adaptive_sr)
            # 强制单调：下界推到上一个峰之后
            if peaks:
                lo = max(lo, peaks[-1] + min_gap)
            if lo >= hi:
                # 搜索窗被压扁——退而求其次，沿期望位置 + 最小步长
                fallback = max(exp, (peaks[-1] + min_gap) if peaks else exp)
                peaks.append(min(length - 1, fallback))
                continue
            offset = int(np.argmax(projection[lo:hi]))
            peaks.append(lo + offset)
        return peaks

    expected_h = [int(img_h * i / rows) for i in range(1, rows)]
    expected_v = [int(img_w * i / cols) for i in range(1, cols)]
    h_lines = _find_peaks(expected_h, h_smooth, img_h)
    v_lines = _find_peaks(expected_v, v_smooth, img_w)
    return h_lines, v_lines


def _measure_line_width(projection: np.ndarray, peak: int,
                        threshold_ratio: float = 0.5) -> int:
    peak_val = projection[peak]
    threshold = peak_val * threshold_ratio
    lo, hi = peak, peak
    while lo > 0 and projection[lo - 1] >= threshold:
        lo -= 1
    while hi < len(projection) - 1 and projection[hi + 1] >= threshold:
        hi += 1
    return max(1, hi - lo + 1)


def _trim_cell_borders(cell: np.ndarray, max_trim: int = 10) -> np.ndarray:
    """裁剪切割后残留的网格线（白边/黑边）。
    只裁剪接近纯白（>240）或纯黑（<15）的均匀色带，避免误裁实际内容。
    """
    gray = _to_gray(cell) if cell.ndim == 3 else cell
    h, w = gray.shape
    if h < max_trim * 4 or w < max_trim * 4:
        return cell

    def _is_grid_line(pixels):
        m = np.mean(pixels)
        return (m > 250 or m < 5) and np.std(pixels) < 8

    top = bot = left = right = 0

    for i in range(min(max_trim, h // 4)):
        if _is_grid_line(gray[i, :]):
            top = i + 1
        else:
            break
    for i in range(min(max_trim, h // 4)):
        if _is_grid_line(gray[h - 1 - i, :]):
            bot = i + 1
        else:
            break
    for i in range(min(max_trim, w // 4)):
        if _is_grid_line(gray[:, i]):
            left = i + 1
        else:
            break
    for i in range(min(max_trim, w // 4)):
        if _is_grid_line(gray[:, w - 1 - i]):
            right = i + 1
        else:
            break

    if top or bot or left or right:
        b_bot = h - bot if bot else h
        b_right = w - right if right else w
        return cell[top:b_bot, left:b_right]
    return cell


def split_grid_clean(image_path: str, output_folder: str,
                     rows: int, cols: int,
                     auto_detect: bool = True) -> int:
    if rows < 1 or cols < 1:
        print(f"[ERROR] invalid grid {rows}x{cols}")
        return 0
    img = _imread_unicode(image_path)
    if img is None:
        print(f"[ERROR] cannot read: {image_path}")
        return 0

    h, w = img.shape[:2]
    print(f"size: {w}x{h}")

    if auto_detect:
        borders = detect_outer_borders(img)
        top, bot, left, right = borders
        if sum(borders) > 0:
            print(f"border: top={top} bot={bot} left={left} right={right}")
            b_bot = h - bot if bot > 0 else h
            b_right = w - right if right > 0 else w
            img = img[top:b_bot, left:b_right]
            h, w = img.shape[:2]
            print(f"after crop: {w}x{h}")

    print(f"grid {rows}x{cols}...")
    h_lines, v_lines = find_grid_lines(img, rows, cols)
    print(f"H-lines: {h_lines}")
    print(f"V-lines: {v_lines}")

    gray = _to_gray(img)
    h_smooth, v_smooth = _compute_projections(gray, smooth_k=10)

    h_widths = [_measure_line_width(h_smooth, y) for y in h_lines] if h_lines else [0]
    v_widths = [_measure_line_width(v_smooth, x) for x in v_lines] if v_lines else [0]
    margin_h = max(4, min(40, max(h_widths) // 2 + 3))
    margin_v = max(4, min(40, max(v_widths) // 2 + 3))
    print(f"margin: H={margin_h}px, V={margin_v}px")

    h_positions = [0] + h_lines + [h]
    v_positions = [0] + v_lines + [w]

    rows_info = []
    for i in range(rows):
        # 边缘格子不加 margin：positions[0]=0 和 positions[-1]=h 是图片边界，不是网格线
        y1 = h_positions[i] + (margin_h if i > 0 else 0)
        y2 = h_positions[i + 1] - (margin_h if i < rows - 1 else 0)
        y1 = min(max(0, y1), h)
        y2 = min(max(0, y2), h)
        if y1 >= y2:
            y1, y2 = h_positions[i], h_positions[i + 1]
        rows_info.append((y1, y2))

    cols_info = []
    for j in range(cols):
        x1 = v_positions[j] + (margin_v if j > 0 else 0)
        x2 = v_positions[j + 1] - (margin_v if j < cols - 1 else 0)
        x1 = min(max(0, x1), w)
        x2 = min(max(0, x2), w)
        if x1 >= x2:
            x1, x2 = v_positions[j], v_positions[j + 1]
        cols_info.append((x1, x2))

    cell_heights = [y2 - y1 for y1, y2 in rows_info if y2 > y1]
    cell_widths = [x2 - x1 for x1, x2 in cols_info if x2 > x1]

    if not cell_heights or not cell_widths:
        print("[ERROR] invalid cell size")
        return 0

    target_h = min(cell_heights)
    target_w = min(cell_widths)
    print(f"cell: {target_w}x{target_h}")

    # 清理旧输出文件，避免上次运行的残留
    if os.path.exists(output_folder):
        for old in glob.glob(os.path.join(output_folder, 'clip_*.jpg')):
            os.remove(old)
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    skipped = 0
    for i, (y1, y2) in enumerate(rows_info):
        for j, (x1, x2) in enumerate(cols_info):
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            count += 1
            crop = img[y1:y2, x1:x2]
            crop = _trim_cell_borders(crop)
            if crop.shape[0] != target_h or crop.shape[1] != target_w:
                crop = cv2.resize(crop, (target_w, target_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            out_path = os.path.join(output_folder, f"clip_{count:03d}.jpg")
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                buf.tofile(out_path)

    if skipped:
        print(f"[WARN] {skipped} empty cells skipped")
    print(f"[DONE] {count} cells, {target_w}x{target_h}")
    print(f"output: {output_folder}")
    return count


def main():
    parser = argparse.ArgumentParser(description='Grid splitter v4')
    parser.add_argument('--rows', '-r', type=int, default=None)
    parser.add_argument('--cols', '-c', type=int, default=None)
    parser.add_argument('--input', '-i', type=str, default='*.png')
    parser.add_argument('--output', '-o', type=str, default='output_result_v4')
    parser.add_argument('--no-auto', action='store_true')
    parser.add_argument('--auto-grid', action='store_true')
    args = parser.parse_args()

    # 参数校验：避免 -r 0、-c -1 这种悄悄走到「invalid cell size」
    if args.rows is not None and args.rows < 1:
        parser.error('--rows must be >= 1')
    if args.cols is not None and args.cols < 1:
        parser.error('--cols must be >= 1')
    # -r 5 + 不带 -c 会被默默 auto-detect 覆盖 → 必须显式
    if (args.rows is None) != (args.cols is None):
        parser.error('--rows and --cols must be specified together '
                     '(or omit both for auto-detect)')

    if args.input == '*.png':
        raw = (glob.glob('*.png') + glob.glob('*.jpg')
               + glob.glob('*.jpeg') + glob.glob('*.PNG')
               + glob.glob('*.JPG') + glob.glob('*.JPEG'))
    else:
        raw = glob.glob(args.input)

    # Windows 文件系统大小写不敏感，去重
    seen = set()
    image_files = []
    for f in raw:
        key = os.path.normcase(os.path.normpath(f))
        if key not in seen:
            seen.add(key)
            image_files.append(f)

    if not image_files:
        print(f"[ERROR] no files: {args.input}")
        return

    # 只有两者都没指定时才走自动；--auto-grid 强制覆盖手动值
    need_auto = args.auto_grid or (args.rows is None and args.cols is None)
    print(f"found {len(image_files)} images")
    print("=" * 55)

    ok_count = 0
    fail_count = 0
    for img_path in image_files:
        print(f"\n>>> {img_path}")
        print("-" * 55)

        rows, cols = args.rows, args.cols

        try:
            if need_auto:
                print("[AUTO] detecting grid...")
                tmp = _imread_unicode(img_path)
                if tmp is None:
                    print(f"[ERROR] cannot read: {img_path}")
                    fail_count += 1
                    continue

                borders = detect_outer_borders(tmp)
                th, tw = tmp.shape[:2]
                if sum(borders) > 0:
                    top, bot, left, right = borders
                    tmp = tmp[top: th - bot if bot else th,
                              left: tw - right if right else tw]

                detected = auto_detect_grid_dimensions(tmp)
                if detected:
                    rows, cols = detected
                    print(f"[AUTO] {rows}x{cols} = {rows * cols} cells")
                elif rows is None or cols is None:
                    print("[ERROR] auto-detect failed, use -r/-c")
                    fail_count += 1
                    continue
                else:
                    print(f"[WARN] fallback to manual: {rows}x{cols}")

            out_dir = os.path.join(args.output, Path(img_path).stem)
            n = split_grid_clean(
                image_path=img_path,
                output_folder=out_dir,
                rows=rows,
                cols=cols,
                auto_detect=not args.no_auto,
            )
            if n > 0:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as e:
            # 单张挂掉不要拖死整批
            print(f"[ERROR] {img_path}: {type(e).__name__}: {e}")
            fail_count += 1

    print("\n" + "=" * 55)
    print(f"summary: ok={ok_count}, fail={fail_count}, total={len(image_files)}")


if __name__ == '__main__':
    main()
