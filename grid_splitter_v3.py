#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宫格图片切割工具 v3 (Fixed)
修复了选择逻辑，优先匹配宽高比
"""

import cv2
import os
import glob
import argparse
import numpy as np
from pathlib import Path


def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR 转灰度，已是灰度则直接返回。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _compute_projections(gray: np.ndarray, smooth_k: int = 15):
    """计算水平/垂直 Sobel 投影并平滑。"""
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    h_proj = np.mean(np.abs(sobel_y), axis=1)
    v_proj = np.mean(np.abs(sobel_x), axis=0)
    kernel = np.ones(smooth_k) / smooth_k
    h_smooth = np.convolve(h_proj, kernel, mode='same')
    v_smooth = np.convolve(v_proj, kernel, mode='same')
    return h_smooth, v_smooth


def _cluster_positions(positions: list, tolerance: int) -> list:
    """将相近坐标聚类，返回每簇的均值。"""
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][0] <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(round(sum(c) / len(c))) for c in clusters]


def detect_outer_borders(img: np.ndarray, max_check: int = 80) -> tuple:
    """检测图片四边的黑/白外边框。"""
    gray = _to_gray(img)
    h, w = gray.shape

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
    """验证 rows×cols 配置与投影的吻合程度。"""
    bg_h = np.median(h_smooth)
    bg_v = np.median(v_smooth)

    expected_h = [int(img_h * i / rows) for i in range(1, rows)]
    expected_v = [int(img_w * i / cols) for i in range(1, cols)]

    def _score_lines(expected: list, projection: np.ndarray, bg: float) -> list:
        scores = []
        search = max(12, int(len(projection) * 0.04))
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

    return (h_ratio + v_ratio) / 2, h_found, v_found


def count_grid_lines_hough(img: np.ndarray) -> tuple:
    """霍夫变换兜底检测，返回 (水平线数, 垂直线数)。"""
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
    """
    自动检测宫格行列数。
    修复版：当多个配置得分相近时，优先匹配宽高比。
    """
    gray = _to_gray(img)
    img_h, img_w = gray.shape
    h_smooth, v_smooth = _compute_projections(gray)

    aspect_ratio = img_w / img_h
    img_aspect = aspect_ratio

    # 标准配置列表
    all_configs = [
        # 方形
        (3, 3), (4, 4), (5, 5), (6, 6),
        # 矩形
        (5, 6), (6, 5), (6, 10), (10, 6), (5, 12), (12, 5),
        # 长条
        (2, 4), (4, 2), (2, 6), (6, 2), (2, 8), (8, 2), (2, 10), (10, 2),
    ]

    def get_config_category(rows, cols):
        if cols > rows * 1.3:
            return 'horizontal'
        elif rows > cols * 1.3:
            return 'vertical'
        return 'square'

    # 优先类别
    if aspect_ratio < 0.7:
        priority_category = 'vertical'
    elif aspect_ratio > 1.5:
        priority_category = 'horizontal'
    else:
        priority_category = 'square'

    priority_configs = [c for c in all_configs if get_config_category(c[0], c[1]) == priority_category]
    other_configs = [c for c in all_configs if get_config_category(c[0], c[1]) != priority_category]
    common_configs = priority_configs + other_configs

    best_config = None
    best_score = 0.0
    best_lines = 0
    best_aspect_diff = float('inf')
    best_cells = 0

    for rows, cols in common_configs:
        total = rows * cols
        if not (min_cells <= total <= max_cells):
            continue

        score, h_found, v_found = verify_grid_config(
            h_smooth, v_smooth, img_h, img_w, rows, cols
        )
        lines = h_found + v_found

        grid_aspect = cols / rows
        aspect_diff = abs(grid_aspect - img_aspect)

        print(f"      {rows}×{cols}: 得分={score:.2f}, "
              f"水平线={h_found}/{rows-1}, 垂直线={v_found}/{cols-1}, "
              f"宽高比差异={aspect_diff:.3f}")

        # ===== 修复的选择逻辑 =====
        total_cells = rows * cols

        # 关键修复：当得分相近（差距在 0.1 以内）时，优先选择宽高比匹配度更高的
        if score > best_score + 0.1:
            # 得分明显更高，直接选择
            best_score, best_config, best_lines, best_aspect_diff, best_cells = score, (rows, cols), lines, aspect_diff, total_cells
        elif abs(score - best_score) <= 0.1:
            # 得分相近，优先选择宽高比匹配度更高的
            if aspect_diff < best_aspect_diff:
                best_score, best_config, best_lines, best_aspect_diff, best_cells = score, (rows, cols), lines, aspect_diff, total_cells
            elif aspect_diff == best_aspect_diff and lines > best_lines:
                # 宽高比相同，选择找到更多网格线的
                best_config, best_lines = (rows, cols), lines

    if best_config and best_score >= min_score:
        return best_config

    # 霍夫变换兜底
    print("      尝试霍夫变换检测...")
    h_cnt, v_cnt = count_grid_lines_hough(img)
    rows_h, cols_h = h_cnt + 1, v_cnt + 1
    total_h = rows_h * cols_h
    print(f"      霍夫检测: {h_cnt} 水平线, {v_cnt} 垂直线 → {rows_h}×{cols_h}")

    if min_cells <= total_h <= max_cells and rows_h >= 2 and cols_h >= 2:
        grid_aspect = cols_h / rows_h
        img_aspect = img_w / img_h
        aspect_diff = abs(grid_aspect - img_aspect)

        if aspect_diff < 0.5:
            print(f"      霍夫兜底: {rows_h}×{cols_h} (宽高比差异: {aspect_diff:.3f})")
            return rows_h, cols_h
        else:
            print(f"      霍夫结果 {rows_h}×{cols_h} 宽高比差异过大 ({aspect_diff:.3f})，放弃")

    print("      无法自动检测宫格，请手动指定 -r / -c 参数")
    return None


# 其余代码保持不变...
def find_grid_lines(img: np.ndarray, rows: int, cols: int,
                    search_range: int = 25) -> tuple:
    """在期望位置附近搜索投影峰值，精确定位每条内部网格线。"""
    gray = _to_gray(img)
    img_h, img_w = gray.shape
    h_smooth, v_smooth = _compute_projections(gray, smooth_k=10)

    def _find_peaks(expected_positions: list, projection: np.ndarray,
                    length: int) -> list:
        peaks = []
        for exp in expected_positions:
            lo = max(0, exp - search_range)
            hi = min(length, exp + search_range)
            offset = np.argmax(projection[lo:hi])
            peaks.append(lo + int(offset))
        return peaks

    expected_h = [int(img_h * i / rows) for i in range(1, rows)]
    expected_v = [int(img_w * i / cols) for i in range(1, cols)]

    h_lines = _find_peaks(expected_h, h_smooth, img_h)
    v_lines = _find_peaks(expected_v, v_smooth, img_w)

    return h_lines, v_lines


def _measure_line_width(projection: np.ndarray, peak: int,
                        threshold_ratio: float = 0.5) -> int:
    """在投影上测量峰值附近高于阈值的宽度。"""
    peak_val = projection[peak]
    threshold = peak_val * threshold_ratio
    lo, hi = peak, peak

    while lo > 0 and projection[lo - 1] >= threshold:
        lo -= 1
    while hi < len(projection) - 1 and projection[hi + 1] >= threshold:
        hi += 1

    return max(1, hi - lo + 1)


def split_grid_clean(image_path: str, output_folder: str,
                     rows: int, cols: int,
                     auto_detect: bool = True) -> int:
    """宫格切割主函数。"""
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return 0

    h, w = img.shape[:2]
    print(f"原始尺寸: {w}×{h}")

    if auto_detect:
        borders = detect_outer_borders(img)
        top, bot, left, right = borders
        if sum(borders) > 0:
            print(f"检测到外边框: 上={top} 下={bot} 左={left} 右={right} px")
            b_bot = h - bot if bot > 0 else h
            b_right = w - right if right > 0 else w
            img = img[top:b_bot, left:b_right]
            h, w = img.shape[:2]
            print(f"去除外边框后尺寸: {w}×{h}")

    print(f"正在分析 {rows}×{cols} 网格...")
    h_lines, v_lines = find_grid_lines(img, rows, cols)
    print(f"水平网格线: {h_lines}")
    print(f"垂直网格线: {v_lines}")

    gray = _to_gray(img)
    h_smooth, v_smooth = _compute_projections(gray, smooth_k=10)

    h_widths = [_measure_line_width(h_smooth, y) for y in h_lines] if h_lines else [0]
    v_widths = [_measure_line_width(v_smooth, x) for x in v_lines] if v_lines else [0]
    margin_h = max(h_widths) // 2 + 3
    margin_v = max(v_widths) // 2 + 3
    margin_h = max(4, min(40, margin_h))
    margin_v = max(4, min(40, margin_v))
    print(f"自适应 margin: 水平方向={margin_h}px, 垂直方向={margin_v}px")

    h_positions = [0] + h_lines + [h]
    v_positions = [0] + v_lines + [w]

    rows_info = []
    for i in range(rows):
        # 修复：对所有边都应用 margin，不只是内部边
        y1 = h_positions[i] + margin_h
        y2 = h_positions[i+1] - margin_h
        # 确保不越界
        y1 = min(max(0, y1), h)
        y2 = min(max(0, y2), h)
        # 确保 y1 < y2
        if y1 >= y2:
            y1 = h_positions[i]
            y2 = h_positions[i+1]
        rows_info.append((y1, y2))

    cols_info = []
    for j in range(cols):
        # 修复：对所有边都应用 margin，不只是内部边
        x1 = v_positions[j] + margin_v
        x2 = v_positions[j+1] - margin_v
        # 确保不越界
        x1 = min(max(0, x1), w)
        x2 = min(max(0, x2), w)
        # 确保 x1 < x2
        if x1 >= x2:
            x1 = v_positions[j]
            x2 = v_positions[j+1]
        cols_info.append((x1, x2))

    cell_heights = [y2 - y1 for y1, y2 in rows_info if y2 > y1]
    cell_widths = [x2 - x1 for x1, x2 in cols_info if x2 > x1]

    if not cell_heights or not cell_widths:
        print("[ERROR] 无法计算有效格子尺寸，请检查行列参数")
        return 0

    target_h = min(cell_heights)
    target_w = min(cell_widths)
    print(f"目标格子尺寸: {target_w}×{target_h}")

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for i, (y1, y2) in enumerate(rows_info):
        for j, (x1, x2) in enumerate(cols_info):
            count += 1

            if x2 <= x1 or y2 <= y1:
                print(f"  [WARN] 格子({i},{j}) 区域无效，跳过")
                continue

            crop = img[y1:y2, x1:x2]

            if crop.shape[0] != target_h or crop.shape[1] != target_w:
                crop = cv2.resize(crop, (target_w, target_h),
                                  interpolation=cv2.INTER_LANCZOS4)

            out_path = os.path.join(output_folder, f"clip_{count:03d}.jpg")
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if ok:
                buf.tofile(out_path)
            else:
                print(f"  [WARN] 编码失败: {out_path}")

    print(f"[DONE] 切割完成，共 {count} 张，统一尺寸 {target_w}×{target_h}")
    print(f"输出目录: {output_folder}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description='宫格图片切割工具 v3 - 干净、精准、尺寸一致',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python grid_splitter_v3.py -i "图片.png"            # 自动检测
  python grid_splitter_v3.py -r 5 -c 5 -i "图片.png" # 手动 5×5
  python grid_splitter_v3.py -r 3 -c 3               # 当前目录所有图片
  python grid_splitter_v3.py -r 5 -c 5 -i "图片.png" --no-auto
  python grid_splitter_v3.py -r 5 -c 5 -i "图片.png" -o "my_output"
"""
    )
    parser.add_argument('--rows', '-r', type=int, default=None,
                        help='行数（不指定则自动检测）')
    parser.add_argument('--cols', '-c', type=int, default=None,
                        help='列数（不指定则自动检测）')
    parser.add_argument('--input', '-i', type=str, default='*.png',
                        help='输入图片路径或通配符（默认 *.png）')
    parser.add_argument('--output', '-o', type=str, default='output_result',
                        help='输出文件夹前缀（默认 output_result）')
    parser.add_argument('--no-auto', action='store_true',
                        help='禁用自动外边框检测')
    parser.add_argument('--auto-grid', action='store_true',
                        help='强制自动检测宫格（即使已指定 -r/-c）')
    args = parser.parse_args()

    if args.input == '*.png':
        image_files = (glob.glob('*.png') + glob.glob('*.jpg')
                       + glob.glob('*.jpeg') + glob.glob('*.PNG')
                       + glob.glob('*.JPG') + glob.glob('*.JPEG'))
    else:
        image_files = glob.glob(args.input)

    if not image_files:
        print(f"[ERROR] 未找到图片文件: {args.input}")
        print("提示: 确保图片在当前目录，或用 -i 指定路径")
        return

    need_auto = args.auto_grid or (args.rows is None or args.cols is None)

    print(f"找到 {len(image_files)} 个图片文件")
    if args.rows and args.cols and not args.auto_grid:
        print(f"网格设置: {args.rows}行 × {args.cols}列 = {args.rows * args.cols} 格")
    print(f"自动去边框: {'禁用' if args.no_auto else '启用'}")
    print("=" * 55)

    for img_path in image_files:
        print(f"\n正在处理: {img_path}")
        print("-" * 55)

        rows, cols = args.rows, args.cols

        if need_auto:
            print("[INFO] 自动检测宫格行列数...")
            tmp = cv2.imread(img_path)
            if tmp is None:
                tmp = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            if tmp is None:
                print(f"[ERROR] 无法读取: {img_path}")
                continue

            borders = detect_outer_borders(tmp)
            th, tw = tmp.shape[:2]
            if sum(borders) > 0:
                top, bot, left, right = borders
                tmp = tmp[top:th - bot if bot else th,
                           left:tw - right if right else tw]

            # 使用修复的检测函数
            detected = auto_detect_grid_dimensions(tmp)
            if detected:
                rows, cols = detected
                print(f"[INFO] 自动检测: {rows}行 × {cols}列 = {rows * cols} 格")
            elif rows is None or cols is None:
                print("[ERROR] 自动检测失败，请手动指定 -r 和 -c")
                continue
            else:
                print(f"[WARN] 自动检测失败，使用手动指定: {rows}×{cols}")

        out_dir = f"{args.output}/{Path(img_path).stem}"
        split_grid_clean(
            image_path=img_path,
            output_folder=out_dir,
            rows=rows,
            cols=cols,
            auto_detect=not args.no_auto,
        )


if __name__ == '__main__':
    main()
