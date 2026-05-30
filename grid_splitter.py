import cv2
import os
import numpy as np
import argparse
from pathlib import Path

def count_grid_lines(img):
    """
    使用霍夫变换检测直线，直接数出网格线数量
    只保留跨越大部分图像的长直线
    返回: (horizontal_lines_count, vertical_lines_count)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 1. 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 2. 霍夫变换检测直线
    hough_threshold = int(min(h, w) * 0.2)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_threshold)

    if lines is None:
        return 0, 0

    horizontal_positions = []
    vertical_positions = []

    min_line_length = min(h, w) * 0.7  # 线条至少要跨越70%的图像

    for line in lines:
        rho, theta = line[0]

        # 计算线条长度（估计）
        if theta < np.pi/8 or theta > np.pi - np.pi/8:
            # 接近垂直的线
            x = rho * np.cos(theta)
            vertical_positions.append(int(x))
        elif np.pi/2 - np.pi/8 < theta < np.pi/2 + np.pi/8:
            # 接近水平的线
            y = rho * np.sin(theta)
            horizontal_positions.append(int(y))

    def cluster_positions(positions, tolerance):
        """聚类相近的位置"""
        if not positions:
            return []
        positions = sorted(positions)
        clusters = [[positions[0]]]
        for p in positions[1:]:
            if p - clusters[-1][-1] <= tolerance:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [int(sum(c)/len(c)) for c in clusters]

    h_lines = cluster_positions(horizontal_positions, int(min(h, w) * 0.02))
    v_lines = cluster_positions(vertical_positions, int(min(w, h) * 0.02))

    return len(h_lines), len(v_lines)


def verify_grid_config(img, rows, cols):
    """
    严格验证 rows×cols 配置
    标准：线条峰值必须强、间距必须一致、必须显著高于背景
    返回: (score, h_found, v_found)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Sobel边缘检测 + 投影
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    h_proj = np.mean(np.abs(sobel_y), axis=1)
    v_proj = np.mean(np.abs(sobel_x), axis=0)

    # 平滑
    h_smooth = np.convolve(h_proj, np.ones(15)/15, mode='same')
    v_smooth = np.convolve(v_proj, np.ones(15)/15, mode='same')

    # 计算背景水平（取中间区域的均值）
    bg_h = np.mean(h_smooth[int(h*0.3):int(h*0.7)])
    bg_v = np.mean(v_smooth[int(w*0.3):int(w*0.7)])

    # 预期网格线位置
    expected_h = [int(h * i / rows) for i in range(1, rows)]
    expected_v = [int(w * i / cols) for i in range(1, cols)]

    # 检测峰值
    def score_lines(expected, projection, bg, is_horizontal=True):
        scores = []
        for exp in expected:
            search_range = max(10, int(len(projection) * 0.05))
            start = max(0, exp - search_range)
            end = min(len(projection), exp + search_range)

            peak_idx = start + np.argmax(projection[start:end])
            peak_val = projection[peak_idx]

            # 峰值必须显著高于背景
            snr = peak_val / (bg + 1)
            scores.append(snr)

        return scores

    h_scores = score_lines(expected_h, h_smooth, bg_h)
    v_scores = score_lines(expected_v, v_smooth, bg_v)

    # 计算匹配分数
    h_found = sum(1 for s in h_scores if s > 2.0)  # 信噪比 > 2
    v_found = sum(1 for s in v_scores if s > 2.0)

    # 总分：找到的线条数 / 预期线条数
    h_ratio = h_found / len(expected_h) if expected_h else 1
    v_ratio = v_found / len(expected_v) if expected_v else 1

    score = (h_ratio + v_ratio) / 2

    return score, h_found, v_found


def auto_detect_grid_dimensions(img_path, min_cells=4, max_cells=100):
    """
    自动检测图片的宫格行列数
    方法：遍历常见配置，用网格线位置验证哪个正确
    """
    img = cv2.imread(img_path)
    if img is None:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    h, w = img.shape[:2]

    # 去除外边框
    borders = detect_outer_borders(img)
    if sum(borders) > 0:
        img = img[borders[0]:h-borders[1], borders[2]:w-borders[3]]
        h, w = img.shape[:2]

    # 常见宫格配置（按格子数排序）
    # 包含方形网格、竖向长条、横向长条等多种布局
    common_configs = [
        # 方形网格
        (3, 3),    # 9宫格
        (4, 4),    # 16宫格
        (5, 5),    # 25宫格
        (5, 6),    # 30宫格
        (6, 6),    # 36宫格
        (6, 10),   # 60宫格
        (10, 6),   # 60宫格
        (5, 12),   # 60宫格
        (12, 5),   # 60宫格
        # 竖向长条（2列多行）
        (2, 6),    # 12格竖条
        (2, 7),    # 14格竖条
        (2, 8),    # 16格竖条
        (2, 9),    # 18格竖条
        (2, 10),   # 20格竖条
        # 横向长条（多行2列）
        (6, 2),    # 12格横条
        (7, 2),    # 14格横条
        (8, 2),    # 16格横条
        # 其他常用配置
        (3, 4),    # 12格
        (4, 3),    # 12格
    ]

    # 遍历所有配置，用网格线位置验证
    best_config = None
    best_score = 0
    best_lines_found = 0

    for rows, cols in common_configs:
        total = rows * cols
        if total < min_cells or total > max_cells:
            continue

        score, h_found, v_found = verify_grid_config(img, rows, cols)
        lines_found = h_found + v_found
        print(f"      {rows}x{cols}: 得分={score:.2f}, 水平线={h_found}/{rows-1}, 垂直线={v_found}/{cols-1}")

        # 选择策略：分数优先，其次是格子数（选择更紧凑的配置）
        current_cells = rows * cols
        if score > best_score + 0.01:
            # 分数明显更高，选择此配置
            best_score = score
            best_config = (rows, cols)
            best_lines_found = lines_found
            best_cells = current_cells
        elif abs(score - best_score) <= 0.01:
            # 分数相近时，优先选择格子数更少（更紧凑）的配置
            if 'best_cells' not in locals() or current_cells < best_cells or (current_cells == best_cells and lines_found > best_lines_found):
                best_config = (rows, cols)
                best_lines_found = lines_found
                best_cells = current_cells

    if best_config:
        return best_config

    # 如果没有验证通过的，尝试霍夫变换检测
    h_line_count, v_line_count = count_grid_lines(img)
    rows = h_line_count + 1
    cols = v_line_count + 1
    total = rows * cols

    if min_cells <= total <= max_cells and rows >= 2 and cols >= 2:
        print(f"      霍夫检测: {h_line_count} 水平线, {v_line_count} 垂直线")
        return (rows, cols)

    print(f"      无法自动检测宫格")
    return None

def detect_outer_borders(img, max_check=50):
    """
    检测图片四边的黑/白边（外边框）
    返回: (top, bottom, left, right) - 需要裁剪的像素数
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    borders = [0, 0, 0, 0]  # top, bottom, left, right
    
    # 检测上边缘
    for i in range(min(max_check, h)):
        row = gray[i, :]
        # 如果整行都是黑或白（或接近），认为是边框
        if np.all(row < 50) or np.all(row > 200):
            borders[0] = i + 1
        else:
            break
    
    # 检测下边缘
    for i in range(min(max_check, h)):
        row = gray[h - 1 - i, :]
        if np.all(row < 50) or np.all(row > 200):
            borders[1] = i + 1
        else:
            break
    
    # 检测左边缘
    for i in range(min(max_check, w)):
        col = gray[:, i]
        if np.all(col < 50) or np.all(col > 200):
            borders[2] = i + 1
        else:
            break
    
    # 检测右边缘
    for i in range(min(max_check, w)):
        col = gray[:, w - 1 - i]
        if np.all(col < 50) or np.all(col > 200):
            borders[3] = i + 1
        else:
            break
    
    return tuple(borders)

def find_grid_lines(img, rows, cols, threshold=30):
    """
    自动检测网格线位置
    返回精确的内部网格线坐标
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # 使用Sobel检测边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 投影
    h_proj = np.mean(np.abs(sobel_y), axis=1)
    v_proj = np.mean(np.abs(sobel_x), axis=0)
    
    # 平滑
    h_smooth = np.convolve(h_proj, np.ones(10)/10, mode='same')
    v_smooth = np.convolve(v_proj, np.ones(10)/10, mode='same')
    
    # 检测水平网格线
    horizontal_lines = []
    expected_y_positions = [int(h * i / rows) for i in range(1, rows)]
    
    for expected_y in expected_y_positions:
        search_range = 20
        y_start = max(0, expected_y - search_range)
        y_end = min(h, expected_y + search_range)
        
        best_y = expected_y
        max_val = 0
        
        for y in range(y_start, y_end):
            if y < len(h_smooth):
                val = h_smooth[y]
                if val > max_val:
                    max_val = val
                    best_y = y
        
        horizontal_lines.append(best_y)
    
    # 检测垂直网格线
    vertical_lines = []
    expected_x_positions = [int(w * i / cols) for i in range(1, cols)]
    
    for expected_x in expected_x_positions:
        search_range = 20
        x_start = max(0, expected_x - search_range)
        x_end = min(w, expected_x + search_range)
        
        best_x = expected_x
        max_val = 0
        
        for x in range(x_start, x_end):
            if x < len(v_smooth):
                val = v_smooth[x]
                if val > max_val:
                    max_val = val
                    best_x = x
        
        vertical_lines.append(best_x)
    
    return horizontal_lines, vertical_lines

def split_grid_clean(image_path, output_folder, rows, cols, auto_detect=True):
    """
    清洁切割图片，确保：
    1. 去除外边框
    2. 不切割到内部网格线
    3. 所有输出图片尺寸一致
    4. 无重影、无叠加
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        # 尝试用numpy方式读取
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] 无法读取图片 {image_path}")
            return
    
    h, w = img.shape[:2]
    original_size = (w, h)
    print(f"原始尺寸: {w}x{h}")
    
    # 2. 去除外边框
    if auto_detect:
        borders = detect_outer_borders(img)
        if sum(borders) > 0:
            print(f"检测到外边框: 上{borders[0]} 下{borders[1]} 左{borders[2]} 右{borders[3]} 像素")
            img = img[borders[0]:h-borders[1], borders[2]:w-borders[3]]
            h, w = img.shape[:2]
            print(f"去除外边框后尺寸: {w}x{h}")
    
    # 3. 查找网格线位置
    print(f"正在分析 {rows}x{cols} 网格...")
    h_lines, v_lines = find_grid_lines(img, rows, cols)
    
    # 4. 计算每个格子的精确边界（在网格线之间）
    # 水平分割点（包括上下边缘）
    h_positions = [0] + h_lines + [h]
    # 垂直分割点（包括左右边缘）
    v_positions = [0] + v_lines + [w]
    
    # 计算每个格子的范围
    # 加大margin，确保去除网格线的白边
    margin = 60  # 大幅增加margin确保去除所有网格线残留

    rows_info = []
    for i in range(rows):
        # 对内部网格线应用margin，边界也应用margin以去除外边框
        y1 = h_positions[i] + margin
        y2 = h_positions[i+1] - margin
        rows_info.append((max(0, y1), max(0, y2)))

    cols_info = []
    for j in range(cols):
        # 对内部网格线应用margin，边界也应用margin以去除外边框
        x1 = v_positions[j] + margin
        x2 = v_positions[j+1] - margin
        cols_info.append((max(0, x1), max(0, x2)))
    
    # 5. 计算统一的目标尺寸
    cell_heights = [y2 - y1 for y1, y2 in rows_info]
    cell_widths = [x2 - x1 for x1, x2 in cols_info]
    
    target_h = min(cell_heights)
    target_w = min(cell_widths)
    
    print(f"检测到的网格线位置: 水平{h_lines}, 垂直{v_lines}")
    print(f"每个格子目标尺寸: {target_w}x{target_h}")
    
    # 6. 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 7. 切割并保存
    count = 0
    for i, (y1_orig, y2_orig) in enumerate(rows_info):
        for j, (x1_orig, x2_orig) in enumerate(cols_info):
            count += 1

            # 直接使用去除margin后的边界切割（不使用统一的target尺寸）
            # 这样可以确保每个格子都按照其自身的margin边界切割
            x1 = max(0, min(x1_orig, w))
            y1 = max(0, min(y1_orig, h))
            x2 = min(x2_orig, w)
            y2 = min(y2_orig, h)

            # 确保切割区域有效
            if x2 <= x1 or y2 <= y1:
                print(f"  警告: 格子({i},{j})的切割区域无效，跳过")
                continue

            # 切割
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            
            # 额外裁剪：去除可能的边缘白边（每个方向裁剪2-4像素）
            crop_h, crop_w = crop.shape[:2]
            if crop_h > 20 and crop_w > 20:
                # 检查边缘亮度，如果偏高则裁剪
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # 顶部
                top_mean = np.mean(crop_gray[0:5, :])
                if top_mean > 200:
                    crop = crop[5:, :, :]
                
                # 底部
                bottom_mean = np.mean(crop_gray[-5:, :])
                if bottom_mean > 200:
                    crop = crop[:-5, :, :]
                
                # 左侧
                left_mean = np.mean(crop_gray[:, 0:5])
                if left_mean > 200:
                    crop = crop[:, 5:, :]
                
                # 右侧
                right_mean = np.mean(crop_gray[:, -5:])
                if right_mean > 200:
                    crop = crop[:, :-5, :]
            
            # 保存为高质量JPEG
            filename = os.path.join(output_folder, f"clip_{count:03d}.jpg")
            cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tofile(filename)
    
    print(f"[DONE] 成功切割 {count} 张图片")
    print(f"每张图片尺寸: {target_w}x{target_h}")
    print(f"输出目录: {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description='宫格图片清洁切割工具 - 去除边框、无重影、尺寸一致',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 自动检测宫格并切割（推荐）
  python grid_splitter.py -i "*.png"
  
  # 25宫格 (5x5)
  python grid_splitter.py -r 5 -c 5 -i "*.png"
  
  # 9宫格 (3x3)
  python grid_splitter.py -r 3 -c 3 -i "9宫格.png"
  
  # 60宫格 (6x10)
  python grid_splitter.py -r 6 -c 10 -i "*.png"
  
  # 60宫格 (10x6)
  python grid_splitter.py -r 10 -c 6 -i "*.png"
  
  # 禁用自动边框检测
  python grid_splitter.py -r 5 -c 5 -i "*.png" --no-auto
        '''
    )
    
    parser.add_argument('--rows', '-r', type=int, default=None,
                        help='行数 (如: 5 表示5行, 不指定则自动检测)')
    parser.add_argument('--cols', '-c', type=int, default=None,
                        help='列数 (如: 5 表示5列, 不指定则自动检测)')
    parser.add_argument('--auto-grid', action='store_true',
                        help='强制自动检测宫格 (即使指定了行列数)')
    parser.add_argument('--no-auto', action='store_true',
                        help='禁用自动边框检测 (默认启用)')
    parser.add_argument('--input', '-i', type=str, default='*.png',
                        help='输入图片模式 (默认*.png)')
    parser.add_argument('--output', '-o', type=str, default='output_result',
                        help='输出文件夹前缀')
    
    args = parser.parse_args()
    
    # 查找图片文件
    image_files = []
    if args.input == '*.png':
        image_files = glob.glob('*.png') + glob.glob('*.jpg') + glob.glob('*.jpeg')
    else:
        image_files = glob.glob(args.input)
    
    if not image_files:
        print(f"[ERROR] 未找到图片文件: {args.input}")
        print("提示: 确保图片在当前目录")
        return
    
    # 检查是否需要自动检测行列数
    need_auto_detect = args.auto_grid or (args.rows is None or args.cols is None)
    
    if need_auto_detect:
        print("[INFO] 未指定行列数或启用了自动检测，将自动分析宫格...")
    
    print(f"找到 {len(image_files)} 个图片文件")
    if args.rows and args.cols and not args.auto_grid:
        print(f"网格设置: {args.rows}行 x {args.cols}列 = {args.rows * args.cols}张")
    print(f"自动去边框: {'禁用' if args.no_auto else '启用'}")
    print("=" * 50)
    
    for img_path in image_files:
        print(f"\n正在处理: {img_path}")
        print("-" * 50)
        
        # 自动检测行列数
        rows = args.rows
        cols = args.cols
        
        if need_auto_detect:
            print("[INFO] 正在自动检测宫格行列数...")
            detected = auto_detect_grid_dimensions(img_path)
            if detected:
                rows, cols = detected
                print(f"[INFO] 自动检测到: {rows}行 x {cols}列 = {rows * cols}宫格")
            else:
                if rows is None or cols is None:
                    print("[ERROR] 无法自动检测宫格行列数，请手动指定 -r 和 -c 参数")
                    continue
                else:
                    print(f"[WARN] 自动检测失败，使用手动指定: {rows}行 x {cols}列")
        
        base_name = Path(img_path).stem
        output_folder = f"{args.output}/{base_name}"
        
        split_grid_clean(
            image_path=img_path,
            output_folder=output_folder,
            rows=rows,
            cols=cols,
            auto_detect=not args.no_auto
        )

if __name__ == "__main__":
    import glob
    main()
