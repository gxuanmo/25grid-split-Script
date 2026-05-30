# Grid Splitter Project

## Overview
宫格图片切割工具，将拼图（如 3x3、5x5、6x2、10x6 等）自动切割为独立小图。当前主版本为 `grid_splitter_v4.py`。

## Tech Stack
- Python 3, OpenCV (`cv2`), NumPy
- Windows 兼容（中文路径用 `np.fromfile`，glob 去重处理大小写）

## Key Commands
```bash
# 自动检测网格并切割所有图片
python grid_splitter_v4.py --auto-grid

# 指定网格切割单张图片
python grid_splitter_v4.py -r 5 -c 5 -i "图片.png"

# 安装依赖
pip install -r requirements.txt
```

## Architecture
- 核心算法：Sobel 边缘投影 + 峰值检测定位网格线，Hough 变换兜底
- 自动检测流程：候选配置评分（信噪比） → min_found_per_dim tiebreaker → 选最佳 rows x cols
- 切割流程：外边框检测去除 → 网格线定位 → margin 跳过网格线 → 裁剪残留边 → Lanczos 统一尺寸

## Directory Structure
- `test_images/` — 测试输入图片（宫格原图）
- `screenshots/` — UI 截图、debug 产物，非项目代码
- `web/` — FastAPI Web UI（`app.py` + `static/` + `uploads/`）
- `output_result_v4/` — CLI 切割输出

## Conventions
- 输出目录：`output_result_v4/{文件名}/clip_001.jpg ...`
- 输出质量：JPEG 100%
- v1-v4 历史版本 `.py` 文件保留在项目根目录
- 测试图片放 `test_images/`，截图放 `screenshots/`，根目录只放代码和文档
