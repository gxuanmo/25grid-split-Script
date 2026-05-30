# 宫格图片切割工具

## 功能
- 支持 25 宫格、9 宫格、60 宫格等任意 m x n 网格
- 自动检测网格维度（无需手动指定行列数）
- 自动检测并去除外边框（黑/白边）
- 智能识别内部网格线，确保切割不带到相邻图片内容
- 切割后自动裁剪网格线残留
- 所有输出图片尺寸完全一致

## 支持的网格配置
2x2, 2x3, 3x2, 3x3, 3x4, 4x3, 3x5, 5x3, 3x6, 6x3,
4x4, 4x5, 5x4, 4x6, 6x4, 5x5, 5x6, 6x5, 5x12, 12x5,
6x6, 6x10, 10x6, 7x7, 2x4, 4x2, 2x6, 6x2, 2x8, 8x2, 2x10, 10x2

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### Web UI（推荐）
```bash
# 启动后浏览器打开 http://127.0.0.1:8765
python -m uvicorn web.app:app --host 127.0.0.1 --port 8765

# 或双击 run_web.bat
```
上传图片 → 自动检测网格 → 预览网格线 → 可手动调整行列数 → 切割 → 单张预览或打包下载 ZIP。

### CLI：自动检测网格
```bash
# 批量处理 test_images/ 下所有图片
python grid_splitter_v4.py --auto-grid -i "test_images/*"

# 自动检测单张图片
python grid_splitter_v4.py --auto-grid -i "test_images/九宫格.png"
```

### CLI：手动指定网格
```bash
# 9 宫格（3x3）
python grid_splitter_v4.py -r 3 -c 3 -i "test_images/九宫格.png"

# 25 宫格（5x5）
python grid_splitter_v4.py -r 5 -c 5 -i "test_images/01.png"

# 60 宫格（10x6）
python grid_splitter_v4.py -r 10 -c 6 -i "test_images/测试2.png"
```

### CLI：高级选项
```bash
# 禁用自动边框检测
python grid_splitter_v4.py -r 5 -c 5 -i "test_images/01.png" --no-auto

# 指定输出目录
python grid_splitter_v4.py -r 5 -c 5 -i "test_images/01.png" -o "我的输出"
```

## 输出说明
- 输出文件夹：`output_result_v4/原文件名/`
- 文件格式：`clip_001.jpg`, `clip_002.jpg` ...
- 图片质量：100% 高质量 JPEG

## 算法简介
1. **外边框检测**：逐行/列扫描，去除纯黑/纯白边框
2. **网格线定位**：Sobel 算子计算水平/垂直投影，在预期位置附近搜索峰值
3. **自动维度检测**：对候选配置逐一评分（信噪比），以 min_found_per_dim 为 tiebreaker 选出最佳配置；Hough 变换作为兜底
4. **切割**：根据网格线位置 + margin 提取每个格子，裁剪残留边，Lanczos 统一输出尺寸

## 常见问题

**Q: 为什么输出的图片比预期的少？**
A: 脚本会跳过无效格子（宽或高为 0），并在日志中输出 WARN 提示。确保使用完整的宫格图。

**Q: 切割后的图片还有边框怎么办？**
A: 脚本默认会检测并去除边框。如果还有残留，可以尝试调整网格参数，或使用 `--no-auto` 禁用自动检测后手动处理。

**Q: 支持哪些图片格式？**
A: PNG、JPG、JPEG

**Q: 自动检测选错了网格怎么办？**
A: 用 `-r` 和 `-c` 手动指定正确的行列数即可。
