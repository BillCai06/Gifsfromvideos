# 视频转 GIF 工具

本项目是一个 Python 脚本，用于从指定文件夹中的视频文件中抽取帧并生成 GIF 动图。脚本支持常见视频格式（如 `.mp4`, `.mov`, `.avi`, `.wmv`, `.m4v`, `.rmvb`），并自动处理视频修复（调用 ffmpeg）以及帧抽取、大小调整和 GIF 优化等操作。

## 特性

- **视频处理**：自动检测指定文件夹中的视频文件并进行处理。
- **帧抽取**：从视频中按照指定百分比抽取帧，并生成固定帧率的 GIF 动图。
- **视频修复**：若视频无法正常打开，尝试使用 ffmpeg 修复视频编码问题。
- **GIF 优化**：生成的 GIF 动图会限制最大宽度，并根据目标帧率进行优化。
- **进度显示**：使用 `tqdm` 显示处理进度，方便观察运行状态。

## 先决条件

- **Python 3.x**：确保已安装 Python 3 环境。
- **ffmpeg**：如果需要修复视频文件，请确保系统中已安装 [ffmpeg](https://ffmpeg.org/)，并将其添加到系统 PATH 中。

## 安装依赖

首先，建议使用虚拟环境，然后运行以下命令安装项目所需依赖：

```bash
pip install -r requirements.txt


安装ffmpeg示例 (Windows)：
choco install ffmpeg  # Windows (需管理员权限)
```

## 使用方法

1. 将脚本文件和你要转换的所有视频文件放在同一文件夹内。
2. 根据需要可以调整脚本顶部的参数配置，例如：

   - \`MAX\_GIF\_SIZE\_MB\`：生成的 GIF 动图最大允许大小（单位：MB）。
   - \`MAX\_WIDTH\`：GIF 动图的最大宽度（单位：像素）。
   - \`GIF\_FPS\`：生成 GIF 的帧率。
3. 运行脚本：

\`\`\`bash

python gifscirpt.py

\`\`\`

脚本会自动遍历文件夹内的所有视频文件，抽取指定帧，并生成对应的 GIF 文件。如果目标 GIF 已存在，则会跳过该文件。

## 注意事项

- 若视频文件存在播放或编码问题，脚本会自动调用 ffmpeg 进行修复，但这需要系统正确安装 ffmpeg。
- 生成的 GIF 动图大小可能超过设定的目标大小，请根据需要调整参数或后续压缩 GIF。

## 许可

本项目采用 [MIT License](LICENSE) 许可，欢迎使用和贡献代码！
