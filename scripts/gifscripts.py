import cv2
import os
import subprocess
from PIL import Image
import numpy as np
from tqdm import tqdm

# 参数设置
MAX_GIF_SIZE_MB = 5
MAX_WIDTH = 320
SEGMENT_DURATION_SEC = 1
GIF_FPS = 10

folder_path = os.path.dirname(os.path.realpath(__file__))

video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov', '.avi', '.wmv', '.m4v', '.rmvb'))]

def fix_video_ffmpeg(video_path):
    temp_path = video_path + "_temp.mp4"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        temp_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    os.replace(temp_path, video_path)

for filename in tqdm(video_files, desc="处理视频文件"):
    video_path = os.path.join(folder_path, filename)
    output_filename = os.path.splitext(filename)[0] + '.gif'
    output_path = os.path.join(folder_path, output_filename)

    if os.path.exists(output_path):
        tqdm.write(f"GIF 文件 {output_filename} 已存在，跳过。")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"{filename} 打开失败，尝试用ffmpeg修复...")
        fix_video_ffmpeg(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            tqdm.write(f"视频 {filename} 修复后仍无法打开，跳过。")
            continue
        else:
            tqdm.write(f"{filename} 已修复并重新打开。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        tqdm.write(f"无法获取视频信息: {filename}，跳过。")
        cap.release()
        continue

    frame_indices = []
    percents = [0.1 * i for i in range(1, 10)]
    for perc in percents:
        start_frame = int(total_frames * perc)
        if start_frame >= total_frames:
            continue
        segment_frames = min(int(fps), total_frames - start_frame)
        if segment_frames <= 0:
            continue
        if segment_frames >= GIF_FPS:
            relative_indices = np.linspace(0, segment_frames - 1, num=GIF_FPS, dtype=int)
        else:
            relative_indices = np.arange(segment_frames)
        frames_indices_segment = [start_frame + idx for idx in relative_indices]
        frames_to_capture.extend(frames_indices)

    frames_to_capture = sorted(set(frames_to_capture))

    frames = []
    current_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_capture[0])

    for frame_idx in tqdm(frames_to_capture, desc=f"抽取 {filename} 帧", leave=False):
        if frame_idx != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success or frame is None:
            continue

        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

        current_frame = frame_idx + 1

    cap.release()

    if not frames:
        tqdm.write(f"视频 {filename} 未抽取到任何帧，跳过。")
        continue

    frame_duration = int(1000 / GIF_FPS)
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )
    except Exception as e:
        tqdm.write(f"保存 GIF 时出错（{filename}）：{e}")
        continue

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if file_size_mb > MAX_GIF_SIZE_MB:
        tqdm.write(f"警告：{filename} GIF 大小为 {file_size_mb:.2f} MB，超过目标 {MAX_GIF_SIZE_MB} MB。")
    else:
        tqdm.write(f"成功生成 {filename} 的 GIF，大小 {file_size_mb:.2f} MB。")
