from moviepy.editor import VideoFileClip

# 加载GIF文件
clip = VideoFileClip("input.gif")

# 将其转换为MP4并保存
clip.write_videofile("output.mp4", codec="libx264")
