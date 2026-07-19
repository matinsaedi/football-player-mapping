from moviepy.editor import *
import glob

files = glob.glob('*.h264')

clips = []

for i in files:
    clip = VideoFileClip(i)
    clips.append(clip)
    
final = concatenate_videoclips(clips)    