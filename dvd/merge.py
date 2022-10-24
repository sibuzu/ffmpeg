from PIL import Image
import subprocess
import argparse
import numpy as np
import re
import os
import sys
import glob
import shutil

TEMP_CLIP = "TEMPX/CLIP"
files = sorted(glob.glob(TEMP_CLIP + "/*.mp4"))
files = files[1::2]

lstfile = "flist.txt"
with open(lstfile, "w") as f:
    for fname in files:
        f.write("file '{}'\n".format(fname))

OUTPUT_FILE = "out2.mp4"
command = "ffmpeg -y -f concat -safe 0 -i {} -c copy {} -hide_banner".format(lstfile, OUTPUT_FILE)
print(command)
subprocess.call(command, shell=True)
