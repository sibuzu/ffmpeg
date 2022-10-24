import subprocess as sp
import sys
import numpy

FFMPEG_BIN = "ffmpeg.exe"

print('asplit.py <src.mp3> <silence duration in seconds> <threshold amplitude 0.0 .. 1.0>')

src = sys.argv[1]
dur = float(sys.argv[2])
thr = int(float(sys.argv[3]) * 65535)

f = open('{}-out.bat'.format(src), 'w')

tmprate = 22050
len2 = dur * tmprate
buflen = int(len2 * 2)
#  t * rate * 16 bits

oarr = numpy.arange(1, dtype='int16')
# just a dummy array for the first chunk

command = [ FFMPEG_BIN,
        '-i', src,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(tmprate), # ouput sampling rate
        '-ac', '1', # '1' for mono
        '-']        # - output to stdout

pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

tf = True
pos = 0
opos = 0
part = 0

while tf :

    raw = pipe.stdout.read(buflen)
    if raw == '' :
        tf = False
        break

    arr = numpy.fromstring(raw, dtype = "int16")
 
    rng = numpy.concatenate([oarr, arr])
    if len(rng) == 0:
        break

    mx = numpy.amax(rng)
    if mx <= thr :
        # the peak in this range is less than the threshold value
        trng = (rng <= thr) * 1
        # effectively a pass filter with all samples <= thr set to 0 and > thr set to 1
        sm = numpy.sum(trng)
        # i.e. simply (naively) check how many 1's there were
        if sm >= len2 :
            part += 1
            apos = pos + dur * 0.0
            print(mx, thr, sm, len2, apos)
            f.write('ffmpeg -i "{}" -ss {} -to {} -c copy -y "{}-p{:04d}.mp3"\r\n'.format(src, opos, apos, src, part))
            opos = apos

    pos += dur

    oarr = arr

part += 1    
f.write('ffmpeg -i "{}" -ss {} -to {} -c copy -y "{}-{:04d}.mp3"\r\n'.format(src, opos, pos, src, part))
f.close()

print("done")
