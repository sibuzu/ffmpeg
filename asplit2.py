import subprocess as sp
import sys
import numpy

FFMPEG_BIN = "ffmpeg.exe"

print('asplit.py <src.mp3> <silence duration in seconds> <threshold amplitude 0.0 .. 1.0>')

src = sys.argv[1]
dur = float(sys.argv[2])
thr = int(float(sys.argv[3]) * 65535)

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
silence_list = [(0,0)]
last_silence = None

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
            if pos-opos==1:
                # it is silence
                if last_silence:
                    last_silence = (last_silence[0], pos)
                else:
                    last_silence = (opos, pos)
            else:
                if last_silence:
                    silence_list.append(last_silence)
                    last_silence = None

            print(mx, thr, sm, len2, opos, pos)
            opos = pos

    pos += 1

    oarr = arr

silence_list.append((pos, pos))

bwlist = []
for i in range(0, len(silence_list)-1):
    fm = silence_list[i][1] * dur
    to = silence_list[i+1][0] * dur
    bwlist.append((fm, to))

sw = "".join(["{},{}\n".format(x[0]*dur, x[1]*dur) for x in silence_list])
with open("silence.txt", "w") as f:
    f.write(sw)

bw = "+".join(["between(t,{},{})".format(*x) for x in bwlist])
vf = "select='{}',setpts=N/FRAME_RATE/TB".format(bw)
af = "aselect='{}',asetpts=N/SR/TB".format(bw)
with open("vf.txt", "w") as f:
    f.write(vf)
with open("af.txt", "w") as f:
    f.write(af)

print("done")
