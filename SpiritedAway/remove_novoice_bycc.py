def getTime(t):
    t = int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) + 0.001 * int(t[9:12])
    return t

def getTimes(x):
    t1 = getTime(x[:12])
    t2 = getTime(x[17:])
    return t1, t2

def printTime(f, cc):
    t1 = cc[1] - cc[4]
    t2 = cc[2] - cc[4]
    f.write("{} --> {}\n".format(strTime(t1), strTime(t2)))

def printText(f, cc):
    for line in cc[5:]:
        f.write(line + "\n")
    f.write("\n")

def strTime(t):
    s = int(t)
    ms = int((t - s) * 1000)
    h = s // 3600
    m = s // 60 % 60
    s = s % 60
    return "{:02}:{:02}:{:02},{:03}".format(h,m,s,ms)

lines = open('SpiritedAway.srt', encoding='utf8').readlines()

cclist = []
cc = []

for xx in lines:
    x = xx.strip()
    if "-->" in x:
        t1, t2 = getTimes(x)
        cc = [x, t1, t2, 0, 0]
    elif x == "":
        if cc:
            cclist.append(cc)
        cc = []
    else:
        cc.append(x)

#for x in cclist:
#    print(x)

delta = cclist[0][1]
cclist[0][4] = round(delta, 3)
for i in range(len(cclist) - 1):
    ccx = cclist[i]
    ccy = cclist[i+1]
    t = round(ccy[1] - ccx[2], 3)
    if t > 1:
        ccy[3] = t
        delta += t
    ccy[4] = round(delta, 3)

with open('SpiritedAway-N.srt', 'w', encoding='utf8') as f:
    for i in range(len(cclist)):
        cc = cclist[i]
        f.write(str(i+1) + "\n")
        printTime(f, cc)
        printText(f, cc)

for cc in cclist:
    print(cc)

bw = []
t1 = cclist[0][1]
t2 = cclist[0][2]
n = len(cclist)
for i in range(n):
    cc = cclist[i]
    if i==n-1 or cclist[i+1][3]>0:
        t2 = cc[2]
        t2 = round(cc[2] - 0.001, 3)
        bw.append((t1, t2))
        if i<n-1:
            t1 = cclist[i+1][1]

# for x in bw:
#     print(x)

bws = "+".join(["between(t,{},{})".format(*x) for x in bw])
vfs = "select='{}',setpts=N/FRAME_RATE/TB".format(bws)
afs = "aselect='{}',asetpts=N/SR/TB".format(bws)
with open("vf.txt", "w") as f:
    f.write(vfs)
with open("af.txt", "w") as f:
    f.write(afs)
