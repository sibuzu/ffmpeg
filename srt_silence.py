import pandas as pd

def getTime(t):
    t = int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) + 0.001 * int(t[9:12])
    idx = df[df.fm>=t].index
    if len(idx) == 0:
        idx = len(df) - 1
    else:
        idx = idx[0] - 1
    row = df.loc[idx]
    if t >= row.to:
        t = t - row.cumdelta
    else:
        t = row.to - row.cumdelta
    return t


def timeToStr(t):
    s = int(t)
    ms = int((t - s) * 1000)
    h = s // 3600
    m = s // 60 % 60
    s = s % 60
    return "{:02}:{:02}:{:02},{:03}".format(h,m,s,ms)

def convertTime(x):
    t1 = getTime(x[:12])
    t2 = getTime(x[17:])
    x = "{} --> {}\n".format(timeToStr(t1), timeToStr(t2))
    return x

df = pd.read_csv("silence.txt", header=None)
df[2] = df[1] - df[0]
df[3] = df[2].cumsum()
df.columns = ['fm', 'to', 'delta', 'cumdelta']
print(df)

lines = open('a.srt', encoding='utf8').readlines()

with open('b.srt', 'w', encoding='utf8') as f:
    for x in lines:
        if "-->" in x:
            x = convertTime(x)
        f.write(x)    
