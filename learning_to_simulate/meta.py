
import numpy as np
import md

def getDx(p):
    v = []
    for c in range (0,len(p)-1):
        v.append(p[c+1]-p[c])
    return v

def getDxVel(p):
    v = getDx(p[0])
    for i in range(1, len(p)):
        temp = getDx(p[i])
        v += temp
    return v

def getDxAcc(p):
    v = getDx(getDx(p[0]))
    for i in range(1, len(p)):
        temp = getDx(getDx(p[i]))
        v += temp
    return v

def getMean(l):
    r = []
    for i1 in range(0,3):
        t = []
        for i2 in range(0,len(l)):
            for i3 in range(0,len(l[i2])):
                t.append(l[i2][i3][i1])
        r.append(np.mean(t))
    return r

def getStd(l):
    r = []
    for i1 in range(0,3):
        t = []
        for i2 in range(0,len(l)):
            for i3 in range(0,len(l[i2])):
                t.append(l[i2][i3][i1])
        r.append(np.std(t))
    return r

def getStats(i):
    p = []
    t, pos = md.getDataFrames(1)
    p.append(pos)
    for c in range(2,i+1):
        x, y = md.getDataFrames(c)
        p.append(y)
    mean_vel = getMean(getDxVel(p))
    std_vel = getStd((getDxVel(p)))

    mean_acc = getMean(getDxAcc(p))
    std_acc = getStd(getDxAcc(p))
    return mean_vel, std_vel, mean_acc, std_acc



print(getStats(3))
















