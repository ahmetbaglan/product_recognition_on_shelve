def getHistogram(a, length):
    import collections
    c = collections.Counter(a)
    out = []
    sum = 0.0
    for i in c:
        sum+=c[i]

    for i in range(length):
        if i not in c.keys():
            out.append(0)
        else:
            out.append(c[i]/sum)

    return out

def includeDirs():
    print 'kkdlksnoin'
    pass

