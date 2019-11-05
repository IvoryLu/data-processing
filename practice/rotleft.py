def rotLeft(a, d):
    for j in range(d):
        tmp = a[0]
        a.remove(tmp)
        a.append(tmp)
    return a
