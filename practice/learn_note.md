# Learning Note

shape = [(1,2),(4,4),(6,4),(3,6)]
n = len(shape)
total = 0
for i in range(n):
  a = shape[i]
  b = shape[(i+1)% n]
  total += distance(a,b)
