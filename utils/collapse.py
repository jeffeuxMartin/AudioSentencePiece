from itertools import groupby    
SPACE = chr(32)
while True:
    try:
        line = input().split(SPACE)
        print(SPACE.join([k for k, g in groupby(line)]))
    except EOFError:
        break
