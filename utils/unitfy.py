from itertools import groupby    
SPACE = chr(32)
while True:
    try:
        line = input().split(SPACE)
        print(SPACE.join([f'uni_{int(ele):04d}' for ele in line]))
    except EOFError:
        break
