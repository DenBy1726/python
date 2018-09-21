for x in range(10):
    print(x, end=' ')
print()

for x in range(10):
    if x is 1:
       continue
    if x > 5:
        break
    print(x,end=' ')
print()

x = 0
while x < 7:
    print (x,end=' ')
    x+=1
print()