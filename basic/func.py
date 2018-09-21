def Square(x):
    return x*x

def doSomething(func,param):
    return func(param)

print(Square(2))
print(doSomething(Square,3))
print(doSomething(lambda x: (x*x*x)/2,3))

