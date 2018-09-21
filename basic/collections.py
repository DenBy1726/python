import numpy as np

randoms = np.random.normal(25.0, 5.0, 10)
print (randoms)
randoms.sort()
print(randoms)

A = [1,2,3,4,5,6]
C = A[:4]
C[0] = "String"
print (A[:4])
print (A[3:])
print (A[-2:])

A.extend([7,8])
print (A)

A.append(9)
print (A)

y = [10,11,12]
listOfLists = [A,y]

B = [(1,1), (0.5,0.5)]
B.sort(key = lambda x: x[0])
print(B)

listNum = (1,2,3)
print(len(listNum))
listNum2 = (4,5,6)
print(listNum2[2])

listOfTuples = [listNum,listNum2]
tupleOfTuples = (listNum,listNum2)
print(listOfTuples)
print(tupleOfTuples)

(age, income) = "32,120000".split(',')
[age1,income1] = "32,120000".split(',')
print(age,income)
print(age1,income1)

# Like a map or hash table in other languages
captains = {}
captains["Enterprise"] = "Kirk"
captains["Enterprise D"] = "Picard"
captains["Deep Space Nine"] = "Sisko"
captains["Voyager"] = "Janeway"

print(captains["Voyager"])
print(captains.get("Enterprise"))
print(captains.get("Unknown"))
for ship in captains:
    print(ship + " : " + captains[ship])