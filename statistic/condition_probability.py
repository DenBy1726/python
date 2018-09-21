from numpy import random

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = float(ageDecade) / 100.0
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1

print()

# P(E|F)  40% вероятность что любой 40 летний покупатель что то купил
PEF = float(purchases[40]) / float(totals[40])
print('P(purchase | 40s): ' + str(PEF))

# P(F)  16,6% вероятность что покупателю 40 лет
PF = float(totals[40]) / 100000.0
print("P(40's): " +  str(PF))

# P(E) 45% вероятность что покупатель (любого возраста) что то купил
PE = float(totalPurchases) / 100000.0
print("P(Purchase):" + str(PE))

# P(E)P(F) 7,5% вероятность что выбранному покупатель 40 лет и этот самый покупатель что то купил
print("P(40's)P(Purchase)" + str(PE * PF))

print("P(40's, Purchase)" + str(float(purchases[40]) / 100000.0))
print((purchases[40] / 100000.0) / PF)