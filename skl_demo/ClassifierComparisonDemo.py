import common.ClassifierView as cv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

moons = datasets.make_moons(noise=0.3, random_state=0)
circles = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
datasets = [moons, circles]
knc = KNeighborsClassifier(3)
dtc = DecisionTreeClassifier(max_depth=5)
classifiers = [knc, dtc]

row = len(datasets)
col = len(classifiers) + 1

plt.figure(figsize=(4 * row, 2 * col))
for i in range(row):
    for j in range(col):
        print i, j
        ax = plt.subplot(row, col, i * col + j + 1)
        x, y = datasets[i]
        if j == 0:
            cv.viewCalssifierData(x, y, ax)
        else:
            cv.viewCalssifier(classifiers[j - 1], x, y, ax)
        print i, j

