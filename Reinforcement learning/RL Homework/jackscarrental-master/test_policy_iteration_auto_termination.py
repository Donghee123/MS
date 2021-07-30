from jackscarrental import PolicyIterationSolver
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np

solver = PolicyIterationSolver()

solver.policy_iteration()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = []
Y = []
Z = []

for i in range(21):
    for j in range(21):
        X.append(j)
        
for i in range(21):
    for j in range(21):
        Y.append(i)
        
for i in range(21):
    for j in range(21):
        Z.append(solver.value[i][j])
        
        
for m, zlow, zhigh in [('o', -50, -25)]:
    xs = X
    ys = Y
    zs = Z
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

plt.subplot(121)
CS = plt.contour(solver.policy, levels=range(-6, 6))
plt.clabel(CS)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.axis('equal')
plt.xticks(range(21))
plt.yticks(range(21))
plt.grid('on')

#plt.subplot(122)
#plt.pcolor(solver.value)
#plt.colorbar()
#plt.axes(projection='3d')
#plt.axis('auto')

#plt.show()

