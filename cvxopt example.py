# problem discription is in "problem discription.png"

import numpy as np
import cvxopt

n = 5 # dimenstion of x
A = np.random.randn(n, n)
b = np.random.randn(n)
l_i = -0.5 
u_i = 0.5

# Form P,q,G,h matrices.
# Write your code here:
AT = np.transpose(A)
P = 2*np.dot(AT,A)

bt = np.transpose(b)
Q = -2*np.dot(bt,A)
q = np.transpose(Q)

G = np.array([[-1,0,0,0,0],[0,-1,0,0,0],[0,0,-1,0,0],[0,0,0,-1,0],[0,0,0,0,-1],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
h = np.array([0.5,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5,-0.5])

P = cvxopt.matrix(P, tc='d')
q = cvxopt.matrix(q, tc='d')
G = cvxopt.matrix(G, tc='d')
h = cvxopt.matrix(h, tc='d')
sol = cvxopt.solvers.qp(P,q,G,h)

print('x*=', sol['x'])
print('p*=', sol['primal objective'])
