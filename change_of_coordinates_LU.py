import numpy as np
from scipy.linalg import lu

# Monoatomic basis M = (a1, a2, a3)
mat = np.zeros((3, 3))
mat[:, 0] = [0.5, np.sqrt(3)/2, 0]
mat[:, 1] = [-0.5, np.sqrt(3)/2, 0]
mat[:, 2] = [0, 0, 1]
# Hexagonal Bravais lattice 
print("M matrix:"+"\n",mat)

# Checking if M is singular
if np.linalg.det(mat) <= np.min(mat):
    raise ValueError("Matrix is singular")

# LU decomposition with partial pivoting
# scipy's lu returns P, L, U directly
# Computational Complexity O(N^3)
p, l, u = lu(mat)

# Printing results 
print("P matrix:"+"\n",p)
print("L matrix:"+"\n",l)
print("U matrix:"+"\n",u)

### DEFECTS' STUDIES
# The same LU decomposition can be used for both the defects

# Defects in cartesian coordinates
b1 = np.array([2, 0, 2])
b2 = np.array([1, 2*np.sqrt(3), 3])

# z = P^T * b
z1 = p.T @ b1
z2 = p.T @ b2

# Manual (to avoiding inversions) forward substitution for L * y = z  and backward substitution for U * x = y 
# The different procedure is due to the fact the L is lower triangular L_{ij}=0 for j>i and U is upper triangular U_{ij}=0 for j<i 
# Partial Pivoting and Non-Singularity of M imply U[i,i]!=0 and L[i,i]!=0
def forward_inversion(matrix, vector):
    y = np.zeros_like(vector)
    for i in range(len(vector)):
        sum = 0
        for j in range(i):
            sum += matrix[i, j] * y[j]
        y[i] = (vector[i] - sum) / matrix[i, i]
    return y

def backward_inversion(matrix, vector):
    x = np.zeros_like(vector)
    for i in reversed(range(len(vector))):
        sum = 0
        for j in range(i+1,len(vector)):
            sum += matrix[i, j] * x[j]
        x[i] = (vector[i] - sum) / matrix[i, i]
    return x

# Solve step-by-step
# Computational Complexity O(N^2)
y1 = forward_inversion(l, z1)
# Computational Complexity O(N^2)
x1 = backward_inversion(u, y1)
# Total Computational Complexity per Defect is O(N^3)
y2 = forward_inversion(l, z2)
x2 = backward_inversion(u, y2)

print("First defect in cartesian coordinates:",b1)
print("First defect in crystal coordinates:",x1)
print("First defect in crystal coordinates with usual inversion:",np.linalg.inv(mat)@b1)
print("Second defect in cartesian coordinates:",b2)
print("Second defect in crystal coordinates:",x2)
print("Second defect in crystal coordinates with usual inversion:",np.linalg.inv(mat)@b2)

### BASIS TRANSFORMATION STUDIES
# Changing the order of the basis vectors 
tmat = np.zeros((3, 3))
tmat[:, 0] = mat[:, 2]
tmat[:, 1] = mat[:, 1]
tmat[:, 2] = mat[:, 0]
print("TM matrix:"+"\n",tmat)

# Taking a look at the transformation made above
# The transformation can be written in the basis of the crystal coordinates as
# T = ((001),(010),(100))
t = np.zeros((3,3))
t[0,2]=1
t[1,1]=1
t[2,0]=1
# new_crystal_coordinates = T @ old_crystal_coordinates = T @ M^{-1} @ old_cartesian_coordinates 
# So the new and old matrices are related by tM^{-1}=T @ M^{-1}, i.e. tM=M@T^{-1}
# The crystal coordinates in the new basis are give by T@x1
print("New crystal coordinates",np.linalg.inv(tmat)@b1)
print("Transforming the old crystal coordinates",t@x1)
print("Transforming the old matrix applied to the old crystal coordinates",t@np.linalg.inv(mat)@b1)

# Considering the LU-factorization tM=tP@tL@tU=P@L@U@T^{-1}
tp, tl, tu = lu(tmat)

# Printing results 
# Permutation matrix depend on the pivoting procedure
print("TP matrix:\n",tp,"\nP matrix: \n",p)
# tu and u have different structers
print("TL matrix:\n",tl,"\nL matrix: \n",l)
# tu and u have different structers
print("TU matrix:\n",tu,"\nU matrix: \n",u)
# LU-factorization is not invariant to column permutations

### SOME CONSIDERATIONS
## tL@tU=tP^{-1}@P@L@U@T^{-1}
#print("Comparing TL@TU and TP^{-1}@P@L@U@T^{-1} \n",tl@tu,np.linalg.inv(tp)@p@l@u@np.linalg.inv(t))
## imposing tL=tP^{-1}@P@L@S then tU=S^{-1}@U@T^{-1} (arbitrary factorization)
#print("S transformation linking TU and U: S@TU=U@T^{-1} \n",u@np.linalg.inv(t)@np.linalg.inv(tu))
#print("Comparing TL and TP^{-1}@P@L@S \n",tl-np.linalg.inv(tp)@p@l@u@np.linalg.inv(t)@np.linalg.inv(tu))





