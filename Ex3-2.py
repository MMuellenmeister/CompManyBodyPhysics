import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def hamilton_matrix(J, N):
    """
    Generates the hamiltonian Matrix for a list of arbitrary couplings.
    @param N: Number of sites to model.
    @param J: Array of dimensions [3,2**N] with N= number of sites. Entries in the list represent the couplings.
        The first index describes the coupling dimension (0=x, 1=y, 3=z).
        The second index describes the (two) sites which are coupled as a binary number i.e. "2-3" coupling = 5 = 0b110
    @return: hamilton matrix with dimensions [2**N, 2**N]. States are ordered in increasing binary representation
        with physical states having the *same* ordering as the binary values.
        The first state is thus positioned at the rightmost position, the last one at the leftmost position.
    """
    if not J.shape[1] == 2**N:
        raise Exception("J must have the second dimension equal to 2**N")
    arrsize = [J.shape[0], J.shape[1], J.shape[1]]
    H = np.zeros(arrsize, dtype=float)  # calculate each dimension separately, sum them in the end
    #  generate list of all two-particle interactions in binary format, i.e for (N=3)1-2 '011', for (N=9)3-9 '100000100'
    masks = []
    for j in range(1, N):
        for i in range(0, j):
            masks.append(np.power(2, i, dtype=int) + np.power(2, j, dtype=int))
    kMax = H.shape[1]
    for a in range(0, kMax):
        for m in masks:
            b = a ^ m  # state with applicable raising & lowering operators applied, unapplicables are 0 either way
            H[0, a, b] = J[0, m] / 4  # x-coupling. No case distinction necessary.
            eq = a & m  # get only the effected spins
            g = 0
            for i in range(H.shape[1]):
                g += (eq >> i) & 1  # go through the effected spins and sum outcome
            #  case distinction for parallel (g=1 or g=2) or antiparallel (g=1) spins.
            if g == 1:
                H[1, a, b] += J[1, m] / 4  # y-coupling for antiparallel spins
                H[2, a, a] -= J[2, m] / 4  # z-coupling for antiparallel spins
            else:
                H[1, a, b] -= J[1, m] / 4  # y-coupling for parallel spins
                H[2, a, a] += J[2, m] / 4  # z-coupling for parallel spins
            "breakpoint"
    # return the sum over all dimensions
    ges = H[0]
    for i in range(1, H.shape[0]):
        ges += H[i]
    return ges

J = 1
N = 3

#1b
J1 = np.ones([3, 2 ** N], dtype=float)
J1[1] = np.zeros([2 ** N])
J1[0] = np.zeros([2 ** N])

#1c
J2 = np.ones([3, 2 ** N], dtype=float)

#1d
J3 = np.zeros([3, 2 ** N])
J3[0, 0b011] = 1
J3[1, 0b110] = 1
J3[2, 0b101] = 1

Jtest = np.ones([3, 2**5])

J1 *= J
J2 *= J
J3 *= J
print("J1 (Ising model): ")
print(hamilton_matrix(J1, N))
print("\nJ2 (isotropic Heisenberg model): ")
print(hamilton_matrix(J2, N))
print("\nJ3 (Exercise 1d): ")
print(hamilton_matrix(J3, N))
print("\nJtest for N>3: ")
H=hamilton_matrix(Jtest, 5)
print(H)
plt.imshow(H)
plt.show()
