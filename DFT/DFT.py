# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import cmath
import math
import numpy as np

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        N = 15
        val = 0
        fwd_transform = np.zeros((15, 15), dtype=complex)
        for u in range(N):
            for v in range(N):
                for i in range(N):
                    for j in range(N):
                        x = -1 * cmath.sqrt(-1) * (2 * math.pi * (u * i + v * j)) / N
                        val = val + (matrix[i, j] * cmath.exp(x))

                fwd_transform[u, v] = val
                val = 0

        return fwd_transform

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        N = 15
        val = 0
        inv_transform = np.zeros((15, 15), dtype=complex)
        for u in range(N):
            for v in range(N):
                for i in range(N):
                    for j in range(N):
                        x = cmath.sqrt(-1) * (2 * math.pi * (u * i + v * j)) / N
                        val = val + (1 / (N * N)) * (matrix[i, j] * cmath.exp(x))
                #print(val)
                inv_transform[u, v] = val
                val = 0
                #print(u, v)

        return inv_transform

    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        N = 15
        val = 0
        cosine_transform = np.zeros((15, 15))
        for u in range(N):
            for v in range(N):
                for i in range(N):
                    for j in range(N):
                        x = (2 * math.pi * (u * i + v * j)) / 15
                        val = val + (matrix[i, j] * math.cos(x))
                # print(val)
                cosine_transform[u, v] = val
                val = 0
                # print(u, v)

        return cosine_transform

    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        N = 15
        mag = np.zeros((15, 15), dtype=complex)
        for u in range(N):
            for v in range(N):
                val = cmath.sqrt((matrix[u, v].real * matrix[u, v].real) + (matrix[u, v].imag * matrix[u, v].imag))
                mag[u, v] = val

        return mag
