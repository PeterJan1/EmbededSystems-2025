import numpy as np 

def matrix_addition(A,B):
    return np.add(A,B)

def matrix_multiplication(C,D):
    return C @ D

def matrix_transpose(E):
    return E.T

def matrix_determinant(F):
    return np.linalg.det(F)

def matrix_inverse(G):
    return np.linalg.inv(G)

if __name__ =='__main__':
    
    A = np.array([[1, 2],[3, 4]])
    B = np.array([[5, 6],[7, 8]])
    print(matrix_addition(A,B))
    
    C = np.array([[1, 2],[0, 4]])
    D = np.array([[6, 6],[7, 3]])
    print(matrix_multiplication(C,D))
    
    E = np.array([[1, 2, 3],[4, 5, 6]])
    
    F = np.array([[4, 7],[5, 2]])
    
    print(matrix_transpose(E))
    
    print(matrix_inverse(F))
    
    print(matrix_determinant(F))
    