import numpy as np
import matplotlib.pyplot as plt

def swap_rows(M, row_index_1, row_index_2):
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def transform_to_row_echelon_form(M):
    M = M.astype(float)  # Convert to floating point type
    num_rows, num_cols = M.shape
    for i in range(min(num_rows, num_cols)):  # Iterate over columns or rows up to the smaller of the two
        pivot_row = i
        while pivot_row < num_rows and np.isclose(M[pivot_row, i], 0, atol=1e-5):
            pivot_row += 1
        if pivot_row == num_rows:  # If there are no nonzero elements in the column, move to the next column
            continue
        M = swap_rows(M, i, pivot_row)  # Swap rows to move the nonzero element to the diagonal
        pivot_val = M[i, i]
        for j in range(i + 1, num_rows):  # Eliminate nonzero elements in the lower rows
            multiplier = M[j, i] / pivot_val
            M[j] -= multiplier * M[i]
    return M

def back_substitution(M):
    num_rows, num_cols = M.shape
    solutions = np.zeros(num_rows)

    for i in range(num_rows - 1, -1, -1):
        solutions[i] = M[i, -1] / M[i, i]
        for j in range(i - 1, -1, -1):
            M[j, -1] -= M[j, i] * solutions[i]
            M[j, i] = 0  # Zero out elements above the diagonal
    return solutions

def check_determinant(a):
    if np.linalg.det(a) == 0:
        raise ValueError("The determinant of matrix 'a' is zero. The system may have no unique solution.")

# Example matrix
a = np.array([[2, 1, -3],
              [1, -1, 2],
              [3, -2, 1]])

b = np.array([[4],
              [1],
              [2]])

# Check determinant
check_determinant(a)

# Combine matrices a and b
combined_matrix = np.hstack((a, b))

print("Initial combined matrix:")
print(combined_matrix)
print("----------")

# Transform to row echelon form
transformed_matrix = transform_to_row_echelon_form(combined_matrix.copy())  # Copying matrix a to avoid changing its original value
print("Matrix after transformation to row echelon form:")
print(transformed_matrix)
print("----------")

# Calculate solutions
solutions = back_substitution(transformed_matrix)
print("Solutions:")
print(solutions)

# Plot lines defined by the equations
x = np.linspace(-1000, 1000, 1000)
for i in range(a.shape[0]):
    label = f'{a[i, 0]}x + {a[i, 1]}y = {b[i, 0]}'
    plt.plot(x, (b[i, 0] - a[i, 0]*x) / a[i, 1], label=label)

# Plot solutions
plt.scatter(solutions[0], solutions[1], color='red', label='Solution')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Equations and Solutions')
plt.legend()
plt.grid(True)
plt.show()
