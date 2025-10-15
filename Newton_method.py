import sympy as sp
import numpy as np

# Input function
f_expr = sp.sympify(input("f(x, y) = "))
print("Function f(x, y)= ", f_expr)

x, y = sp.symbols('x y')
vars = sp.Matrix([x, y])
grad_f = sp.Matrix([sp.diff(f_expr, var) for var in vars])
hessian_f = grad_f.jacobian(vars)
print("Gradient of f:", grad_f)
print("Hessian of f:", hessian_f)

# Initial point
input_str = input("x0, y0 = ")
x0, y0 = map(float, input_str.split(','))
print("Initial guess:", (x0, y0))

# Create numeric functions
f = sp.lambdify((x, y), f_expr, "numpy")
grad = sp.lambdify((x, y), grad_f, "numpy")
hessian = sp.lambdify((x, y), hessian_f, "numpy")

tol = float(input("Tolerance = "))
print("Threshold:", tol, "\nIterations stop when ||grad f|| < ", tol, " or if xk+1 ≈ xk ")
max_iter = 100

# Newton’s iteration
point = np.array([x0, y0], dtype=float)

for k in range(max_iter):
    g = np.array(grad(point[0], point[1]), dtype=float).flatten()
    H = np.array(hessian(point[0], point[1]), dtype=float)
    grad_norm = np.linalg.norm(g)
    
    if grad_norm < tol:
        print(f"\nConverged at iteration {k}")
        break

    try:
        
        p = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        print("Hessian is singular — cannot invert. Stopping.")
        break

    # Newton update rule
    point = point - p
    print(f"Iter {k+1}: x = [{point[0]:.3f}, {point[1]:.3f}], ||grad|| = {grad_norm:.4f}")

print(f"\nMinimum point ≈ [{point[0]:.3f}, {point[1]:.3f}], f(x, y) = {f(point[0], point[1]):.4f}")
