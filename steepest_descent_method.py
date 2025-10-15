import sympy as sp
import numpy as np

f_expr = sp.sympify(input("f(x, y) = "))
print("Function f(x, y)= ", f_expr)

x, y = sp.symbols('x y')
vars = sp.Matrix([x, y])
grad_f = sp.Matrix([sp.diff(f_expr, var) for var in vars])
print("Gradient of f:", grad_f)
hessian_f = grad_f.jacobian(vars)
print("Hessian of f:", hessian_f)
input_str = input("x0, y0 = ")
x0, y0 = map(float, input_str.split(','))    
print("Initial guess:", (x0, y0))

f = sp.lambdify((x, y), f_expr, "numpy")
grad = sp.lambdify((x, y), grad_f, "numpy")

tol = float(input("Tolerance = "))
print("Threshold: ", tol , "\n","Iterations will stop when ||grad f|| < ", tol)
max_iter = 1000


point = np.array([x0, y0], dtype=float)

for k in range(max_iter):
    g = np.array(grad(point[0], point[1]), dtype=float).flatten()
    grad_norm = np.linalg.norm(g)

    if grad_norm < tol:
        print(f"\nConverged at iteration {k}")
        break

    # Convert point to symbolic for line search
    g_sym = sp.Matrix([sp.N(val) for val in g])
    alpha = sp.Symbol('alpha', real=True)

    # symbolic expression for f(x - alpha * grad)
    new_x = point[0] - alpha * g_sym[0]
    new_y = point[1] - alpha * g_sym[1]
    f_alpha = f_expr.subs({x: new_x, y: new_y})

    # derivative with respect to alpha, set to zero
    df_dalpha = sp.diff(f_alpha, alpha)
    alpha_candidates = sp.solve(df_dalpha, alpha)

    # choose a valid real positive alpha
    alpha_vals = [a.evalf() for a in alpha_candidates if a.is_real and a > 0]
    alpha_k = float(alpha_vals[0]) if alpha_vals else 0.01

    # update
    point = point - alpha_k * g
    print(f"Iter {k+1}: x = [{point[0]:.3f}, {point[1]:.3f}], α = {alpha_k:.3f}, ||grad|| = {grad_norm:.4f}")


print(f"\nMinimum point ≈ [{point[0]:.3f}, {point[1]:.3f}], f(x, y) = {f(point[0], point[1]):.3f}")