import sympy as sp
import numpy as np

# --- Input residuals ---
n_res = int(input("Number of residuals (m): "))
print("Declare residuals r1, r2, ..., r_m as functions of x and y.")
print("Example for an r: 10 - (x + 2*y)  or  x**2 - y")

x, y = sp.symbols('x y')
res_exprs = []
for i in range(1, n_res+1):
    s = input(f"r{i}(x,y) = ")
    res_exprs.append(sp.sympify(s))

# Build residual vector and Jacobian
r_sym = sp.Matrix(res_exprs)                 # m x 1
J_sym = r_sym.jacobian([x, y])               # m x 2

print("\nResidual vector r(x):")
sp.pprint(r_sym)
print("\nJacobian J(x):")
sp.pprint(J_sym)

# Lambdify for numeric evaluations
r_func = sp.lambdify((x, y), r_sym, "numpy")     # returns m x 1 array
J_func = sp.lambdify((x, y), J_sym, "numpy")     # returns m x 2 array

# Initial guess and settings
x0, y0 = map(float, input("\nInitial guess (x0, y0): ").split(','))
tol = float(input("Tolerance (e.g. 1e-6): "))
max_iter = int(input("Max iterations (e.g. 100): "))

# Gauss-Newton loop
point = np.array([x0, y0], dtype=float)

for k in range(max_iter):
    r_val = np.array(r_func(point[0], point[1]), dtype=float).reshape(-1, 1)  # m x 1
    J_val = np.array(J_func(point[0], point[1]), dtype=float)               # m x 2

    # Compute gradient approx: g = J^T r  (2 x 1)
    g = (J_val.T @ r_val).reshape(-1)
    g_norm = np.linalg.norm(g)

    # Stopping criterion: gradient small
    if g_norm < tol:
        print(f"\nConverged (|J^T r| < tol) at iteration {k}")
        break

    # Normal equations matrix: J^T J (2 x 2)
    JTJ = J_val.T @ J_val

    # Try solve JTJ * delta = J^T r  for delta.
    # Newton step: x_new = x - delta
    try:
        # Prefer np.linalg.solve for stability if JTJ is well-conditioned
        delta = np.linalg.solve(JTJ, g)
    except np.linalg.LinAlgError:
        # If singular, use Tikhonov damping (Levenberg-like) or pseudo-inverse
        lam = 1e-6
        try:
            delta = np.linalg.solve(JTJ + lam * np.eye(JTJ.shape[0]), g)
            print(f"  Warning: JTJ singular — used damping λ={lam}")
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse
            delta = np.linalg.pinv(JTJ) @ g
            print("  Warning: JTJ singular — used pseudo-inverse fallback")

    # Update
    new_point = point - delta.reshape(2)

    step_norm = np.linalg.norm(new_point - point)
    res_norm = np.linalg.norm(r_val)

    print(f"Iter {k+1}: x = [{new_point[0]:.3f}, {new_point[1]:.3f}], ||J^T r|| = {g_norm:.4f}, ||r|| = {res_norm:.3f}, |step| = {step_norm:.3f}")

    # Convergence by step size too
    if step_norm < tol:
        point = new_point
        print(f"\nConverged (step size < tol) at iteration {k+1}")
        break

    point = new_point

else:
    print("\nReached max iterations without strict convergence.")

# Final output
r_final = np.array(r_func(point[0], point[1]), dtype=float).reshape(-1, 1)
J_final = np.array(J_func(point[0], point[1]), dtype=float)
cost = 0.5 * float((r_final.T @ r_final))
print(f"\nFinal estimate x* = [{point[0]:.3f}, {point[1]:.3f}]")
print(f"Final cost (1/2 sum r_i^2) = {cost:.4f}")
