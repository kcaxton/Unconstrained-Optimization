import sympy as sp
import numpy as np

# Step 1: User input
f_expr = sp.sympify(input("Enter quadratic function f(x, y) = a + b^T x + 0.5 * x^T A x : "))
x, y = sp.symbols('x y')
print("\nFunction f(x, y) =", f_expr)

# Step 2: Compute gradient and Hessian
grad_f = sp.Matrix([sp.diff(f_expr, x), sp.diff(f_expr, y)])
hessian_f = grad_f.jacobian([x, y])

# Step 3: Extract A, b, and a
A = np.array(hessian_f, dtype=float)
b = np.array([grad_f.subs({x:0, y:0})[0], grad_f.subs({x:0, y:0})[1]], dtype=float)
a = float(f_expr.subs({x:0, y:0}))

print("\nExtracted components:")
print("A =\n", A)
print("b =", b)
print("a =", a)

# Step 4: Numeric functions
f = sp.lambdify((x, y), f_expr, "numpy")

# Step 5: Initialize
x0, y0 = map(float, input("\nInitial guess (x0, y0): ").split(','))
tol = float(input("Tolerance = "))

xk = np.array([x0, y0], dtype=float)
rk = -(b + A @ xk)   # Initial residual)
pk = rk.copy()

print("\nStarting Conjugate Gradient iterations...\n")

# Step 6: CG iterations
for k in range(100):
    r_norm = np.linalg.norm(rk)
    if r_norm < tol:
        print(f"Converged at iteration {k}")
        break

    Apk = A @ pk
    alpha_k = (rk.T @ rk) / (pk.T @ Apk)
    xk = xk + alpha_k * pk
    rk_next = rk - alpha_k * Apk
    beta_k = (rk_next.T @ rk_next) / (rk.T @ rk)
    pk = rk_next + beta_k * pk
    rk = rk_next

    print(f"Iter {k+1}: x = [{xk[0]:.3f}, {xk[1]:.3f}], α = {alpha_k:.3f}, ||r|| = {r_norm:.4f}")

print(f"\nMinimum point ≈ [{xk[0]:.3f}, {xk[1]:.3f}], f(x, y) = {f(xk[0], xk[1]):.4f}")
