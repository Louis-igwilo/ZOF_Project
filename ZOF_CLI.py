#!/usr/bin/env python3
"""
ZOF_CLI.py
Zero Of Functions (ZOF) Solver - CLI

Supported methods:
1. Bisection
2. Regula Falsi (False Position)
3. Secant
4. Newton-Raphson
5. Fixed Point Iteration (requires g(x))
6. Modified Secant

Uses sympy to parse functions safely and compute derivatives where needed.
"""
import sys
import math
from typing import Callable, Tuple, List, Any, Optional
import sympy as sp

# --- Utility helpers ---
def parse_function(expr: str, var_symbol='x') -> Tuple[Callable[[float], float], sp.Expr, sp.Symbol]:
    """Return (callable, sympy_expr, sympy_symbol)"""
    x = sp.symbols(var_symbol)
    try:
        sym_expr = sp.sympify(expr)
    except Exception as e:
        raise ValueError(f"Can't parse expression: {e}")
    f = sp.lambdify(x, sym_expr, 'math')
    return f, sym_expr, x

def safe_float(s: str) -> float:
    return float(s.strip())

def print_iteration_table(headers: List[str], rows: List[List[Any]]):
    col_widths = [max(len(str(cell)) for cell in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(col_widths[i]) for i,h in enumerate(headers))
    sep = "-+-".join("-"*w for w in col_widths)
    print(header_line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))

# --- Methods ---
def bisection_method(f: Callable[[float], float], a: float, b: float, max_iter:int, tol:float):
    rows = []
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    for i in range(1, max_iter+1):
        c = (a + b)/2.0
        fc = f(c)
        error = abs(b - a)/2.0
        rows.append([i, a, b, c, fc, error])
        if abs(fc) < tol or error < tol:
            return c, abs(fc), i, rows
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(fc), max_iter, rows

def regula_falsi(f: Callable[[float], float], a: float, b: float, max_iter:int, tol:float):
    rows = []
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    c = a
    for i in range(1, max_iter+1):
        c = (a*fb - b*fa)/(fb - fa)
        fc = f(c)
        error = abs(fc)
        rows.append([i, a, b, c, fc, error])
        if abs(fc) < tol:
            return c, abs(fc), i, rows
        if fa*fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c, abs(fc), max_iter, rows

def secant_method(f: Callable[[float], float], x0: float, x1: float, max_iter:int, tol:float):
    rows = []
    for i in range(1, max_iter+1):
        f0 = f(x0)
        f1 = f(x1)
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Zero denominator in Secant method.")
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        err = abs(x2 - x1)
        rows.append([i, x0, x1, x2, f(x2), err])
        if abs(f(x2)) < tol or err < tol:
            return x2, abs(f(x2)), i, rows
        x0, x1 = x1, x2
    return x2, abs(f(x2)), max_iter, rows

def newton_raphson(f_expr: sp.Expr, f: Callable[[float], float], x_sym: sp.Symbol, x0: float, max_iter:int, tol:float):
    df_expr = sp.diff(f_expr, x_sym)
    df = sp.lambdify(x_sym, df_expr, 'math')
    rows = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative at x = {:.6g}".format(x))
        x_new = x - fx/dfx
        err = abs(x_new - x)
        rows.append([i, x, fx, dfx, x_new, err])
        if abs(fx) < tol or err < tol:
            return x_new, abs(fx), i, rows
        x = x_new
    return x, abs(fx), max_iter, rows

def fixed_point_iteration(g: Callable[[float], float], x0: float, max_iter:int, tol:float):
    rows = []
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        rows.append([i, x, x_new, err])
        if err < tol:
            return x_new, err, i, rows
        x = x_new
    return x_new, err, max_iter, rows

def modified_secant(f: Callable[[float], float], x0: float, delta: float, max_iter:int, tol:float):
    rows = []
    x = x0
    for i in range(1, max_iter+1):
        f_x = f(x)
        denom = f(x + delta) - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in Modified Secant.")
        x_new = x - (delta * f_x) / denom
        err = abs(x_new - x)
        rows.append([i, x, f_x, x_new, err])
        if abs(f_x) < tol or err < tol:
            return x_new, abs(f_x), i, rows
        x = x_new
    return x_new, abs(f_x), max_iter, rows

# --- CLI interface ---
def run_cli():
    print("Zero Of Functions (ZOF) Solver - CLI")
    print("Enter the function f(x). Example: x**3 - x - 2")
    func_in = input("f(x) = ").strip()
    try:
        f, f_expr, x_sym = parse_function(func_in)
    except Exception as e:
        print("Error parsing function:", e)
        sys.exit(1)

    print("\nChoose method:")
    print("1) Bisection")
    print("2) Regula Falsi (False Position)")
    print("3) Secant")
    print("4) Newton-Raphson")
    print("5) Fixed Point Iteration (requires g(x))")
    print("6) Modified Secant")
    choice = input("Method (1-6): ").strip()

    max_iter = int(input("Maximum iterations (e.g., 50): ").strip() or "50")
    tol = float(input("Tolerance (e.g., 1e-6): ").strip() or "1e-6")

    try:
        if choice == '1':
            a = float(input("Left endpoint a: ").strip())
            b = float(input("Right endpoint b: ").strip())
            root, final_err, iters, rows = bisection_method(f, a, b, max_iter, tol)
            print("\nBisection iterations:")
            print_iteration_table(["iter","a","b","c","f(c)","error"], rows)
        elif choice == '2':
            a = float(input("Left endpoint a: ").strip())
            b = float(input("Right endpoint b: ").strip())
            root, final_err, iters, rows = regula_falsi(f, a, b, max_iter, tol)
            print("\nRegula Falsi iterations:")
            print_iteration_table(["iter","a","b","c","f(c)","error"], rows)
        elif choice == '3':
            x0 = float(input("x0: ").strip())
            x1 = float(input("x1: ").strip())
            root, final_err, iters, rows = secant_method(f, x0, x1, max_iter, tol)
            print("\nSecant iterations:")
            print_iteration_table(["iter","x0","x1","x2","f(x2)","error"], rows)
        elif choice == '4':
            x0 = float(input("Initial guess x0: ").strip())
            root, final_err, iters, rows = newton_raphson(f_expr, f, x_sym, x0, max_iter, tol)
            print("\nNewton-Raphson iterations:")
            print_iteration_table(["iter","x","f(x)","f'(x)","x_new","error"], rows)
        elif choice == '5':
            print("For Fixed Point Iteration you must provide g(x) such that x = g(x).")
            g_in = input("g(x) = ").strip()
            try:
                g, _, _ = parse_function(g_in)
            except Exception as e:
                print("Error parsing g(x):", e)
                sys.exit(1)
            x0 = float(input("Initial guess x0: ").strip())
            root, final_err, iters, rows = fixed_point_iteration(g, x0, max_iter, tol)
            print("\nFixed Point iterations:")
            print_iteration_table(["iter","x","g(x)","error"], rows)
        elif choice == '6':
            x0 = float(input("Initial guess x0: ").strip())
            delta = float(input("Delta (perturbation), e.g., 1e-3: ").strip() or "1e-3")
            root, final_err, iters, rows = modified_secant(f, x0, delta, max_iter, tol)
            print("\nModified Secant iterations:")
            print_iteration_table(["iter","x","f(x)","x_new","error"], rows)
        else:
            print("Invalid choice.")
            return
    except Exception as exc:
        print("Computation error:", exc)
        return

    print("\nFinal estimated root: {:.12g}".format(root))
    print("Final function value / error estimate: {:.12g}".format(final_err))
    print("Iterations used:", iters)

if __name__ == "__main__":
    run_cli()
