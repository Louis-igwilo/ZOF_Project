from flask import Flask, render_template, request, redirect, url_for
import sympy as sp
from typing import Callable
import math

app = Flask(__name__)

# Reuse the same solver implementations (import from a module would be better;
# keeping duplication limited here for an easy single-file app).
def parse_function(expr: str):
    x = sp.symbols('x')
    sym_expr = sp.sympify(expr)
    f = sp.lambdify(x, sym_expr, 'math')
    return f, sym_expr, x

def bisection_method(f, a, b, max_iter, tol):
    rows=[]
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    for i in range(1, max_iter+1):
        c=(a+b)/2.0
        fc=f(c)
        error=abs(b-a)/2.0
        rows.append((i, a, b, c, fc, error))
        if abs(fc)<tol or error<tol:
            return c, abs(fc), i, rows
        if fa*fc < 0:
            b=c; fb=fc
        else:
            a=c; fa=fc
    return c, abs(fc), max_iter, rows

def regula_falsi(f, a, b, max_iter, tol):
    rows=[]
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    c=a
    for i in range(1, max_iter+1):
        c = (a*fb - b*fa)/(fb-fa)
        fc = f(c)
        error = abs(fc)
        rows.append((i,a,b,c,fc,error))
        if abs(fc)<tol:
            return c, abs(fc), i, rows
        if fa*fc < 0:
            b=c; fb=fc
        else:
            a=c; fa=fc
    return c, abs(fc), max_iter, rows

def secant_method(f, x0, x1, max_iter, tol):
    rows=[]
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Zero denominator in Secant method.")
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        err = abs(x2 - x1)
        rows.append((i, x0, x1, x2, f(x2), err))
        if abs(f(x2)) < tol or err < tol:
            return x2, abs(f(x2)), i, rows
        x0, x1 = x1, x2
    return x2, abs(f(x2)), max_iter, rows

def newton_raphson(f_expr, f, x_sym, x0, max_iter, tol):
    df_expr = sp.diff(f_expr, x_sym)
    df = sp.lambdify(x_sym, df_expr, 'math')
    rows=[]
    x=x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Zero derivative.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        rows.append((i, x, fx, dfx, x_new, err))
        if abs(fx)<tol or err<tol:
            return x_new, abs(fx), i, rows
        x = x_new
    return x, abs(fx), max_iter, rows

def fixed_point_iteration(g, x0, max_iter, tol):
    rows=[]
    x=x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        rows.append((i, x, x_new, err))
        if err < tol:
            return x_new, err, i, rows
        x = x_new
    return x_new, err, max_iter, rows

def modified_secant(f, x0, delta, max_iter, tol):
    rows=[]
    x=x0
    for i in range(1, max_iter+1):
        f_x = f(x)
        denom = f(x + delta) - f_x
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in Modified Secant.")
        x_new = x - (delta * f_x)/denom
        err = abs(x_new - x)
        rows.append((i, x, f_x, x_new, err))
        if abs(f_x) < tol or err < tol:
            return x_new, abs(f_x), i, rows
        x = x_new
    return x_new, abs(f_x), max_iter, rows

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    table_headers = []
    table_rows = []
    error_msg = None
    if request.method == "POST":
        method = request.form.get("method")
        func_str = request.form.get("func").strip()
        max_iter = int(request.form.get("max_iter") or 50)
        tol = float(request.form.get("tol") or 1e-6)
        try:
            if method == "5":  # Fixed point needs g
                g_str = request.form.get("g_func","").strip()
                if not g_str:
                    raise ValueError("g(x) required for Fixed Point Iteration.")
                g, _, _ = parse_function(g_str)
            f, f_expr, x_sym = parse_function(func_str)
            if method == "1":
                a = float(request.form.get("a"))
                b = float(request.form.get("b"))
                root, final_err, iters, rows = bisection_method(f, a, b, max_iter, tol)
                table_headers = ["iter","a","b","c","f(c)","error"]
                table_rows = rows
            elif method == "2":
                a = float(request.form.get("a"))
                b = float(request.form.get("b"))
                root, final_err, iters, rows = regula_falsi(f, a, b, max_iter, tol)
                table_headers = ["iter","a","b","c","f(c)","error"]
                table_rows = rows
            elif method == "3":
                x0 = float(request.form.get("x0"))
                x1 = float(request.form.get("x1"))
                root, final_err, iters, rows = secant_method(f, x0, x1, max_iter, tol)
                table_headers = ["iter","x0","x1","x2","f(x2)","error"]
                table_rows = rows
            elif method == "4":
                x0 = float(request.form.get("x0"))
                root, final_err, iters, rows = newton_raphson(f_expr, f, x_sym, x0, max_iter, tol)
                table_headers = ["iter","x","f(x)","f'(x)","x_new","error"]
                table_rows = rows
            elif method == "5":
                x0 = float(request.form.get("x0"))
                g, _, _ = parse_function(g_str)
                root, final_err, iters, rows = fixed_point_iteration(g, x0, max_iter, tol)
                table_headers = ["iter","x","g(x)","error"]
                table_rows = rows
            elif method == "6":
                x0 = float(request.form.get("x0"))
                delta = float(request.form.get("delta") or 1e-3)
                root, final_err, iters, rows = modified_secant(f, x0, delta, max_iter, tol)
                table_headers = ["iter","x","f(x)","x_new","error"]
                table_rows = rows
            else:
                raise ValueError("Invalid method.")
            result = {"root": root, "final_err": final_err, "iters": iters}
        except Exception as e:
            error_msg = str(e)

    return render_template("index.html",
                           result=result,
                           headers=table_headers,
                           rows=table_rows,
                           error=error_msg)

if __name__ == "__main__":
    app.run(debug=True)
