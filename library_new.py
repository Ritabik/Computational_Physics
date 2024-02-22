import numpy as np
import matplotlib.pyplot as plt

# Fixed Point method to solve the root of the solution
def fixed_point(g, initial_guess, tolerance, max_iterations=100):
    x = initial_guess
    for iteration in range(max_iterations):
        x_next = g(x)
        if abs(x_next - x) < tolerance:
            return x_next
        x = x_next
    raise ValueError("Fixed-point iteration did not converge within the specified tolerance.")

# Simpson's Rule to evaluate integration
def simpson(f,a,b,n):
    h=(b-a)/n
    x=a 
    sum=0
    for i in range(n):
        sum+= (f(x)+4*f(x+h/2)+f(x+h))*h/6
        x+=h
    return sum

# Guassian quadrature method to evaluate the integration
def gaussian_quadrature(f, a, b):
    # Weights and nodes for 3-point Gaussian quadrature
    weights = np.array([5/9, 8/9, 5/9])
    nodes = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])

    # Change of interval from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w = 0.5 * (b - a) * weights

    # Evaluate the integrand and sum up
    r = np.sum(w * f(x))
    return r

# Range kutta4 method to solve ODE
def range_kutta4(f,x0,y0,xk,h):
    x=[x0]
    y=[y0]
    i=0
    while x[i]<xk:
        k1=h*f(x[i],y[i])
        k2=h*f(x[i]+h/2,y[i]+k1/2)
        k3=h*f(x[i]+h/2,y[i]+k2/2)
        k4=h*f(x[i]+h,y[i]+k3)
        y.append(y[i]+(k1+2*k2+2*k3+k4)/6)
        x.append(x[i]+h)
        i+=1
    return x,y

# Crack-Nicoleson solver to solve any PDE

def crank_nicolson_solver(L, T, Nx, Nt, alpha, initial_condition, boundary_conditions):
    # Discretization
    dx = L / (Nx - 1)
    dt = T / Nt

    x_values = np.linspace(0, L, Nx)
    t_values = np.linspace(0, T, Nt)

    # Initialize solution matrix
    u = np.zeros((Nx, Nt))

    # Set initial condition
    u[:, 0] = initial_condition(x_values)

    # Set boundary conditions
    u[0, :] = boundary_conditions['x_start'](t_values)
    u[-1, :] = boundary_conditions['x_end'](t_values)

    # Coefficients for the tridiagonal system
    a = -alpha / 2
    b = 1 + alpha
    c = -alpha / 2

    # Time-stepping loop
    for n in range(1, Nt):
        # Right-hand side of the system
        rhs = np.zeros(Nx)
        rhs[1:-1] = u[1:-1, n-1] + (alpha / 2) * (u[:-2, n-1] - 2 * u[1:-1, n-1] + u[2:, n-1])

        # Forward substitution (without external modules)
        for i in range(1, Nx):
            m = a / b
            b = b - m * c
            rhs[i] = rhs[i] - m * rhs[i-1]

        u[-1, n] = rhs[-1] / b

        for i in range(Nx-2, -1, -1):
            u[i, n] = (rhs[i] - c * u[i+1, n]) / b

    return x_values, t_values, u

def regula_falsi(f, a, b, tol=1e-6, max_iter=1000):
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)
        x_next = (a * fb - b * fa) / (fb - fa)
        
        if abs(f(x_next)) < tol:
            return x_next, i + 1  # Return the root and the number of iterations
        
        if f(a) * f(x_next) < 0:
            b = x_next
        else:
            a = x_next
    
    return None, max_iter  # Return None if the method does not converge within max_iter iterations

def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        x = x - fx / fpx
        
        if abs(fx) < tol:
            return x, i + 1  # Return the root and the number of iterations
    
    return None, max_iter  # Return None if the method does not converge within max_iter iterations

def Lagrange_interpol(zeta_h, zeta_l, yh, yl, y):    

    zeta = zeta_l + (zeta_h - zeta_l) * (y - yl)/(yh - yl)
    return zeta
def RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z0, xf, h):      

    x = [x0]
    y = [y0]
    z = [z0]
    N = int((xf-x0)/h)

    for i in range(N): 

        k1 = h * func_dydx(x[i], y[i], z[i])
        l1 = h * Func_d2ydx2(x[i], y[i], z[i])
        
        k2 = h * func_dydx(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        l2 = h * Func_d2ydx2(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        
        k3 = h * func_dydx(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        l3 = h * Func_d2ydx2(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        
        k4 = h * func_dydx(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * Func_d2ydx2(x[i] + h, y[i] + k3, z[i] + l3)
        
        x.append(x[i] + h)
        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        z.append(z[i] + (l1 + 2*l2 + 2*l3 + l4)/6)

    return x, y, z

def RKshooting_method(Func_d2ydx2, func_dydx, x0, y0, xf, yf, z1, z2, h, tol=1e-6):  
                   
    x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z1, xf, h)

    yn = y[-1]

    if abs(yn - yf) > tol:

        if yn < yf:

            zeta_l = z1
            yl = yn
            x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z2, xf, h)
            yn = y[-1]

            if yn > yf:

                zeta_h = z2
                yh = yn
                zeta = Lagrange_interpol(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, zeta, xf, h)
                return x, y
            
            else:

                print("Invalid bracketing.")

        elif yn > yf:

            zeta_h = z1
            yh = yn
            x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, z2, xf, h)
            yn = y[-1]

            if yn < yf:

                zeta_l = z2
                yl = yn
                zeta = Lagrange_interpol(zeta_h, zeta_l, yh, yl, yf)
                x, y, z = RK_shooting(Func_d2ydx2, func_dydx, x0, y0, zeta, xf, h)
                return x, y
            
            else:

                print("Invalid bracketig.")
    else:

        return x, y

def lu_decomposition_solve(A, b, n):
    L = [[0 for x in range(n)] for y in range(n)]
    U = [[0 for x in range(n)] for y in range(n)]

    for j in range(0, n):
        for i in range(0, n):
            sum = 0
            for k in range(0, i):
                sum += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - sum

        for i in range(0, n):
            sum = 0
            for k in range(0, j):
                sum += L[i][k] * U[k][j]
            L[i][j] = (1 / U[j][j]) * (A[i][j] - sum)

    for i in range(0, n):
        sum = 0
        for j in range(0, i):
            sum += L[i][j] * b[j]
        b[i] = b[i] - sum

    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i][j] * b[j]
        b[i] = (b[i] - sum) / U[i][i]

    return b


