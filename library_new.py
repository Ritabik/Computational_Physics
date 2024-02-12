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



