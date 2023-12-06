# Spectral, Runge-Kutta and Magnus Solvers

"""
To solve 

x'(t) = A(t) x(t)

"""

import numpy as np
from numpy.lib.scimath import sqrt as csqrt

import time
import sympy as sym
from scipy import special, linalg

# choose numerical integrator
from scipy.integrate import quadrature as quad
from scipy.integrate import complex_ode
from scipy import optimize

from matplotlib import rc
rc('text', usetex=True)

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import gridspec

from sys import exit as sysexit

T_start = time.time()


############# Set up Equations / A matrices ########################

"""
Define a function for the A matrix and the true solution
"""
def A_from_w2(w2, num_vs_sym):
	def f(t):
		if num_vs_sym:
			# numpy matrix
			M = np.matrix([[0, 1], [-w2(t), 0]])
		elif num_vs_sym == False:
			# sympy matrix
			M = sym.Matrix([[0, 1], [-w2(t), 0]])
		return M
	return f

def Simplify(Expr):
	#E1 = sym.powsimp(Expr, deep=True, force=True)
	E1 = sym.simplify(Expr)
	E2 = sym.nsimplify(E1)
	return E2

ts0 = sym.Symbol('ts0', real=True)
ts = sym.Symbol('ts', real=True)
ts1 = sym.Symbol('ts1', real=True)

array2mat = [{'ImmutableDenseMatrix': np.matrix}, 'numpy']
array2mat_c = [{'ImmutableDenseMatrix': np.matrix}, {'sqrt': csqrt}, 'numpy']

# --- Airy equation stuff --- #
Airy = {}
Airy["name"] = "Airy"
Airy["t_start"] = 1
Airy["t_stop"] = 35
Ai0, Aip0, Bi0, Bip0 = special.airy(-Airy["t_start"])
Airy["x0"] = np.array([Ai0, -Aip0])
Airy["ylim"] = (-0.5, 1.0)

def w2_Airy(t):
	return t
Airy["w2"] = w2_Airy
Airy["A_num"] = A_from_w2(w2_Airy, True)
Airy["A_sym"] = A_from_w2(w2_Airy, False)(ts)

def Airy_sol(t):
	Ai0, Aip0, Bi0, Bip0 = special.airy(-Airy["t_start"])
	M = np.linalg.inv(np.matrix([[Ai0, Bi0], [-Aip0, -Bip0]]))
	ab = M @ Airy["x0"].reshape(2, 1)	
	Ai, Aip, Bi, Bip = special.airy(-t)
	a = ab[0, 0]
	b = ab[1, 0]
	x_true = a*Ai + b*Bi
	dxdt_true = -(a*Aip + b*Bip)
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x
	
Airy["true_sol"] = Airy_sol # function

Airy["title"] = "Airy equation"
# ---------------------------- #
# --- Burst equation stuff --- #
burst = {}
burst["n"] = 40
#burst["n"] = 80
burst["name"] = "Burst_n=" + str(burst["n"]) #+ "_centre"
burst["ylim"] = (-0.4, 0.4)

def w2_burst(t):
	n = burst["n"]
	w = (n**2 - 1)/(1 + t**2)**2
	return w
burst["w2"] = w2_burst
burst["A_num"] = A_from_w2(w2_burst, True)
burst["A_sym"] = A_from_w2(w2_burst, False)(ts)

def burst_soln(t, n):
	if n % 2 == 0:
		x = (np.sqrt(1 + t**2)/n)*((-1)**(n/2))*np.sin(n*np.arctan(t))
	elif (n+1) % 2 == 0:
		x = (np.sqrt(1 + t**2)/n)*((-1)**((n-1)/2))*np.cos(n*np.arctan(t))
	return x

def dburst_soln(t, n):
	if n % 2 == 0:
		x = (1/(np.sqrt(1 + t**2)*n) )*((-1)**(n/2))*(t*np.sin(n*np.arctan(t)) + n*np.cos(n*np.arctan(t)))
	elif (n+1) % 2 == 0:
		x = (1/(np.sqrt(1 + t**2)*n))*((-1)**((n-1)/2))*(t*np.cos(n*np.arctan(t)) - n*np.sin(n*np.arctan(t)))
	return x

def Burst_sol(t):
	x_true = burst_soln(t, burst["n"])
	dxdt_true = dburst_soln(t, burst["n"])
	x = np.hstack((x_true.reshape(t.size, 1),dxdt_true.reshape(t.size, 1))) 
	return x

burst["t_start"] = -10
burst["t_stop"] = +10
#burst["t_start"] = -2
#burst["t_stop"] = +2
burst["x0"] = Burst_sol(np.array([burst["t_start"]]))
burst["true_sol"] = Burst_sol
burst["title"] = "Burst equation (n = " + str(burst["n"]) + ")"
# ---------------------------- #
# -- Triplet equation stuff -- #

def f(t):
	return (t*t+t)
	
def f_num(t):
	return (t*t+t)
	
def g(t):
	return (sym.exp(-t**2/64))
	
def g_num(t):
	return (np.exp(-t**2/64))

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

F = f(ts)
G = g(ts)
dF = sym.diff(F, ts)
ddF = sym.diff(dF, ts)
dG = sym.diff(G, ts)
ddG = sym.diff(dG, ts)

#print("F = ", F)
#print("G = ", G)
#print("dF = ", dF)
#print("dG = ", dG)
#print("ddG = ", ddG)
#print("ddF = ", ddF)

A_sym_triplet = sym.Matrix([[dG/G, -dF, 0], [+dF, 0, 1], [+(dF*dG)/G, ddG/G, 0]])
#A = sym.Matrix([[0, 1, 0], [(ddG/G - dF**2), 0, ddF - 2*dG/G], [dF, 0, dG/G]])
	
A_num_triplet = sym.lambdify((ts), A_sym_triplet, modules=array2mat)
	
triplet = {}
triplet["name"] = "triplet_v2"
triplet["title"] = "3D equation, with $x = e^{-t^2/4}\\cos(t^2)$"
triplet["n"] = 10
triplet["sigma"] = 4
triplet["f"] = f
triplet["g"] = g
triplet["A_sym"] = Simplify(A_sym_triplet)
triplet["A_num"] = A_num_triplet

def triplet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(g_num(t)*np.sin(f_num(t)))
	z = np.array(dg(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1),z.reshape(t.size, 1))) 
	return x_

triplet["w2"] = 0	# dummy value
triplet["true_sol"] = triplet_sol
triplet["ylim"] = (-1.0, 1.0)
triplet["t_start"] = 1
triplet["t_stop"] = 20
triplet["x0"] = triplet_sol(np.array([triplet["t_start"]]))

# ---------------------------- #
# -- doublet equation stuff -- #
def f(t):
	return (t**2)
	
def f_num(t):
	return (t**2)
	
def g(t):
	return (1/t)
	
def g_num(t):
	return (1/t)

g_sym = g(ts)
f_sym = f(ts)
dg = sym.lambdify((ts), sym.diff(g_sym, ts), modules=array2mat)
df = sym.lambdify((ts), sym.diff(f_sym, ts), modules=array2mat)

F = f(ts)
G = g(ts)
dF = sym.diff(F, ts)
ddF = sym.diff(dF, ts)
dG = sym.diff(G, ts)
ddG = sym.diff(dG, ts)
"""
print("F = ", F)
print("G = ", G)
print("dF = ", dF)
print("dG = ", dG)
print("ddF = ", ddF)
print("ddG = ", ddG)
"""
A_sym_doublet = sym.Matrix([[0, 1], [ddG/G - dF**2 - 2*(dG/G)**2 - (dG*ddF)/(G*dF), ddF/dF + 2*dG/G]])
	
A_num_doublet = sym.lambdify((ts), A_sym_doublet, modules=array2mat)
	
doublet = {}
doublet["name"] = "doublet"
doublet["title"] = "2D eq. $x = t^{-1}\\cos(t^2)$"
doublet["n"] = 10
doublet["sigma"] = 4
doublet["f"] = f
doublet["g"] = g
doublet["A_sym"] = Simplify(A_sym_doublet)
doublet["A_num"] = A_num_doublet

def doublet_sol(t):
	x = np.array(g_num(t)*np.cos(f_num(t)))
	y = np.array(dg(t)*np.cos(f_num(t)) - df(t)*g_num(t)*np.sin(f_num(t)))
	x_ = np.hstack((x.reshape(t.size, 1),y.reshape(t.size, 1))) 
	return x_

doublet["w2"] = 0	# dummy value
doublet["true_sol"] = doublet_sol
doublet["ylim"] = (-1.0, 1.0)
doublet["t_start"] = 2
doublet["t_stop"] = 15
doublet["x0"] = doublet_sol(np.array([doublet["t_start"]]))
# ---------------------------- #


################### Choose equation #########################

Eq = triplet


Index = 0	# index of x_i variable to plot

############# define some functions ##########

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA

def Com(A, B):
	return (A*B - B*A)
	
#------> set up alpha functions

def alpha_D(t0, t, A, order=4):
	# compute the alpha coefficients using the autograd
	# derivative 
	# 4 Function evaluations for order 4
	h = t - t0
	dt = 0.000001*h
	a_1 = h*A(t0 + 0.5*h)
	dA = eg(A, dt)
	a_2 = (h**2)*dA(t0 + 0.5*h)
	if order == 4:
		return (a_1, a_2)
	elif order == 6:
		ddA = eg(dA, dt)
		a_3 = (1/2)*(h**3)*ddA(t0 + 0.5*h)
		return (a_1, a_2, a_3)
	
def alpha_GL(t0, t, A, order=4):
	# compute the alpha coefficients using the Gauss-Legendre quadrature
	# rule
	# 2 Function evaluations for order 4
	h = t - t0
	if order == 4:
		A1 = A(t0 + (0.5 - np.sqrt(3)/6)*h)
		A2 = A(t0 + (0.5 + np.sqrt(3)/6)*h)
		a_1 = 0.5*h*(A1 + A2)
		a_2 = (np.sqrt(3)/12)*h*(A2 - A1)
		return (a_1, a_2)
	elif order == 6:
		A1 = A(t0 + (0.5 - 0.1*np.sqrt(15))*h)
		A2 = A(t0 + 0.5*h)
		A3 = A(t0 + (0.5 + 0.1*np.sqrt(15))*h)
		a_1 = h*A2
		a_2 = (np.sqrt(15)/3)*h*(A3 - A1)
		a_3 = (10/3)*h*(A3 - 2*A2 + A1)
		return (a_1, a_2, a_3)
		
def alpha_SNC(t0, t, A, order=4):
	# compute the alpha coefficients using the Simpson and Newton–
	# Cotes quadrature rules using equidistant A(t) points
	# 3 Function evaluations for order 4
	h = t - t0
	if order == 4:
		A1 = A(t0)
		A2 = A(t0 + 0.5*h)
		A3 = A(t0 + h)
		a_1 = (h/6)*(A1 + 4*A2 + A3)
		a_2 = h*(A3 - A1)
		return (a_1, a_2)
	elif order == 6:
		A1 = A(t0)
		A2 = A(t0 + 0.25*h)
		A3 = A(t0 + 0.5*h)
		A4 = A(t0 + 0.75*h)
		A5 = A(t0 + h)
		a_1 = (1/60)*(-7*(A1 + A5) + 28*(A2 + A4) + 18*A3)
		a_2 = (1/15)*(7*(A5 - A1) + 16*(A4 - A2))
		a_3 = (1/3)*(7*(A1 + A5) - 4*(A2 + A4) - 6*A3)
		return (a_1, a_2, a_3)

def alpha_SNC2(t0, t, A1, A2, A3, order=4):
	# compute the alpha coefficients using the Simpson and Newton–
	# Cotes quadrature rules using equidistant A(t) points for fully numerical magnus methods
	h = t - t0
	if order == 4:
		#A1 = A(t0)
		#A2 = A(t0 + 0.5*h)
		#A3 = A(t0 + h)
		a_1 = (h/6)*(A1 + 4*A2 + A3)
		a_2 = h*(A3 - A1)
		return (a_1, a_2)
#------> set up quadrature integragrators  

scipy_quad_maxiter=200
	
def scipy_c_quad(f, t0, t, ARGS=()):
	# integrate complex valued function f(t) from t0 to t using scipy.integrate.quadrature
	MAXITER=scipy_quad_maxiter
	def f_real(x, *args):
		f_ = f(x, *args)
		return np.real(f_)
	def f_imag(x, *args):
		f_ = f(x, *args)
		return np.imag(f_)
	Int_real = quad(f_real, t0, t, args=ARGS, maxiter=MAXITER, vec_func=False)[0]
	Int_imag = 1j*quad(f_imag, t0, t, args=ARGS, maxiter=MAXITER, vec_func=False)[0]
	Int_ = Int_real + Int_imag
	return Int_
	
def scipy_M_quad(A, t0, t, ARGS=()):
	# integrate complex matrix valued function f(t) from t0 to t using scipy.integrate.quadrature
	MAXITER=scipy_quad_maxiter
	ni, nj = A(1).shape
	def f_real(x, I, J, *args):
		f_ = A(x, *args)[I, J]
		return np.real(f_)
	def f_imag(x, I, J, *args):
		f_ = A(x, *args)[I, J]
		return np.imag(f_)
	Int_M = np.zeros((ni, nj))*(1.0+0.j)
	for I in range(ni):
		for J in range(nj):
			IJ_ARGS = (I, J) + ARGS
			Int_M[I, J] = quad(f_real, t0, t, args=IJ_ARGS, maxiter=MAXITER, vec_func=False)[0] + 1j*quad(f_imag, t0, t, args=IJ_ARGS, maxiter=MAXITER, vec_func=False)[0]
	return Int_M
	
# set quadrature integrator (for the moment just set one)
c_quad = scipy_c_quad
	
#----> other functions

def Omega_num(A, alpha, order):
	# function to return an Omega(t0, t) function
	def Omega(t0, t):
		# the Magnus expansion Omega truncated to the appropriate order in h
		if order == 4:
			a_1, a_2 = alpha(t0, t, A, 4)
			Om = a_1 - (1/12)*Com(a_1, a_2)
			return Om
		elif order == 6:
			a_1, a_2, a_3 = alpha(t0, t, A, 6)
			C1 = Com(a_1, a_2)
			C2 = -(1/60)*Com(a_1, 2*a_3 + C1)
			Om = a_1 + (1/12)*a_3 + (1/240)*Com(-20*a_1-a_3+C1, a_2+C2)
			return Om
	return Omega


def Omega_num2(A1,A2,A3,t0,t, alpha, order):
	# function to return an Omega(t0, t) function
	def Omega(t0, t):
		# the Magnus expansion Omega truncated to the appropriate order in h
		if order == 4:
			a_1, a_2 = alpha(t0, t, A1, A2, A3, 4)
			Om = a_1 - (1/12)*Com(a_1, a_2)
			return Om
		elif order == 6:
			a_1, a_2, a_3 = alpha(t0, t, A, 6)
			C1 = Com(a_1, a_2)
			C2 = -(1/60)*Com(a_1, 2*a_3 + C1)
			Om = a_1 + (1/12)*a_3 + (1/240)*Com(-20*a_1-a_3+C1, a_2+C2)
			return Om
	return Omega

def ferr(x_0, x_l):
	# a function to evaluate an error between step estimates
	# returns a vector
	err = np.abs(x_0 - x_l)
	return err
	
def log_minor_ticks(ax):
	locmin = ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10)) 
	ax.yaxis.set_minor_locator(locmin)
	ax.yaxis.set_minor_formatter(ticker.NullFormatter())


###################### Symbolics #########################
#
#	Symbolic manipulation using sympy

A_sym = Eq["A_sym"]

print("A = ", A_sym)
print()

A_num = Eq["A_num"]
	
"""
define the first and second terms of the Magnus expansion (symbolic form)

Ω_1(t) = \int_t_0^t ( A(t') ) dt'

Ω_2(t) = 0.5 \int_t_0^t( \int_t_0^t'( [A(t'),A(t'')] )dt'' )dt'

"""

made_Om_1 = False

def Omega_1_sym(A):
	integral = sym.integrate(A.subs(ts, ts1), (ts1, ts0, ts))
	return integral
	
def Omega_2_sym(A):
	ts2 = sym.Symbol('ts2')
	integral_1 = sym.integrate(Com(A.subs(ts, ts1),A.subs(ts, ts2)), (ts2, ts0, ts1))
	print("integral_1 = ", integral_1)
	print()
	integral_2 = sym.integrate(integral_1, (ts1, ts0, ts))
	return 0.5*integral_2
	
def Magnus1(alpha):
	def Make_func():
		if alpha == "analytic":
			global made_Om_1
			if not made_Om_1:
				Om_1 = Omega_1_sym(A_sym)
				print("Omega 1 = ", sym.nsimplify(Om_1))
				print()
				global Omega_1_exact
				Omega_1_exact = sym.lambdify((ts0, ts), Om_1, modules=array2mat)
				made_Om_1 = True
			Omega = Omega_1_exact
		elif alpha != "analytic":
			Omega = Omega_num(A_num, alpha, 4)
		def Mf(t0, t):
			Om = Omega(t0, t)
			return linalg.expm(Om)
		return Mf
	return Make_func
	
def Magnus2(alpha):
	def Make_func():
		if alpha == "analytic":
			global made_Om_1
			if not made_Om_1:
				Om_1 = Omega_1_sym(A_sym)
				print("Omega 1 = ", sym.nsimplify(Om_1))
				print()
				global Omega_1_exact
				Omega_1_exact = sym.lambdify((ts0, ts), Om_1, modules=array2mat)
				made_Om_1 = True
			Om_2 = Omega_2_sym(A_sym)
			print("Omega 2 = ", sym.nsimplify(Om_2))
			print()
			Omega_2_exact = sym.lambdify((ts0, ts), Om_2, modules=array2mat)
			def Omega(t0, t):
				Om = Omega_1_exact(t0, t) + Omega_2_exact(t0, t)
				return Om
		elif alpha != "analytic":
			Omega = Omega_num(A_num, alpha, 6)
		def Mf(t0, t):
			Om = Omega(t0, t)
			return linalg.expm(Om)
		return Mf
	return Make_func

def Magnus3(alpha,):
	def Make_func():
		def Mf(t0, t,P,Pinv):
			B=Pinv @ A_num @ P
			Omega = Omega_num(A_num, alpha, 6)
			Om = Omega(t0, t)
			return P @ linalg.expm(Om) @ Pinv
		return Mf
	return Make_func
	
def Cayley(alpha, order):
	# Caley method
	def Make_func():
		if alpha == "analytic":
			# only a order 4 method available
			A = A_sym.subs(ts, ts0)
			Ndim = A.shape[0]
			Om = Omega_1_sym(A_sym) + Omega_2_sym(A_sym)
			Id = sym.eye(Ndim)
			C_ = Om*(Id - (1/12)*(Om**2)*(Id - (1/10)*(Om**2)))
			M_sym = (Id - (1/2)*C_).inv()*(Id + (1/2)*C_)
			print("4th order Cayley matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
			return Mf
		elif alpha != "analytic":
			# order 4 or order 6 methods available
			Omega = Omega_num(A_num, alpha, order)
			Ndim = Eq["x0"].size
			Id = np.identity(Ndim)
			def Mf(t0, t):
				Om = Omega(t0, t)
				if order == 4:
					C_ = Om*(Id - (1/12)*(Om**2))
				elif order ==6:
					C_ = Om*(Id - (1/12)*(Om**2)*(1 - (1/10)*(Om**2)))
				M_ = np.linalg.inv(Id - 0.5*C_)*(Id + 0.5*C_)
			return Mf
	return Make_func
	
def w1_func(t):
	return sym.sqrt(Eq["w2"](t))

def WKB_analytic():
	xA = sym.cos(sym.integrate(w1_func(ts1), (ts1, ts0, ts)))/sym.sqrt(w1_func(ts))
	xB = sym.sin(sym.integrate(w1_func(ts1), (ts1, ts0, ts)))/sym.sqrt(w1_func(ts))
	dxA = sym.diff(xA, ts)
	dxB = sym.diff(xB, ts)
	x_mat = sym.Matrix([[xA, xB], [dxA, dxB]])
	x_mat_0 = x_mat.subs(ts, ts0)
	M_sym = x_mat*x_mat_0.inv()
	print("WKB matrix = ", M_sym)
	print()
	Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
	return Mf

### new methods

def Jordan_WKB(Use_numerics):
	Use_Aprime2_or_J = False
	def Make_func():
		# symbolics
		A = A_sym.subs(ts, ts0)
		Aprime = sym.diff(A, ts0) + A*A
		Ndim = A.shape[0]
		P_0, J_0 = Aprime.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
		if Use_numerics == 0 or Use_numerics == 1:
			J = sym.simplify(J_0)
			P = sym.simplify(P_0)
			print("JWKB:")
			print("J = ", J)
			print()
			print("P = ", P)
			print()
			Pinv = P.inv()
			print("Pinv = ", Pinv)
			print()
			dPinv = sym.diff(Pinv, ts0)
			print("dPinv = ", dPinv)
			print()
			if Use_Aprime2_or_J:
				ddPinv = sym.diff(dPinv, ts0)
				print("ddPinv = ", ddPinv)
				print()
				Aprime2 = ddPinv*P + 2*dPinv*A*P + J 
				print("A'' = ", Aprime2)
				print()
				W2 = -Aprime2
			elif not Use_Aprime2_or_J:
				W2 = -J
			w1_sym = []
			for i in range(0, Ndim):
				w2 = W2[i,i]
				print("w2 = ", w2)
				w1 = sym.sqrt(w2)
				w1_sym.append(w1)
			if Use_numerics == 0:
				# symbolic version
				M11 = sym.eye(Ndim)
				M12 = sym.eye(Ndim)
				for i in range(0, Ndim):
					w1 = w1_sym[i]
					C = sym.cos(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
					S = sym.sin(sym.integrate(w1.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1/w1.subs(ts0, ts))
					dw1 = sym.diff(w1, ts0)
					M11[i,i] = C + S*dw1/(2*w1**2)
					M12[i,i] = S/w1
				M_sym = (P.subs(ts0, ts))*(M11*Pinv + M12*(dPinv + Pinv*A))
				print()
				print("Jordan_WKB matrix = ", M_sym)
				print()
				Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
			elif Use_numerics == 1:
				# semi-numerical version
				A_num = Eq["A_num"]
				P_num = sym.lambdify((ts0), P, modules=array2mat_c)
				Pinv_num = sym.lambdify((ts0), Pinv, modules=array2mat_c)
				dPinv_num = sym.lambdify((ts0), dPinv, modules=array2mat_c)
			
				if Use_Aprime2_or_J:
					Aprime2_num = sym.lambdify((ts0), Aprime2, modules=array2mat_c)
				elif not Use_Aprime2_or_J:
					J_num = sym.lambdify((ts0), J, modules=array2mat_c)
				Id = np.identity(Ndim)
				M11 = Id.astype(np.complex64)
				M12 = Id.astype(np.complex64)
				w1_num = []
				dw1_num = []
				# convert symbolic form into numerical functions
				for i in range(0, Ndim):
					w1_num.append(sym.lambdify((ts0), w1_sym[i], modules=array2mat_c))
					dw1_num.append(eg(w1_num[i], 0.00001))
				def Mf(t0, t):
					# define a function to compute the M matrix
					for i in range(Ndim):
						w1 = w1_num[i](t)
						w10 = w1_num[i](t0)
						dw10 = dw1_num[i](t0)
						Int_w1 = c_quad(w1_num[i], t0, t, ARGS=())
						C = np.cos(Int_w1)*csqrt(w10/w1)
						S = np.sin(Int_w1)*csqrt(w10/w1)
						M11[i,i] = C + S*dw10/(2*(w10)**2)
						M12[i,i] = S/w10
					M_ = P_num(t) @ (M11 @ Pinv_num(t0) + M12 @ (dPinv_num(t0) + Pinv_num(t0) @ A_num(t0)))
					return M_
		elif Use_numerics == 2:
			# version minimising the amount of symbolic manipulation required
			J = J_0
			P = P_0
			print("JWKB:")
			print("J = ", J)
			print()
			print("P = ", P)
			print()
			#Pinv = P.inv()
			#print("Pinv = ", Pinv)
			#print()
			P_num = sym.lambdify((ts0), P, modules=array2mat_c)
			def Pinv_num(t):
				Pt = P_num(t)
				Pinvt = np.linalg.inv(Pt)
				return Pinvt
			J_num = sym.lambdify((ts0), J, modules=array2mat_c)
			dPinv_num = eg(Pinv_num, 0.00001)
			if Use_Aprime2_or_J:
				ddPinv_num = eg(dP_num, 0.00001)
				A_num = Eq["A_num"]
				def Aprime2_num(t):
					ddPinvt = ddPinv_num(t)
					Pt = P_num(t)
					Pinvt = np.linalg.inv(Pt)
					At = A_num(t)
					Jt = J_num(t)
					Aprime2t = ddPinvt @ Pt + 2*dPinvt @ At @ Pt + Jt
					return Aprim2t
				negW2 = Aprime2_num
			elif not Use_Aprime2_or_J:
				negW2 = J_num
			def w1_num(t, n):
				return csqrt(-negW2(t)[n,n])
			def w1_vec(t):
				w1 = np.ones(Ndim)
				W2 = - -negW2(t)
				for i in range(0, Ndim):
					w1[i] = csqrt(W2[i, i])
				return w1
			dw1 = eg(w1_vec, 0.00001)
			def Mf(t0, t):
				# define a function to compute the M matrix
				w1 = w1_vec(t)
				w10 = w1_vec(t0)
				dw10 = dw1(t0)
				for i in range(Ndim):
					Int_w1 = c_quad(w1_sing, t0, t, ARGS=(i))
					C = np.cos(Int_w1)*csqrt(w10[i]/w1[i])
					S = np.sin(Int_w1)*csqrt(w10[i]/w1[i])
					M11[i,i] = C + S*dw10[i]/(2*(w10[i])**2)
					M12[i,i] = S/w10[i] 
					Pinvt0 = dPinv_num(t0)
				M_ = P_num(t) @ (M11 @ Pinvt0 + M12 @ () + Pinvt0 @ A_num(t0))
				return M_
		return Mf
	return Make_func



def Jordan_Magnus(Lambda_only, Use_numerics):
	# The bamber version of the Jordan Magnus expansion
	def Make_func():
		# symbolics
		tic=time.process_time()
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		P_, J_ = A.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
		P = sym.simplify(P_)
		J = sym.simplify(J_)
		toc1=time.process_time()
		print("J = ", J)
		print()
		print("P = ", P)
		print()
		Pinv = P.inv()
		LK_ = J + sym.diff(Pinv, ts0)*P
		LK = sym.simplify(LK_)
		toc2=time.process_time()
		print("LK = ", LK)
		print()
		if Lambda_only:
			# only use the diagonal elements
			LK = sym.eye(Ndim).multiply_elementwise(LK)
			print("L = ", LK)
			print()
		if Use_numerics == 0:
			Om1 = sym.integrate(LK.subs(ts0, ts1), (ts1, ts0, ts)) 
			print("Ω1 = ", Om1)
			print()
			Om1_num = sym.lambdify((ts0, ts), Om1, modules=array2mat_c)
			#JM1["name"] = JM1["name"] + " (analytic)"
		elif Use_numerics == 1:
			LK_num = sym.lambdify((ts0), LK, modules=array2mat_c)
			#
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""
			Om1_num = Omega_num(LK_num, alpha_GL, 4)
		toc3=time.process_time()
		P_num = sym.lambdify((ts0), P, modules=array2mat_c)
		#
		def Mf(t0, t):
			M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ np.linalg.inv(P_num(t0))
			toc4=time.process_time()
			print('Time to decompose = {}'.format(toc1-tic))
			print('Time to produce LK = {}'.format(toc2-toc1))
			print('Time to calculate first maguns term = {}'.format(toc3-toc2))
			print('Time to calculate stepping matrix (matrix exponential) = {}'.format(toc4-toc3))
			return M_.astype(np.float64)
		return Mf
	return Make_func
'''
def DecomposeSchur(t,dt,A):
	A00_np = np.array(A(t))
	J00_, P00_ = linalg.schur(A00_np)
	J00 = sym.Matrix(J00_)
	P00 = sym.Matrix(P00_)
	A00dt_np = np.array(A(t+dt))
	J00dt_, P00dt_ = linalg.schur(A00dt_np)
	P00dt_ = sym.Matrix(P00dt_)
	dP00inv = (P00dt_.inv()-P00.inv())/dt
	return P00, dP00inv, J00
'''

def Jordan_Magnus2(Lambda_only, Use_numerics):
	# The old version of schur decomp Jordan Magnus expansion
	def Make_func():
		def Mf(t0, t):
			tic=time.process_time()
			A=A_num
			dt=1e-8
			Ndim = A_sym.subs(ts, ts0).shape[0]

			#t0
			P0, dP0inv, J0 = DecomposeSchur(t0,dt,A)
	 		#t0_5
			P05, dP05inv, J05 = DecomposeSchur(t0+(t-t0)/2,dt,A)
	 		#t
			P, dPinv, J = DecomposeSchur(t,dt,A)
	 
			toc1=time.process_time()
	 
			LK0 = J0 + dP0inv*P0

			LK05 = J05 + dP05inv*P05

			LK = J + dPinv*P
	 
			toc2=time.process_time()

			if Lambda_only:
				# only use the diagonal elements
				LK0 = sym.eye(Ndim).multiply_elementwise(LK0)
				LK05 = sym.eye(Ndim).multiply_elementwise(LK05)
				LK = sym.eye(Ndim).multiply_elementwise(LK)

			toc3=time.process_time()
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""

			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)
			toc4=time.process_time()
	 
			M_ = P @ linalg.expm(Om1_num(t0,t)) @ np.linalg.inv(P0)
	 
			toc5=time.process_time()
	 
			print('Time to schur decompose = {}'.format(toc1-tic))
			print('Time to produce LK (Matrix multiplication) = {}'.format(toc2-toc1))
			print('Time to calculate stepping matrix (matrix multiplication and exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func

def DecomposeDiagonal(t,dt,A):
	A00_np = sym.Matrix(np.matrix(A(t)))
	P00_, J00_ = A00_np.diagonalize()
	J00 = sym.Matrix(J00_)
	P00 = sym.Matrix(P00_)
	A00dt_np = sym.Matrix(A(t+dt))
	P00dt_, J00dt_ = A00dt_np.diagonalize()
	P00dt_ = sym.Matrix(P00dt_)
	dP00inv = (P00dt_.inv()-P00.inv())/dt
	return P00, dP00inv, J00

def NPDiagonalise(t,dt,A):
	A00_np = np.array(A(t)).astype(np.complex128)
	W00_, V00_ = np.linalg.eig(A00_np)
	J00 = np.linalg.inv(V00_) @ A00_np @ V00_
	A00dt_np = np.array(A(t+dt)).astype(np.complex128)
	W00dt_, V00dt_ = np.linalg.eig(A00dt_np)
	dP00inv = (np.linalg.inv(V00dt_)-np.linalg.inv(V00_))/dt
	return V00_, dP00inv, J00

def Jordan_Magnus3(Lambda_only, Use_numerics):
	#The old version of diagonaliased Jordan Magnus expansion
	def Make_func():
		def Mf(t0, t):

			tic=time.process_time()
	 
			A=A_num
			dt=1e-8*(t-t0)
			Ndim = A_sym.subs(ts, ts0).shape[0]
			
			#t0
			P0, dP0inv, J0 = NPDiagonalise(t0,dt,A)
	 		#t0_5
			P05, dP05inv, J05 = NPDiagonalise(t0+(t-t0)/2,dt,A)
	 		#t
			P, dPinv, J = NPDiagonalise(t,dt,A)
	 
			toc1=time.process_time()
			LK0 = J0 + dP0inv*P0
			

			LK05 = J05 + dP05inv*P05
	
	 		
			LK = J + dPinv*P
		
	 
			toc2=time.process_time()
	 
			#print("LK = ", LK)

			if Lambda_only:
				# only use the diagonal elements
				LK0 = np.diag(np.diag(LK0))
				LK05 = np.diag(np.diag(LK05))
				LK = np.diag(np.diag(LK))
				#print("L = ", LK)

			toc3=time.process_time()
	 
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""

			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)
			Om1_num = np.array(Om1_num(t0,t)).astype(np.complex128)
	 
			toc4=time.process_time()
	 
			M_ = P @ linalg.expm(Om1_num) @ np.linalg.inv(P0)
	 
			toc5=time.process_time()
	 
			print('Time to decompose = {}'.format(toc1-tic))
			print('Time to produce LK = {}'.format(toc2-toc1))
			print('Time to process LK diagonal = {}'.format(toc3-toc2))
			print('Time to calculate first maguns term = {}'.format(toc4-toc3))
			print('Time to calculate stepping matrix (matrix exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func


def Jordan_Magnus4(Lambda_only, Use_numerics):
	#Jordan magnus with reduced number of decompositions
	def Make_func():
		def Mf(t0, t, P, Pinv):
			tic=time.process_time()
			A=A_num
			dt=1e-8
			Ndim = A_sym.subs(ts, ts0).shape[0]
			
			#t0
			LK0 = Pinv*A(t0)*P
	 		#t0_5
			LK05 = Pinv*A(t0+(t-t0)/2)*P
	 		#t
			LK1 = Pinv*A(t)*P
	 
			toc1=time.process_time()
			toc2=time.process_time()

			if Lambda_only:
				# only use the diagonal elements
				LK0 = sym.eye(Ndim).multiply_elementwise(LK0)
				LK05 = sym.eye(Ndim).multiply_elementwise(LK05)
				LK1 = sym.eye(Ndim).multiply_elementwise(LK1)
				#print("L = ", LK)
			toc3=time.process_time()
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""
			Om1_num = Omega_num2(LK0,LK05,LK1, t0, t, alpha_SNC2, 4)
			toc4=time.process_time()
			M_ = P @ linalg.expm(np.array(Om1_num(t0,t)).astype(np.complex64)) @ Pinv
			toc5=time.process_time()
			print('Time to produce LK (matrix multiplication) = {}'.format(toc1-tic))
			print('Time to calculate first maguns term = {}'.format(toc4-toc3))
			print('Time to calculate stepping matrix (matrix multiplication and exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex64)
		return Mf
	return Make_func

	
'''
def NPDiagonalise(t,dt,A):
	A00_np = np.array(A(t)).astype(np.complex128)
	W00_, V00_ = np.linalg.eig(A00_np)
	J00 = np.linalg.inv(V00_) @ A00_np @ V00_
	#J00 = np.diag(W00_)
	A00dt_np = np.array(A(t+dt)).astype(np.complex128)
	W00dt_, V00dt_ = np.linalg.eig(A00dt_np)
	dP00inv = (np.linalg.inv(V00dt_)-np.linalg.inv(V00_))/dt
	return V00_, dP00inv, J00
'''

def NPDiagonalise(t,dt,A): 

	tic=time.process_time()
	A00_np = A(t).astype(np.complex128)
	A00dt_np = A(t+dt).astype(np.complex128)
	#A00dt_np = np.array(A(t+dt)).astype(np.complex128)

	toc1=time.process_time()
	
	W00_, V00_ = np.linalg.eig(A00_np)

	toc2=time.process_time()

	J00 = np.linalg.inv(V00_) @ A00_np @ V00_
	#J00 = np.diag(W00_)

	toc3=time.process_time()

	W00dt_, V00dt_ = np.linalg.eig(A00dt_np)

	toc4=time.process_time()

	a=np.linalg.inv(V00dt_)
	b=np.linalg.inv(V00_)
	dP00inv = (a-b)/dt
	toc5=time.process_time()
	print('A(t)={}'.format(A00_np))
	print('P00_={}'.format(V00_))
	print('J00_={}'.format(J00))
	print('Time to rhs A(t) and A(t+dt) = {}'.format(toc1-tic))
	print('Time to decompose A(t) = {}'.format(toc2-toc1))
	print('Time to inverse and matrix multiply = {}'.format(toc3-toc2))
	print('Time to decompose A(t+dt) = {}'.format(toc4-toc3))
	print('Time to inverse and calculate derivative (subtract)= {}'.format(toc5-toc4))
	print('Total time for one diag process = {}'.format(toc5-tic))
	return V00_, dP00inv, J00

def JMNumpyDiag(Lambda_only, Use_numerics):
	def Make_func():
		def Mf(t0, t):
			tic=time.process_time()

			A=A_num
			dt=1e-8*(t-t0)
			Ndim = A_sym.subs(ts, ts0).shape[0]
			
			#t0
			P0, dP0inv, J0 = NPDiagonalise(t0,dt,A)
	 		#t0_5
			P05, dP05inv, J05 = NPDiagonalise(t0+(t-t0)/2,dt,A)
	 		#t
			P, dPinv, J = NPDiagonalise(t,dt,A)
	 
			toc1=time.process_time()
			LK0 = J0 + dP0inv*P0
			print('LK0={}'.format(LK0))

			LK05 = J05 + dP05inv*P05
			print('LK05={}'.format(LK05))
	 		
			LK = J + dPinv*P
			print('LK={}'.format(LK))
	 
			toc2=time.process_time()
	 
			#print("LK = ", LK)

			if Lambda_only:
				# only use the diagonal elements
				LK0 = np.diag(np.diag(LK0))
				LK05 = np.diag(np.diag(LK05))
				LK = np.diag(np.diag(LK))
				#print("L = ", LK)

			toc3=time.process_time()
	 
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""

			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)
			Om1_num = np.array(Om1_num(t0,t)).astype(np.complex128)
			print('Om1_num={}'.format(Om1_num))
			toc4=time.process_time()
			P0inv= np.linalg.inv(P0)
			print('P0inv={}'.format(P0inv))
			print('P={}'.format(P))
			print('linalg.expm(Om1_num)={}'.format(linalg.expm(Om1_num)))
			M_ = P @ linalg.expm(Om1_num) @ P0inv
	 		
			toc5=time.process_time()
	 
			print('Time to decompose = {}'.format(toc1-tic))
			print('Time to produce LK = {}'.format(toc2-toc1))
			print('Time to process LK diagonal = {}'.format(toc3-toc2))
			print('Time to calculate first maguns term = {}'.format(toc4-toc3))
			print('Time to calculate stepping matrix (matrix exponential) = {}'.format(toc5-toc4))
			print('M={}'.format(M_))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func
	
'''
def DecomposeSchur(t,dt,A):
	A00_np = np.array(A(t))
	J00_, P00_ = linalg.schur(A00_np)
	A00dt_np = np.array(A(t+dt))
	J00dt_, P00dt_ = linalg.schur(A00dt_np)
	dP00inv = (np.linalg.inv(P00dt_)-np.linalg.inv(P00_))/dt
	return P00_, dP00inv, J00_
'''
def DecomposeSchur(t,dt,A):

  tic=time.process_time()
  #A00_np = A(t).astype(np.complex128)
  #A00dt_np = A(t+dt).astype(np.complex128)
  A00_np = A(t)
  A00dt_np = A(t+dt)

  toc1=time.process_time()
  #J00_, P00_ = linalg.schur(A00_np, output='complex')
  J00_, P00_ = linalg.schur(A00_np)
  toc2=time.process_time()
  #J00dt_, P00dt_ = linalg.schur(A00dt_np, output='complex')
  J00dt_, P00dt_ = linalg.schur(A00dt_np)
  print('A(t)={}'.format(A00_np))
  print('P00_={}'.format(P00_))
  print('J00_={}'.format(J00_))
  
  toc3=time.process_time()
  a=np.linalg.inv(P00dt_)
  b=np.linalg.inv(P00_)
  toc4=time.process_time()
  dP00inv = (a-b)/dt
  toc5=time.process_time()


  print('Time to rhs A(t) and A(t+dt) = {}'.format(toc1-tic))
  print('Time to decompose A(t) = {}'.format(toc2-toc1))
  print('Time to decompose A(t+dt) = {}'.format(toc3-toc2))
  print('Time to inverse at t and t+dt = {}'.format(toc4-toc3))
  print('Time to calculate derivative (subtract)= {}'.format(toc5-toc4))
  print('Total time for one decomposeSchur = {}'.format(toc5-tic))
  return P00_, dP00inv, J00_

def JMSchur(Lambda_only, Use_numerics):
	def Make_func():
		def Mf(t0, t):
			tic=time.process_time()
			A=A_num
			dt=1e-8*(t-t0)
			Ndim = A_sym.subs(ts, ts0).shape[0]

			#t0
			P0, dP0inv, J0 = DecomposeSchur(t0,dt,A)
	 		#t0_5
			print('P0={}'.format(P0))
			print('J0={}'.format(J0))
			print('dP0inv={}'.format(dP0inv))
			P05, dP05inv, J05 = DecomposeSchur(t0+(t-t0)/2,dt,A)
	 		#t
			P, dPinv, J = DecomposeSchur(t,dt,A)
	 
			toc1=time.process_time()
	 
			LK0 = J0 + dP0inv*P0
			print('LK0={}'.format(LK0))
			LK05 = J05 + dP05inv*P05

			LK = J + dPinv*P
	 
			toc2=time.process_time()

			if Lambda_only:
				# only use the diagonal elements
				print(LK0)
				LK0 = np.diag(np.diag(LK0))
				print(LK0)
				LK05 = np.diag(np.diag(LK05))
				LK = np.diag(np.diag(LK))

			toc3=time.process_time()
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""

			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)
			Om1_num = np.array(Om1_num(t0,t)).astype(np.complex128)
			toc4=time.process_time()

			P0inv= np.linalg.inv(P0)
			M_ = P @ linalg.expm(Om1_num) @ P0inv
	 
			toc5=time.process_time()
	 
			print('Time to schur decompose at three time points = {}'.format(toc1-tic))
			print('Time to produce LK (Matrix multiplication) = {}'.format(toc2-toc1))
			print('Time to calculate stepping matrix (matrix multiplication and exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func


import numpy as np
import time
from scipy.sparse import block_diag, spdiags


def replace_with_submatrix2(A_fn, x):
    submatrix_size = len(x)  #number of nodes
    m=A_fn(x[0]).shape[0]  #number of variables
    output_size = submatrix_size * m

    output_arr = np.zeros((output_size, output_size))
    

    for i in range(m):
        for j in range(m):
            submatrix = np.zeros((submatrix_size, submatrix_size))
            for k in range(len(x)):
                A_new=A_fn(x[k])[i,j]
                submatrix[k,k] = A_new
            row_start = i * submatrix_size   #row start index
            row_end = (i + 1) * submatrix_size  #row end index
            col_start = j * submatrix_size  #column start index  
            col_end = (j + 1) * submatrix_size  #column end index
            output_arr[row_start:row_end, col_start:col_end] = submatrix
      


        '''
        row_start = i * submatrix_size
        row_end = (i + 1) * submatrix_size

        output_arr[row_start:row_end, row_start:row_end] = submatrix
        '''
       # output_arr[i * submatrix_size, i * submatrix_size] = submatrix


    return output_arr


def replace_with_submatrix(arr, submatrix_shape):
    """
    Replace each element in arr with a submatrix of shape submatrix_shape with that element on the diagonal.

    Parameters:
        arr (ndarray): The input array to modify.
        submatrix_shape (tuple): The shape of the submatrix to replace each element with.

    Returns:
        ndarray: The modified array with submatrices replacing each element.
    """
    # Get the shape of the input array
    orig_shape = arr.shape
    
    # Create a new array with dimensions that can fit the submatrices
    new_shape = (orig_shape[0]*submatrix_shape[0], orig_shape[1]*submatrix_shape[1])
    new_arr = np.zeros(new_shape)

    # Fill each submatrix with the corresponding element from the input array on the diagonal
    for i in range(orig_shape[0]):
        for j in range(orig_shape[1]):
            start_row = i*submatrix_shape[0]
            end_row = (i+1)*submatrix_shape[0]
            start_col = j*submatrix_shape[1]
            end_col = (j+1)*submatrix_shape[1]
            new_arr[start_row:end_row, start_col:end_col] = np.eye(submatrix_shape[0], submatrix_shape[1]) * arr[i,j]

    return new_arr

import numpy as np
import time

def repeat_along_diagonal(A, m):
    """
    Create a new matrix with m copies of A organized along the diagonal.

    Parameters:
        A (ndarray): The input matrix to repeat.
        m (int): The number of times to repeat A along the diagonal.

    Returns:
        ndarray: The new matrix with m copies of A along the diagonal.
    """
    # Get the shape of the input matrix
    n = A.shape[0]

    # Create a new matrix with m copies of A along the diagonal
    new_shape = (n*m, n*m)
    new_A = np.zeros(new_shape)
    for i in range(m):
        new_A[i*n:(i+1)*n, i*n:(i+1)*n] = A

    return new_A

import numpy as np

class Solverinfo:
    """
    Class to store information of an ODE solve run. When initialized, it
    computes Chebyshev nodes and differentiation matrices neede for the run
    once and for all.

    Attributes
    ----------
    w, g: callable
        Frequency and friction function associated with the ODE, defined as :math:`\omega` and
        :math:`\gamma` in :math:`u'' + 2\gamma u' + \omega^2 u = 0`.
    h0: float
        Initial interval length which will be used to estimate the initial derivatives of w, g.
    nini, nmax: int
        Minimum and maximum (number of Chebyshev nodes - 1) to use inside
        Chebyshev collocation steps. The step will use `nmax` nodes or the
        minimum number of nodes necessary to achieve the required local error,
        whichever is smaller. If `nmax` > 2`nini`, collocation steps will be
        attempted with :math:`2^i` `nini` nodes at the ith iteration. 
    n: int
        (Number of Chebyshev nodes - 1) to use for computing Riccati steps.
    p: int
        (Number of Chebyshev nodes - 1) to use for estimating Riccati stepsizes.
    denseout: bool
        Defines whether or not dense output is required for the current run.
    h: float 
        Current stepsize.
    y: np.ndarray [complex]
        Current state vector of size (2,), containing the numerical solution and its derivative.
    wn, gn: np.ndarray [complex]
        The frequency and friction function evaluated at `n` + 1 Chebyshev nodes
        over the interval [x, x + `h` ], where x is the value of the independent
        variable at the start of the current step and `h` is the current
        stepsize.
    Ds: list [np.ndarray [float]]
       List containing differentiation matrices of sizes :math:`(2^i n_{\mathrm{ini}}
       + 1,2^i n_{\mathrm{ini}} + 1)` for :math:`i = 0, 1, \ldots, \lfloor \log_2\\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    nodes: list [np.ndarray [float]]
        List containing vectors of Chebyshev nodes over the standard interval, ordered from +1 to -1, of sizes
        :math:`(2^i n_{\mathrm{ini}} + 1,)` for :math:`i = 0, 1, \ldots, \lfloor \log_2 \\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    ns: np.ndarray [int] 
        Vector of lengths of node vectors stored in `nodes`, i.e. the integers
        :math:`2^i n_{\mathrm{ini}} + 1` for :math:`i = 0, 1, \ldots, \lfloor \log_2 \\frac{n_{\mathrm{max}}}{n_{\mathrm{ini}}} \\rfloor.`
    xn: np.ndarray [float] 
        Values of the independent variable evaluated at (`n` + 1) Chebyshev
        nodes over the interval [x, x + `h`], where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize.
    xp: np.ndarray [float]
        Values of the independent variable evaluated at (`p` + 1) Chebyshev
        nodes over the interval [x, x + `h`], where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize.
    xpinterp: np.ndarray [float]
        Values of the independent variable evaluated at `p` points 
        over the interval [x, x + `h`] lying in between Chebyshev nodes, where x is the value of the
        independent variable at the start of the current step and `h` is the
        current stepsize. The in-between points :math:`\\tilde{x}_p` are defined by
        
        .. math: \\tilde{x}_p = \cos\left( \\frac{(2k + 1)\pi}{2p} \\right), \quad k = 0, 1, \ldots p-1.

    L: np.ndarray [float]    
        Interpolation matrix of size (`p`+1, `p`), used for interpolating a
        function between the nodes `xp` and `xpinterp` (for computing Riccati
        stepsizes). 
    quadwts: np.ndarray [float]
        Vector of size (`n` + 1,) containing Clenshaw-Curtis quadrature weights.
    n_chebnodes: int
        Number of times Chebyhev nodes have been calculated, i.e.
        `riccati.chebutils.cheb` has been called.
    n_chebstep: int
        Number of Chebyshev steps attempted.
    n_chebits: int
        Number of times an iteration of the Chebyshev-grid-based
        collocation method has been performed (note that if `nmax` >=
        4`nini` then a single Chebyshev step may include multiple
        iterations!).
    n_LS:  int
        Number of times a linear system has been solved.
    n_riccstep: int
        Number of Riccati steps attempted.

    Methods
    -------
    increase:
        Increase the various counters (attributes starting with `n_`) by given values.
    output:
        Return the state of the counters as a dictionary.

    """

    def __init__(self, w, g, h, nini, nmax, n, p):

        # Parameters
        self.m = np.shape(A_num(11))[0]
        self.w = w
        self.g = g
        self.h0 = h
        self.nini = nini
        self.nmax = nmax
        self.n = n
        self.p = p
        self.denseout = False
        # Run statistics
        self.n_chebnodes = 0
        self.n_chebstep = 0
        self.n_chebits = 0
        self.n_LS = 0
        self.n_riccstep = 0

        self.h = self.h0
        self.y = np.zeros(2, dtype = complex)
        self.wn, self.gn = np.zeros(n + 1), np.zeros(n + 1)
        Dlength = int(np.log2(self.nmax/self.nini)) + 1
        self.Ds, self.nodes = [], []
        lognini = np.log2(self.nini)
        self.ns = np.logspace(lognini, lognini + Dlength - 1, num = Dlength, base = 2.0)#, dtype=int)
        m = np.shape(A_num(10))[0]
        for i in range(Dlength):
            D, x = cheb(self.nini*2**i,m)
            self.increase(chebnodes = 1)
            self.Ds.append(D)
            self.nodes.append(x)
        if self.n in self.ns:
            i = np.where(self.ns == self.n)[0][0]
            self.Dn, self.xn = self.Ds[i], self.nodes[i]
        else:
            self.Dn, self.xn = cheb(self.n,m)
            self.increase(chebnodes = 1)
        if self.p in self.ns:
            i = np.where(self.ns == self.p)[0][0]
            self.xp = self.nodes[i]
        else:
            self.xp = cheb(self.p,m)[1]
            self.increase(chebnodes = 1)
        self.xpinterp = np.cos(np.linspace(np.pi/(2*self.p), np.pi*(1 - 1/(2*self.p)), self.p))
        self.L = interp(self.xp, self.xpinterp)
        self.increase(LS = 1)
        self.quadwts = quadwts(n)

    def increase(self, chebnodes = 0, chebstep = 0, chebits = 0,
                 LS = 0, riccstep = 0):
        """
        Increases the relevant attribute of the class (a counter for a specific
        arithmetic operation) by a given number. Used for generating performance statistics.

        Parameters
        ----------
        chebnodes: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebnodes`.
        chebstep: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebstep`.
        chebits: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_chebits`.
        LS: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_LS`.
        riccstep: int
            Count by which to increase `riccati.solversetup.Solverinfo.n_riccstep`.

        Returns
        -------
        None
        """
        self.n_chebnodes += chebnodes
        self.n_chebstep += chebstep
        self.n_chebits += chebits
        self.n_LS += LS
        self.n_riccstep += riccstep

    
    def output(self, steptypes):
        """
        Creates a dictionary of the counter-like attributes of `Solverinfo`,
        namely: the number of attempted Chebyshev steps `n_chebsteps`, the
        number of Chebyshev iterations `n_chebits`, the number of attempted
        Riccati steps `n_riccstep`, the number of linear solver `n_LS`, and the
        number of times Chebyshev nodes have been computed, `n_chebnodes`. It
        also logs the number of steps broken down by steptype: in the relevant
        fields, (n, m) means there were n attempted steps out of which m were
        successful.

        Parameters
        ----------
        steptypes: list [int]
            List of steptypes (of successful steps) produced by
            `riccati.evolve.solve()`, each element being 0 (Chebyshev step) or
            1 (Riccati step).

        Returns
        -------
        statdict: dict
            Dictionary with the following keywords:
            
            cheb steps: tuple [int] 
                (n, m), where n is the total number of attempted Chebyshev steps out of which m were successful.
            cheb iterations: int
                Total number of iterations of the Chebyshev collocation method. 
            ricc steps: tuple [int]
                (n, m), where n is the total number of attempted Chebyshev steps out of which m were successful.
            linear solves: int
                Total number of times a linear solve has been performed.
            cheb nodes: int
                Total number of times a call to compute Chebyshev nodes has been made.
        """
        statdict = {"cheb steps": (self.n_chebstep, sum(np.array(steptypes) == 0) - 1), 
                    "cheb iterations": self.n_chebits, 
                    "ricc steps": (self.n_riccstep, sum(np.array(steptypes) == 1)), 
                    "linear solves": self.n_LS, 
                    "cheb nodes": self.n_chebnodes}
        return statdict


def solversetup(w, g, nini, nmax , h0 ,n = 4, p = 4):
    """
    Sets up the solver by generating differentiation matrices based on an
    increasing number of Chebyshev gridpoints (see `riccati.solversetup.Solverinfo`).  Needs to be called
    before the first time the solver is ran or if nini or nmax are changed. 

    Parameters
    ----------
    w: callable(t)
        Frequency function in y'' + 2g(t)y' + w^2(t)y = 0.
    g: callable(t)
        Damping function in y'' + 2g(t)y' + w^2(t)y = 0.
    h0: float
        Initial stepsize.
    nini: int
        Lowest (number of nodes - 1) to start the Chebyshev spectral steps with.
    nmax: int
        Maximum (number of nodes - 1) for the spectral Chebyshev steps to
        try before decreasing the stepsize and reattempting the step with (nini + 1) nodes.
    n: int
        (Fixed) number of Chebyshev nodes to use for interpolation,
        integration, and differentiation in Riccati steps.
    p: int
        (Fixed) number of Chebyshev nodes to use for interpolation when
        determining the stepsize for Riccati steps.

    Returns
    -------
    info: Solverinfo object
        Solverinfo object created with attributes set as per the input parameters.
    """
    info = Solverinfo(w, g, h0, nini, nmax, n, p)
    return info
import numpy as np

def solversetup2(w, g, nini, nmax, h0, n = 4, p = 4):
    """
    Sets up the solver by generating differentiation matrices based on an
    increasing number of Chebyshev gridpoints (see `riccati.solversetup.Solverinfo`).  Needs to be called
    before the first time the solver is ran or if nini or nmax are changed. 

    Parameters
    ----------
    w: callable(t)
        Frequency function in y'' + 2g(t)y' + w^2(t)y = 0.
    g: callable(t)
        Damping function in y'' + 2g(t)y' + w^2(t)y = 0.
    h0: float
        Initial stepsize.
    nini: int
        Lowest (number of nodes - 1) to start the Chebyshev spectral steps with.
    nmax: int
        Maximum (number of nodes - 1) for the spectral Chebyshev steps to
        try before decreasing the stepsize and reattempting the step with (nini + 1) nodes.
    n: int
        (Fixed) number of Chebyshev nodes to use for interpolation,
        integration, and differentiation in Riccati steps.
    p: int
        (Fixed) number of Chebyshev nodes to use for interpolation when
        determining the stepsize for Riccati steps.

    Returns
    -------
    info: Solverinfo object
        Solverinfo object created with attributes set as per the input parameters.
    """
    info = Solverinfo(w, g, h0, nini, nmax, n, p)
    return info
import numpy as np

def coeffs2vals(coeffs):
    """
    Convert the Chebyshev coefficient representation of a set of polynomials `P_j` to their 
    values at Chebyshev nodes of the second kind (ordered from +1 to -1). This function returns a
    matrix `V` such that for an input coefficient matrix `C`,

    .. math:: V_{ij} = P_j(x_i) = \sum_{k=0}^{n} C_{kj}T_k(x_i).

    Taken from the `coeff2vals`_ function in the chebfun package.

    .. _`coeff2vals`: https://github.com/chebfun/chebfun/blob/master/%40chebtech2/coeffs2vals.m

    Parameters
    ----------
    coeffs: numpy.ndarray [float (real)]
       An array of size (n+1, m), with the (i, j)th element representing the
       projection of the jth input polynomial onto the ith Chebyshev
       polynomial.


    Returns
    -------
    values: np.ndarray [float (real)]
        An array of size (n+1, m), with the (i, j)th element representing the
        jth input polynomial evaluated at the ith Chebyshev node.

    """
    n = coeffs.shape[0]
    if n <= 1:
        values = coeffs
    else:
        coeffs[1:n-1,:] /= 2.0
        tmp = np.vstack((coeffs, coeffs[n-2:0:-1,:])) 
        values = np.real(np.fft.fft(tmp, axis = 0))
        values = values[:n,:]
    return values


def vals2coeffs(values):
    """
    Convert a matrix of values of `m` polynomials evaluated at `n+1` Chebyshev
    nodes of the second kind, ordered from +1 to -1, to their interpolating
    Chebyshev coefficients. This function returns the coefficient matrix `C`
    such that for an input matrix of values `V`,

    .. math:: F_j(x) = \sum_{k=0}^{n} C_{kj}T_k(x) 

    interpolates the values :math:`[V_{0j}, V_{1j}, \ldots, V_{nj}]` for :math:`j = 0..(m-1)`.

    Taken from the `vals2coeffs`_ function in the chebfun package.

    .. _`vals2coeffs`: https://github.com/chebfun/chebfun/blob/master/%40chebtech2/vals2coeffs.m

    Parameters
    ----------
    values: np.ndarray [float (real)]
        An array of size (n+1, m), with the (i, j)th element being the jth
        polynomial evaluated at the ith Chebyshev node. 

    Returns
    -------
    coeffs: np.ndarray [float (real)] 
        An array of size (n+1, m) with the (i, j)th element being the
        coefficient multiplying the ith Chebyshev polynomial for interpolating
        the jth input polynomial.

    """
    n = values.shape[0]
    if n <= 1:
        coeffs = values
    else:
        tmp = np.vstack((values[:n-1], values[n-1:0:-1]))
        coeffs = np.real(np.fft.ifft(tmp, axis = 0))
        coeffs = coeffs[0:n,:]
        coeffs[1:n-1,:] *= 2
    return coeffs


def integrationm(n):
    """
    Chebyshev integration matrix. It maps function values at n Chebyshev nodes
    of the second kind, ordered from +1 to -1, to values of the integral of the
    interpolating polynomial at those points, with the last value (start of the
    interval) being zero. Taken from the `cumsummat`_ function in the chebfun package.

    .. _`cumsummat`: https://github.com/chebfun/chebfun/blob/master/%40chebcolloc2/chebcolloc2.m

    Parameters
    ----------
    n: int
        Number of Chebyshev nodes the integrand is evaluated at. Note the nodes
        are always ordered from +1 to -1.

    Returns
    -------
    Q: np.ndarray [float (real)]
        Integration matrix of size (n, n) that maps values of the integrand at
        the n Chebyshev nodes to values of the definite integral on the
        interval, up to each of the Chebyshev nodes (the last value being zero by definition). 

    """
    n -= 1
    T = coeffs2vals(np.identity(n+1))
    Tinv = vals2coeffs(np.identity(n+1))
    k = np.linspace(1.0, n, n)
    k2 = 2*(k-1)
    k2[0] = 1
    B = np.diag(1/(2*k), -1) - np.diag(1/k2, 1)
    v = np.ones(n)
    v[1::2] = -1
    B[0,:] = sum(np.diag(v) @ B[1:n+1,:], 0)
    B[:,0] *= 2
    Q = T @ B @ Tinv
    Q[-1,:] = 0
    return Q


def quadwts(n):
    """
    Clenshaw-Curtis quadrature weights mapping function evaluations at
    (n+1) Chebyshev nodes of the second kind, ordered from +1 to -1, to value of the
    definite integral of the interpolating function on the same interval. Taken
    from [1]_ Ch 12, `clencurt.m`

    Parameters
    ----------
    n: int
        The (number of Chebyshev nodes - 1) for which the quadrature weights are to be computed. 

    Returns
    -------
    w: np.ndarray [float (real)]
        Array of size (n+1,) containing the quadrature weights.

    References
    ----------
    .. [1] Trefethen, Lloyd N. Spectral methods in MATLAB. Society for
      industrial and applied mathematics, 2000.

    """
    if n == 0:
        w = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        w = np.zeros(n+1)
        v = np.ones(n-1)
        if n % 2 == 0:
            w[0] = 1.0/(n**2 - 1)
            w[n] = w[0]
            for k in range(1, n//2):
                v = v - 2*np.cos(2*k*a[1:-1])/(4*k**2 - 1)
            v -= np.cos(n*a[1:-1])/(n**2 - 1)
        else:
            w[0] = 1.0/n**2
            w[n] = w[0]
            for k in range(1,(n+1)//2):
                v -= 2*np.cos(2*k*a[1:-1])/(4*k**2 - 1)
        w[1:-1] = 2*v/n
    return w



def cheb(n,m):
    """
    Returns a differentiation matrix D of size (n+1, n+1) and (n+1) Chebyshev
    nodes x for the standard 1D interval [-1, 1]. The matrix multiplies a
    vector of function values at these nodes to give an approximation to the
    vector of derivative values. Nodes are output in descending order from 1 to
    -1. The nodes are given by
    
    .. math:: x_p = \cos \left( \\frac{\pi n}{p} \\right), \quad n = 0, 1, \ldots p.

    Parameters
    ----------
    n: int
        Number of Chebyshev nodes - 1.
    m: no. of variables

    Returns
    -------
    D: numpy.ndarray [float]
        Array of size (n+1, n+1) specifying the differentiation matrix.
    x: numpy.ndarray [float]
        Array of size (n+1,) containing the Chebyshev nodes.
    """
    if n == 0:
        x = 1
        D = 0
        w = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        x = np.cos(a)
        b = np.ones_like(x)
        b[0] = 2
        b[-1] = 2
        d = np.ones_like(b)
        d[1::2] = -1
        c = b*d
        X = np.outer(x, np.ones(n+1))
        dX = X - X.T
        D = np.outer(c, 1/c) / (dX + np.identity(n+1))
        D = D - np.diag(D.sum(axis=1))
    D= repeat_along_diagonal(D,m)
    return D, x


def interp(s, t):
    """
    Creates interpolation matrix from an array of source nodes s to target nodes t.
    Taken from `here`_ .

    .. _`here`: https://github.com/ahbarnett/BIE3D/blob/master/utils/interpmat_1d.m


    Parameters
    ----------
    s: numpy.ndarray [float]
        Array specifying the source nodes, at which the function values are known.
    t: numpy.ndarray [float]
        Array specifying the target nodes, at which the function values are to
        be interpolated.

    Returns
    -------
    L: numpy.ndarray [float]
        Array defining the inteprolation matrix L, which takes function values
        at the source points s and yields the function evaluated at target
        points t. If s has size (p,) and t has size (q,), then L has size (q, p).
    """
    r = s.shape[0]
    q = t.shape[0]
    V = np.ones((r, r))
    R = np.ones((q, r))
    for j in range(1, r):
        V[:, j] = V[:, j-1]*s
        R[:, j] = R[:, j-1]*t
    L = np.linalg.solve(V.T, R.T).T
    return L


def spectral_cheb(info, x0, h, y0, dy0, niter):
    """
    Utility function to apply a spectral collocation method based on Chebyshev nodes from
    x = x0 to x = x0+h, starting from the initial conditions y(x0) = y0, y'(x0)
    = dy0. In each spectral collocation/Chebyshev step, the solver iterates
    over how many nodes to perform the step with, starting from `info.nini`,
    doubling in every iteration until `info.nmax` is reached or the required
    tolerance is satisfied, whichever comes sooner. The `niter` parameter keeps
    track of how many iterations have been done and is used to retrieve the
    relevant pre-computed differentiation matrix and vector of nodes from the
    input `info` object.

    Parameters
    ----------
    info: Solverinfo object

    x0: float (real)
        Value of the independent variable to take a spectral step from.

    h: float (real)
        Stepsize.

    y0, dy0: complex
        Initial conditions (value of the solution and its derivative) at x0.

    niter: int
        Integer to keep track of how many iterations of the spectral collocation step have been done.

    Returns
    -------
    y1, dy1: complex
        Numerical estimate of the solution and its derivative at the end of the step, at x0+h.

    xscaled: numpy.ndarray [float]
        Chebyshev nodes used for the current iteration of the spectral
        collocation method scaled to lie between [x0, x0+h].

    """
    D, x = info.Ds[niter], info.nodes[niter]
    xscaled = h/2*x + x0 + h/2

    w2 = info.w(xscaled)
    #w2=replace_with_submatrix(w2,(len(xscaled),len(xscaled)))
    
    #print("w2:")
    #print(w2)
    #print("D shape:",D.shape, "w2 shape:", w2.shape)
    
    D2 = 4/h**2*(D @ D) + w2
    
    #print("D2 shape", (4/h**2*(D @ D)).shape)
    #print("D2:", D2)

    #need to make ic 0001000... and 000...0001
    n = round(info.ns[niter])
    m=info.m

    D2ic = np.zeros(((n+1)*m, m*(n+1)), dtype=complex)
    D2ic[:m*(n+1)] = D2
    for i in range(m):
        D2ic = np.append(D2ic,2/h*D[(n+1)*(i+1)-1].reshape(1,-1),axis=0) 
    print('Dmatrix_ic shape=', np.shape(2/h*D[-1]))
    print('Dmatrix_ic =', (2/h*D[-1]))
    for i in range(m):
        ic = np.zeros((n+1), dtype=complex)
        ic[-1] = 1 # Because nodes are ordered backwards, [1, -1]
        a=np.zeros(m)
        a[i]=1
        ic=np.kron(a,ic)
        D2ic = np.append(D2ic,ic.reshape(1,-1),axis=0)

    print('D2ic shape=', np.shape(D2ic))

    rhs = np.zeros(info.m*(n+1), dtype=complex)
    rhs = np.append(rhs,dy0)
    rhs = np.append(rhs,y0) 
    print('rhs shape=', np.shape(rhs))

    print("D2ic")
    print(D2ic)

    print('rhs')
    print(rhs)


    y1, res, rank, sing = np.linalg.lstsq(D2ic, rhs) # NumPy solve only works for square matrices
    print('y1=', y1)
    dy1 = 2/h*(D @ y1)
    info.increase(LS = 1)
    y2=np.zeros(m, dtype=complex)
    dy2=np.zeros(m, dtype=complex)
    
    for i in range(m):
        y2[i]=y1[i*(n+1)]
        dy2[i]=dy1[i*(n+1)]
    print('y2=', y2)
    
    return y2, dy2, xscaled



def spectral_cheb_A(A, x0, h, y0, dy0, p):
    """
    Takes a single step from x = x0 of size h, with p collocation point, with
    the spectral collocation method. The dependent variables are y and y', and
    their equation of motion is y' = A(x)y. The initial conditions are y(x0) = y0,
    y'(x0) = dy0. The matrix A accepts a vector-valued x if y, y0 are
    themselves vector-valued. 
    """
    m=A.shape[0]
    D, x = cheb(p-1, 1)
    D2 = repeat_along_diagonal(D, 2)
    xscaled = h/2*x + x0 + h/2
    lhs = np.zeros((2*p+2, 2*p), dtype=complex)
    lhs[:2*p] = 2/h*D2 - A(xscaled)
    lhs[-2, p-1] = 1.0
    lhs[-1, -1] = 1.0
    rhs = np.zeros(2*p+2, dtype=complex)
    rhs[-2] = y0
    rhs[-1] = dy0
    y1, res, *misc = np.linalg.lstsq(lhs, rhs)
    return y1, res, xscaled

def spectral_cheb_B(A, x0, h, y0, dy0, p):
    """
    Takes a single step from x = x0 of size h, with p collocation point, with
    the spectral collocation method. The dependent variables are y and y', and
    their equation of motion is y' = A(x)y. The initial conditions are y(x0) = y0,
    y'(x0) = dy0. The matrix A accepts a vector-valued x if y, y0 are
    themselves vector-valued. 
    """
    m=np.ravel(y0).shape[0]
    D, x = cheb(p-1, 1)
    D2 = repeat_along_diagonal(D, m)
    #print('D2 shape=', np.shape(D2))
    #print('m =', m)
    xscaled = h/2*x + x0 + h/2
    #l1 = 2/h*D2 - A(xscaled)
    A = replace_with_submatrix2(A, xscaled)
    l1 = 2/h*D2 - A
    lhs = np.zeros((p*m, m*p), dtype=complex)
    lhs[:m*p] = l1
    for i in range(m):
        lhs = np.append(lhs,2/h*D2[p*(i+1)-1].reshape(1,-1),axis=0) 
    #print('Dmatrix_ic shape=', np.shape(2/h*D2[-1]))
    #print('Dmatrix_ic =', (2/h*D2[-1]))
    for i in range(m):
        ic = np.zeros(p, dtype=complex)
        ic[-1] = 1 # Because nodes are ordered backwards, [1, -1]
        a=np.zeros(m)
        a[i]=1
        ic=np.kron(a,ic)
        lhs = np.append(lhs,ic.reshape(1,-1),axis=0)

    #print('lhs shape=', np.shape(lhs))

    rhs = np.zeros(m*p, dtype=complex)
    rhs = np.append(rhs,dy0)
    rhs = np.append(rhs,y0) 
    #print('rhs shape=', np.shape(rhs))

    #print("lhs")
    #print(lhs)

    #print('rhs')
    #print(rhs)


    y1, res, rank, sing = np.linalg.lstsq(lhs, rhs) # NumPy solve only works for square matrices
    #print('y1=', y1)

    dy1 = 2/h*(D2 @ y1)
    y2=np.zeros(m, dtype=complex)
    dy2=np.zeros(m, dtype=complex)
    
    for i in range(m):
        y2[i]=y1[i*p]
        dy2[i]=dy1[i*p]
    #print('y2=', y2)
    
    return y2, dy2, xscaled



import numpy as np

def nonosc_step(info, x0, h, y0, dy0, epsres):
    """
    A single Chebyshev step to be called from the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial
    conditions `y(x0) = y0`, `y'(x0) = dy0`.
    The function uses a Chebyshev spectral method with an adaptive number of
    nodes. Initially, `info.nini` nodes are used, which is doubled in each
    iteration until `epsres` relative accuracy is reached or the number of
    nodes would exceed `info.nmax`. The relative error is measured as the
    difference between the predicted value of the dependent variable at the end
    of the step obtained in the current iteration and in the previous iteration
    (with half as many nodes). If the desired relative accuracy cannot be
    reached with `info.nmax` nodes, it is advised to decrease the stepsize `h`,
    increase `info.nmax`, or use a different approach. 

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Chebyshev steps.

    Returns
    -------
    y[0], dy[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        (Absolute) value of the relative difference of the dependent variable
        at the end of the step as predicted in the last and the previous
        iteration.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    """
    
    success = 1
    maxerr = 10*epsres
    N = info.nini
    Nmax = info.nmax
    yprev, dyprev, xprev = spectral_cheb(info, x0, h, y0, dy0, 0)
    RHS=1
    while maxerr > epsres:
        N *= 2
        if N > Nmax:
            success = 0
            return y, dy, maxerr, success
        y, dy, x = spectral_cheb(info, x0, h, y0, dy0, int(np.log2(N/info.nini))) 
        maxerr = np.mean(np.abs((yprev - y)/y))
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
        RHS=RHS+1
    info.increase(chebstep = 1)
    if info.denseout:
        # Store interp points
        info.yn = y
        info.dyn = dy
    return y, dy, maxerr, success, RHS


def nonosc_step2(info, x0, h, y0, dy0):
    """
    A single Chebyshev step to be called from the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial
    conditions `y(x0) = y0`, `y'(x0) = dy0`.
    The function uses a Chebyshev spectral method with an adaptive number of
    nodes. Initially, `info.nini` nodes are used, which is doubled in each
    iteration until `epsres` relative accuracy is reached or the number of
    nodes would exceed `info.nmax`. The relative error is measured as the
    difference between the predicted value of the dependent variable at the end
    of the step obtained in the current iteration and in the previous iteration
    (with half as many nodes). If the desired relative accuracy cannot be
    reached with `info.nmax` nodes, it is advised to decrease the stepsize `h`,
    increase `info.nmax`, or use a different approach. 

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Chebyshev steps.

    Returns
    -------
    y[0], dy[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        (Absolute) value of the relative difference of the dependent variable
        at the end of the step as predicted in the last and the previous
        iteration.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    """
    success = 1
    
    N = info.nini
    yprev, dyprev, xprev = spectral_cheb(info, x0, h, y0, dy0, 0)
    y, dy, x = spectral_cheb(info, x0, h, y0, dy0, int(np.log2(N/info.nini))) 
    maxerr = np.mean(np.abs((yprev - y)/y))
   
    yprev = y
    dyprev = dy
    xprev = x
    info.increase(chebstep = 1)
    if info.denseout:
        # Store interp points
        info.yn = y
        info.dyn = dy
    return y, dy, maxerr, success, N


def nonosc_step3(A, x0, h, y0, dy0, nini, nmax, epsres):
    """
    A single Chebyshev step to be called from the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial
    conditions `y(x0) = y0`, `y'(x0) = dy0`.
    The function uses a Chebyshev spectral method with an adaptive number of
    nodes. Initially, `info.nini` nodes are used, which is doubled in each
    iteration until `epsres` relative accuracy is reached or the number of
    nodes would exceed `info.nmax`. The relative error is measured as the
    difference between the predicted value of the dependent variable at the end
    of the step obtained in the current iteration and in the previous iteration
    (with half as many nodes). If the desired relative accuracy cannot be
    reached with `info.nmax` nodes, it is advised to decrease the stepsize `h`,
    increase `info.nmax`, or use a different approach. 

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Chebyshev steps.

    Returns
    -------
    y[0], dy[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        (Absolute) value of the relative difference of the dependent variable
        at the end of the step as predicted in the last and the previous
        iteration.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    """
    
    success = 1
    maxerr = 10*epsres
    N = nini
    Nmax = nmax
    yprev, dyprev, xprev = spectral_cheb_B(A, x0, h, y0, dy0, nini+1)
    RHS=1
    while maxerr > epsres:
        N *= 2
        if N > Nmax:
            success = 0
            return y, dy, maxerr, success
        y, dy, x = spectral_cheb_B(A, x0, h, y0, dy0, N+1) 
        maxerr = np.max(np.abs((yprev - y)/y))
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
        RHS=RHS+1
    return y, dy, maxerr, success, RHS

def nonosc_step4(A, x0, h, y0, dy0, nini):
    """
    A single Chebyshev step to be called from the `solve()` function.
    Advances the solution from `x0` by `h`, starting from the initial
    conditions `y(x0) = y0`, `y'(x0) = dy0`.
    The function uses a Chebyshev spectral method with an adaptive number of
    nodes. Initially, `info.nini` nodes are used, which is doubled in each
    iteration until `epsres` relative accuracy is reached or the number of
    nodes would exceed `info.nmax`. The relative error is measured as the
    difference between the predicted value of the dependent variable at the end
    of the step obtained in the current iteration and in the previous iteration
    (with half as many nodes). If the desired relative accuracy cannot be
    reached with `info.nmax` nodes, it is advised to decrease the stepsize `h`,
    increase `info.nmax`, or use a different approach. 

    Parameters
    ----------
    info: `Solverinfo` object
        `Solverinfo` object used to read off various matrices required for
        numerical differentiation, and `info.wn`, `info.gn` for evaluations of
        w(x), g(x) over [x0, x0+h].
    x0: float
        Starting value of the independent variable.
    h: float
        Stepsize.
    y0, dy0: complex
        Value of the dependent variable and its derivative at `x = x0`.
    epsres: float
        Tolerance for the relative accuracy of Chebyshev steps.

    Returns
    -------
    y[0], dy[0]: complex
        Value of the dependent variable and its derivative at the end of the
        step, `x = x0 + h`.
    maxerr: float
        (Absolute) value of the relative difference of the dependent variable
        at the end of the step as predicted in the last and the previous
        iteration.
    success: int
        Takes the value `1` if the asymptotic series has reached `epsres`
        residual, `0` otherwise.
    """
    
    RHS=0
    
    y, dy, x = spectral_cheb_B(A, x0, h, y0, dy0, nini) 

    RHS=RHS+1
    return y, dy, RHS


############# define some functions ##########

def eg(A, dt):
	# compute the elementwise derivative of a matrix valued function
	def dA(t):
		dA_ = (A(t + 0.5*dt) - A(t - 0.5*dt))/dt
		return dA_
	return dA
A_sym = Eq["A_sym"]


A_num = Eq["A_num"]

def W2_from_A(A_num):
  def f(t):
    W2 = np.matrix((-eg(A_num,dt=1e-14)(t)-A_num(t)*A_num(t)))
    return W2
  return f




###### set up integrator dictionaries #########################

"""
maybe put some other settings in here to make using different integrators easier?
"""
RKF45 = {
	"name" : "RKF 4(5)",
	"fname" : "RKF45" 
}

M1 = {
	"name" : "first Magnus, analytic",
	"fname" : "M1",
	"alpha" : "analytic",
	"order" : 2, 
	"Mfunc" : Magnus1("analytic")
}

M2 = {
	"name" : "second Magnus, analytic",
	"fname" : "M2",
	"alpha" : "analytic",
	"order" : 4, 
	"Mfunc" : Magnus2("analytic")
}

WKB = {
	"name" : "RKWKB",
	"fname" : "WKB",
	"alpha" : "analytic",
	"order" : 1, 
	"Mfunc" : WKB_analytic
}	

M4_GL = {
	"name" : "Magnus 4$^\\circ$, GL quad",
	"fname" : "M4GL",
	"alpha" : alpha_GL,
	"order" : 4, 
	"Mfunc" : Magnus1(alpha_GL)
}

M4_D = {
	"name" : "Magnus 4$^\\circ$, num. diff",
	"fname" : "M4D",
	"alpha" : alpha_D,
	"order" : 4, 
	"Mfunc" : Magnus1(alpha_D)
}

M4_SNC = {
	"name" : "Magnus 4$^\\circ$, Simpson quad",
	"fname" : "M4SNC",
	"alpha" : alpha_SNC,
	"order" : 4,  
	"Mfunc" : Magnus1(alpha_SNC)
}

M6_D = {
	"name" : "Magnus 6$^\\circ$, num. diff",
	"fname" : "M6D",
	"alpha" : alpha_D,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_D)
}

M6_GL = {
	"name" : "Magnus 6$^\\circ$, GL quad",
	"fname" : "M6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_GL)
}

M6_GL2 = {
	"name" : "Magnus 6$^\\circ$, GL quad",
	"fname" : "M6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"Mfunc" : Magnus3(alpha_GL)
}

M6_SNC = {
	"name" : "Magnus 6$^\\circ$, NC quad",
	"fname" : "M6SNC",
	"alpha" : alpha_SNC,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_SNC)
}

C4_GL = {
	"name" : "Cayley 4$^\\circ$, GL quad",
	"fname" : "C4GL",
	"alpha" : alpha_GL,
	"order" : 4, 
	"Mfunc" : Cayley(alpha_GL, 4)
}

C6_GL = {
	"name" : "Cayley 6$^\\circ$, GL quad",
	"fname" : "C6GL",
	"alpha" : alpha_GL,
	"order" : 6,
	"Mfunc" : Cayley(alpha_GL, 6)
}

JWKB = {
	"name" : "JWKB",
	"fname" : "JWKB",
	"order" : 4, 
	"analytic" : 2,
	"Use_numerics" : 0,
	"Mfunc" : Jordan_WKB(0) 
}

JWKBnum = {
	"name" : "JWKB",
	"fname" : "JWKBnum",
	"order" : 4, 
	"Use_numerics" : 1,
	"Mfunc" : Jordan_WKB(1)
}
'''
PWKB = {
	"name" : "PWKB",
	"fname" : "PWKB",
	"order" : 4, 
	"Use_numerics" : 0, 
	"Mfunc" : Pseudo_WKB(0)
}

PWKBnum = {
	"name" : "PWKB",
	"fname" : "PWKB",
	"order" : 4, 
	"Use_numerics" : 1, 
	"Mfunc" : Pseudo_WKB(1)
}
'''
JMl = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Jordan_Magnus(True, 0)
}

JMlk = {
	"name" : "JM ($\\Lambda$ and $K$)",
	"fname" : "JM1lk",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Jordan_Magnus(False, 0)
}

JMlnum = {
	"name" : "JM Old version($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus2(True, 1)
}

JMlknum = {
	"name" : "JM Old version($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus3(False, 1)
}

JMlnumSchur = {
	"name" : "JM Schur($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : JMSchur(True, 1)
}

JMlknumSchur = {
	"name" : "JM Schur ($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : JMSchur(False, 1)
}

JMlnumDiag = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : JMNumpyDiag(True, 1)
}

JMlknumDiag = {
	"name" : "JM Diagonal ($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : JMNumpyDiag(False, 1)
}


JMlnum2 = {
	"name" : "JM Reduced Decomp ($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus4(True, 1)
}

JMlknum2 = {
	"name" : "JM Reduced Decomp($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus4(False, 1)
}
'''
EPWKB = {
	"name" : "EPWKB",
	"fname" : "EPWKB",
	"order" : 4,
	"Use_numerics" : 0,
	"Mfunc" : Ext_Pseudo_WKB(0)
}

EPWKBnum = {
	"name" : "EPWKB",
	"fname" : "EPWKBnum",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Ext_Pseudo_WKB(1)
}

MM1 = {
	"name" : "MM1",
	"fname" : "MM1",
	"alpha" : "analytic",
	"order" : 2,
	"Use_numerics" : 0, 
	"Mfunc" : Modified_M1(0, "analytic")
}

MM1num = {
	"name" : "MM1",
	"fname" : "MM1",
	"alpha" : alpha_GL,
	"order" : 2,
	"Use_numerics" : 1, 
	"Mfunc" : Modified_M1(1, alpha_GL)
}
'''
SpectralAdaptive = {
	"name" : "Spectral Adaptive",
	"fname" : "SpectralA",
	"Use_numerics" : 1, 
    "FixedStep" : 0
 
}

SpectralFixed = {
	"name" : "Spectral Fixed",
	"fname" : "SpectralF",
	"Use_numerics" : 1, 
    "FixedStep" : 1
 
}

######################################

# choose the lines to plot (i.e. the integrators to use)
lines = [SpectralFixed,M1]

############### set up Numerics #################

Use_RK = False

for line in lines:
	if line["fname"] != "RKF45" and line["fname"] != "SpectralF" and line["fname"] != "SpectralA":
		line["M"] = line["Mfunc"]()
	elif line["fname"] == "RKF45":
		Use_RK = True
		
# correct line labels
for M in [JWKBnum, JMlnum, JMlknum]:
	M["name"] = M["name"] + " (scipy quad, maxiter=" + str(scipy_quad_maxiter) + ")"

########## Integrator #################

# set error tolerance
epsilon	= 0.005
epsilon_RK = 0.005
rtol = 1		# rel. error tolerance for Magnus in units of ε
atol = 0.005	# abs. error tolerance for Magnus in units of ε
rtol_RK = 2		# rel. error tolerance for RKF4(5) in units of ε_RK
atol_RK = 1		# abs. error tolerance for RKF4(5) in units of ε_RK

def RKF45_Integrator(t_start, t_stop, h0, x0, A, epsilon_RK):
	# An integrator using a 4(5) RKF method
	T_0 = time.time()
	RHS=0
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	n_step = number of steps
	A = A(t) matrix function
	"""
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.005 
	h_max = 2.5
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using 4th and 5th order RK methods
			k1 = np.dot(h*A(t),x_n)
			k2 = h*A(t + 0.25*h) @ (x_n + 0.25*k1)
			k3 = h*A(t + (3/8)*h) @ (x_n + (3/32)*k1 + (9/32)*k2)
			k4 = h*A(t + (12/13)*h) @ (x_n + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
			k5 = h*A(t + h) @ (x_n + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
			k6 = h*A(t + 0.5*h) @ (x_n - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
			y_np1 = x_n + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (11/40)*k5
			z_np1 = x_n + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
			#
			# catch errors
			if np.isnan(z_np1[0]):
				print("NaN error")
				sysexit()
			if np.any(np.isinf(z_np1)) or np.any(np.isinf(y_np1)):
				print("y_np1 = ", y_np1)
				print("z_np1 = ", z_np1)
				print("Inf error")
				sysexit()
			#
			#Err = np.abs(y_np1[0] - z_np1[0])
			Err = ferr(y_np1, z_np1)
			"""
			Err_max = ε(rtol*|z_np1| + atol)
			"""
			Err_max = epsilon_RK*(rtol_RK*np.abs(z_np1) + atol_RK)
			#Err_ratio = np.asscalar(np.mean(Err / Err_max))
			Err_ratio = (np.mean(Err / Err_max)).item(0)
			#
			RHS+=6
			if Err_ratio <= 1:
				h_new = h*S*np.power(Err_ratio, -1.0/5)
				#Delta = max(np.asscalar(max(Err)), epsilon_RK*0.1)
				#h_new = h*(epsilon_RK*h/Delta)**(1/4)
				if h_new > 10*h:	# limit how fast the step size can increase
					h_new = 10*h
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/4)
				if h_new < 0.2*h:	# limit how fast the step size decreases
					h_new = 0.2*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
		t = t + h
		x_ = np.vstack((x_,z_np1.reshape(1, Ndim))) # add x_n+1 to the array of x values
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		if True: 
			print("\r" + "RKF45" + "\t" + "integrated {:.1%}".format((t-t_start)/(t_stop-t_start)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T, RHS)
	
def Magnus_Integrator(t_start, t_stop, h0, x0, Method, epsilon):
	# An adaptive stepsize integrator for magnus methods
	T_0 = time.time()
	RHS=0
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.05
	h_max = 10
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	#
	M = Method["M"]
	order = Method["order"]
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using one step of h & two steps of h/2
			x_np1_0 = M(t, t+h) @ x_n
			x_np1_l = M(t+0.5*h, t+h) @ (M(t, t+0.5*h) @ x_n)
			#x_np1_l = M(t+0.75*h, t+h) @ M(t+0.25*h, t+0.5*h) @ (M(t, t+0.25*h) @ x_n)
			# compute error
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
			#

			if Method["fname"] == "JM1lk_num":
				RHS+=6
			elif Method["alpha"] == "alpha_GL":
				RHS+=4
			elif Method["alpha"] == "alpha_D":
				RHS+=2
			elif Method["alpha"] == "alpha_SNC":
				RHS+=3
			else:
				RHS+=6

			#
			if Err_ratio <= 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/(order + 1)) # h*1.5
				if h_new > 10*h:	# limit how fast the step size can increase
					h_new = 10*h
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = h*S*np.power(np.abs(Err_ratio), -1.0/(order))
				if h_new < 0.2*h:	# limit how fast the step size decreases
					h_new = 0.2*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
		t = t + h
		x_ = np.vstack((x_,x_np1_l.reshape(1, Ndim))) # add x_n+1 to the array of x values
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		if True: 
			print("\r" + Method["fname"] + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T, RHS)
 
'''
n_steps = 250
t_vec = np.linspace(t_start, t_stop, n_steps)
t_vec0 = np.linspace(t_start, t_stop, 1000)
'''


		
def FixedStepIntegrator_Magnus(t_start, t_stop,n_steps, x0, M):
	#A Fixed step integrator method for magnus methods
	t_vec = np.linspace(t_start, t_stop, n_steps)
	t_vec0 = np.linspace(t_start, t_stop, 500)
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	print(x0)
	Ndim = x0.size
	x0=np.ravel(x0)
	#x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x = np.zeros((len(t_vec), Ndim)) # set up the array of x values
	print(x0)
	#x[0, :] = x0.reshape(Ndim)
	x[0, :] = np.ravel(x0)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		M_ = M(t0, t)
		x[n,:] = (M_ @ x0).reshape(Ndim)
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_vec, x, T)



def FixedStepIntegrator_Spectral3(t_start, t_stop, n_steps, x0,epsres,nini,nmax):
	#A Fixed step integrator method for spectral methods using new cheb function
	t_vec = np.linspace(t_start, t_stop, n_steps)
	t_vec0 = np.linspace(t_start, t_stop, 100)
	T_0 = time.time()
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	m=A_num(t_start).shape[0]
	#w = W2_from_A(A_num)
	#w = lambda t: repeat_along_diagonal(np.diag(t), m)
	A=A_num

	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	h=(t_stop-t_start)/n_steps
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12
	RHS=0

	Ndim=x0.size
	dx0 = A_num(t_start) @ x0.reshape(Ndim, 1)
	ui = x0
	dui = dx0

	uprev=ui
	duprev=dui

	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	Ndim = x0.size
	x = np.zeros((len(t_vec), Ndim)) # set up the array of x values

	#x[0, :] = x0.reshape(Ndim)
	x[0, :] = np.ravel(x0)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		tprev = float(t_vec[n-1])
		print('y0 shape={}'.format(uprev.shape))
		print('nonoxc_step3 = {}'.format(nonosc_step3(A, tprev, h, uprev, duprev,nini,nmax,epsres)))
		x[n,:], duprev, maxerr, status, *misc = nonosc_step3(A, tprev, h, uprev, duprev,nini,nmax,epsres)
		uprev=x[n,:]
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_vec, x, T, RHS)


def FixedStepIntegrator_Spectral4(t_start, t_stop, n_steps, x0,nini):
	#A Fixed step integrator method for spectral methods using new cheb function
	t_vec = np.linspace(t_start, t_stop, n_steps)
	##T_0 = time.time()
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	m=A_num(t_start).shape[0]
	A=A_num

	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	h=(t_stop-t_start)/(n_steps-1)
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12
	RHS=0

	Ndim=x0.size
	dx0 = A_num(t_start) @ x0.reshape(Ndim, 1)
	ui = x0
	dui = dx0

	uprev=ui
	duprev=dui

	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	Ndim = x0.size
	x = np.zeros((len(t_vec), Ndim)) # set up the array of x values

	#x[0, :] = x0.reshape(Ndim)
	x[0, :] = np.ravel(x0)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		tprev = float(t_vec[n-1])
		
		x[n,:], duprev, *misc = nonosc_step4(A, tprev, h, uprev, duprev,nini)
		uprev=x[n,:]
		#print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	##T = time.time() - T_0
	#print(" done in {:.5g}s".format(T))
	return (t_vec, x, 0, RHS)


def Spectral_Integrator(t_start, t_stop, h0, x0,nini,epsres):
	# A stepsize adaptive integrator for spectral methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	nmax=100  # Redundant variable
	epsilon	= 0.005
	rtol = 1		# rel. error tolerance for spectral in units of ε
	atol = 0.005	# abs. error tolerance for spectral in units of ε

	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.005 
	h_max = 2.5
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	m=A_num(t_start).shape[0]
	w = W2_from_A(A_num)
	g = np.zeros(np.shape(A_num))
	w = lambda t: repeat_along_diagonal(np.diag(t), m)

	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.
	info = solversetup(w, g,nini,nmax,h)


	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12
	# What's this?

	Ndim=x0.size
	dx0 = A_num(t_start) @ x0.reshape(Ndim, 1)
	ui = x0
	dui = dx0

	uprev=ui
	duprev=dui

	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			#x[n,:], duprev, maxerr, status, RHSNew = nonosc_step(info, tprev, h, uprev, duprev,epsres)
			# compute the predictions using one step of h & two steps of h/2
			x_np1_0, duprev, maxerr, status, *misc = nonosc_step(info, t, h, uprev, duprev,epsres)

			
			unext0_5, dunext0_5, maxerr0_5, status0_5, *misc0_5  = nonosc_step(info, t, h/2, uprev, duprev,epsres)
			x_np1_l, dunext1, maxerr1, status1, *misc1 = nonosc_step(info, t+h/2, h/2, unext0_5, dunext0_5,epsres)

			#order=N

			#x_np1_0=uprev
			#x_np1_l=duprev
			# compute error
			'''
			Err =  np.mean(ferr(x_np1_0, x_np1_l)/x_np1_l)
			Err_max = epsres
			'''
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
			#

			if Err_ratio <= 1:
				h_new = h*2
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = 0.5*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
		t = t + h
		x_ = np.vstack((x_,x_np1_l.reshape(1, Ndim))) # add x_n+1 to the array of x values
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		if True: 
			print("\r" + "spectral" + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)

def Spectral_Integrator2(t_start, t_stop, h0, x0,nini,epsres,tolerance):
	# A stepsize adaptive integrator for spectral methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	nmax=100  # Redundant variable
	#epsilon	= 0.005
	epsilon = epsres
	rtol = 1		# rel. error tolerance for spectral in units of ε
	atol = 0.005	# abs. error tolerance for spectral in units of ε

	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	#h = h0
	#h=1/np.linalg.norm(A_num(t_start))
	h=0.5
	h_min = 0.5
	h_max = 10
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	m=A_num(t_start).shape[0]
	A=A_num
	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12

	Ndim=x0.size
	dx0 = A_num(t_start) @ x0.reshape(Ndim, 1)
	ui = x0
	dui = dx0

	uprev=ui
	duprev=dui

	#
	while t <= t_stop:
		Err_small = False
		h_new = h
		while Err_small == False:


			x_np1_0, dunext0,*misc = nonosc_step4(A, t, h, uprev, duprev,nini)
			
			x_np1_l, duprev,*misc  = nonosc_step4(A, t, h, uprev, duprev, 2*nini)
			
			#order=N

			#x_np1_0=uprev
			#x_np1_l=duprev
			# compute error
			'''
			Err =  np.mean(ferr(x_np1_0, x_np1_l)/x_np1_l)
			Err_max = epsres
			'''
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
			Err_Simple=np.max(np.abs(Err/x_np1_l))
			#
			'''
			if Err_ratio <= 1:
				h_new = h*2
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = 0.5*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new


			'''
			sum_of_inverse_norms= 0
			for i in range(1,nini+1):
				sum_of_inverse_norms+=np.abs(1/np.linalg.norm(A_num(t+i*h/(nini)))-1/np.linalg.norm(A_num(t+(i-1)*h/(nini))))


			if Err_Simple <= epsres and sum_of_inverse_norms<=tolerance:
				print('Small Errors =',Err_Simple <= epsres and sum_of_inverse_norms<=tolerance)
				h_new = h*1.5
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_Simple > epsres or sum_of_inverse_norms > tolerance:
				print('Big Errors =',Err_Simple > epsres or sum_of_inverse_norms > tolerance)
				print('sum_of_inverse_norms =',sum_of_inverse_norms)
				h_new = 0.5*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
			
		
		print('Err_Simple =',Err_Simple)
		print('h_new =',h_new)
		t = t + h
		x_ = np.vstack((x_,x_np1_0.reshape(1, Ndim))) # add x_n+1 to the array of x values
		uprev=x_np1_l
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		#if True: 
			#print("\r" + "spectral" + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)


def Spectral_Integrator3(t_start, t_stop, h0, x0,nini,epsres,tolerance):
	# A stepsize adaptive integrator for spectral methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	nmax=100  # Redundant variable
	#epsilon	= 0.005
	epsilon = epsres
	rtol = 1		# rel. error tolerance for spectral in units of ε
	atol = 0.005	# abs. error tolerance for spectral in units of ε

	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	#h = h0
	#h=1/np.linalg.norm(A_num(t_start))
	#h=100/np.linalg.norm(A_num(t_start))
	h=h0
	N=nini
	h_min = 0.05
	h_max = 10
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	m=A_num(t_start).shape[0]
	A=A_num
	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12

	Ndim=x0.size
	dx0 = A_num(t_start) @ x0.reshape(Ndim, 1)
	ui = x0
	dui = dx0

	uprev=ui
	duprev=dui
	
	#
	while t <= t_stop:
		Err_small = False
		h_new = h
		N_new = N
		while Err_small == False:


			x_np1_0, dunext0,*misc = nonosc_step4(A, t, h, uprev, duprev,N)
			
			x_np1_l, duprev,*misc  = nonosc_step4(A, t, h, uprev, duprev, 2*N)
			
			#order=N

			#x_np1_0=uprev
			#x_np1_l=duprev
			# compute error
			'''
			Err =  np.mean(ferr(x_np1_0, x_np1_l)/x_np1_l)
			Err_max = epsres
			'''
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
			Err_Simple=np.mean(np.abs(Err/x_np1_l))
			#
			'''
			if Err_ratio <= 1:
				h_new = h*2
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_ratio > 1:
				h_new = 0.5*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new


			'''
			sum_of_inverse_norms= 0
			for i in range(1,nini+1):
				sum_of_inverse_norms+=np.abs(1/np.linalg.norm(A_num(t+i*h/(nini)))-1/np.linalg.norm(A_num(t+(i-1)*h/(nini))))


			if Err_Simple <= epsres and sum_of_inverse_norms<=tolerance:
				print('Small Errors =',Err_Simple <= epsres and sum_of_inverse_norms<=tolerance)
				h_new = h*1.1
				if h_new > h_max:	# limit the maximum step size
					h_new = h_max
				Err_small = True # break loop
			elif Err_Simple > epsres or sum_of_inverse_norms > tolerance:
				print('Big Errors =',Err_Simple > epsres or sum_of_inverse_norms > tolerance)
				print('sum_of_inverse_norms =',sum_of_inverse_norms)
				h_new = 0.9*h
				if h_new < h_min:	# limit the minimum step size
					h_new = h_min
					Err_small = True # break loop
				elif h_new >= h_min:
					h = h_new
			
		
		print('Err_Simple =',Err_Simple)
		print('h_new =',h_new)
		t = t + h
		x_ = np.vstack((x_,x_np1_0.reshape(1, Ndim))) # add x_n+1 to the array of x values
		uprev=x_np1_l
		t_ = np.append(t_, t) 						  # add t_n+1 to the array of t values
		n = n + 1
		h = h_new
		print('h= ',h)
		#if True: 
			#print("\r" + "spectral" + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)




import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{pgf}']





def Plot_RHS_Evaluations_Tolerance(lines):
	# function for plotting a graph of the tolerance of an integration method over one step against the size of the step.
	
	# function for plotting a graph of the results.
	h0 = 0.5
	
	if Eq == Airy:
		MKR_size = 6	# marker size
	elif Eq == burst:
		MKR_size = 6	# marker size
	log_h = True
	log_t = False
		
	######## Integration ##################
	
	t_start = Eq["t_start"]
	t_stop = Eq["t_stop"]
	
	epsilonArray=np.logspace(-4,-2,10)

	for M in lines:
		M['RHS']=np.zeros((len(epsilonArray), 1))
		for i in range(len(epsilonArray)):	
			if M["fname"] == "RKF45":
				M["RHS"][i] = RKF45_Integrator(t_start, t_stop, h0, Eq["x0"], Eq["A_num"], epsilonArray[i])[3]
			elif M["fname"] == "SpectralF":
				M["RHS"][i] = FixedStepIntegrator_Spectral(t_start, t_stop,100, Eq["x0"], epsilonArray[i])[3]
			else:
				M["RHS"][i] = Magnus_Integrator(t_start, t_stop, h0, Eq["x0"], M, epsilonArray[i])[3]
			#	M["data"] = FixedStepIntegrator_Magnus(t_start, t_stop,1000, Eq["x0"], M["M"])	
		t_vec0 = np.linspace(t_start, t_stop, 500)
	x_true = Eq["true_sol"](t_vec0)
	

	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames  
	
	MKR_size = 2	# marker size
	colours = ["c", 'g', 'y', 'b', 'm', 'r']
	m_facecolours = ["c", 'g', 'y', 'b', '1', 'r']
	markertypes = ['x', '.', '.', '+', '^', 'o']
	marker_size = [MKR_size, MKR_size, MKR_size, MKR_size, MKR_size+1, MKR_size]
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
	
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	fig = plt.gcf()
	fig.set_size_inches(3.5,2)
	font_size = 4
	title_font_size = 6
	label_size = 5
	legend_font_size = 2.5
	rc('xtick',labelsize=font_size)
	rc('ytick',labelsize=font_size)
	
	#ax0.plot(h, x_true[:,Index], color="0", linewidth=1, linestyle="--", label="true soln")

	for i in range(0, len(lines)):
		line = lines[i]
		t = np.log10(epsilonArray)
		x = np.log10(line["RHS"])
		print(x)
		#T = line["data"][2]
		T=1
		ax0.plot(t, x, markertypes[i], markerfacecolor=m_facecolours[-i], markeredgecolor=colours[-i], markersize=marker_size[i], linewidth=1, label="{:s}".format(line["name"]))

	#ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	#ax0.set_xlim(0, t[-1])
	ax0.minorticks_on()
	ax0.set_ylabel("Log(Function Evaluations)", fontsize=label_size)
	ax0.set_xlabel("$Log(\epsilon)$", fontsize=label_size)

	ax0.minorticks_on()
	ax0.set_title(Eq["title"]+" : Comparing Cost Against Tolerance", y=1.09, fontsize=title_font_size)
	if Eq == Airy:
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best',ncol = 2, shadow=False)
	elif Eq == burst:
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best', ncol = 2, shadow=False)
	'''for i in range(0, len(lines)+1):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)'''
	

	#ax1.minorticks_on()
	# remove last tick label for the second subplot
	#plt.setp(ax1.get_yticklabels()[-2], visible=False) 

	if Eq == Airy:	
		plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	ax0.tick_params(axis='y', labelsize=font_size)
	

	
	ax0.tick_params(axis='x', labelsize=font_size)
	


	print("made plot")
	plt.savefig("Figure RHS.png", bbox_inches='tight', dpi=1000)




def Plot_Error_Collocation():
	# function for plotting a graph of the error of an integration method over one step against the size of the step.
	M=SpectralFixed
	if Eq == Airy:
		MKR_size = 4	# marker size
	else:
		MKR_size = 3	# marker size
	log_h = True
	log_t = False
	
	######## Integration ##################
	
	t_start = Eq["t_start"]
	#t_start=0
	t_stop = Eq["t_stop"]
	#h=np.arange(0.01,0.11,0.001)

	Repetitions=1
	harray=[0.01,0.1,1]

	niniArray=np.arange(2,60,2)
	#niniArray=np.power(10, np.linspace(np.log10(1), np.log10(80), 40)).astype(int)

	M['data']=np.zeros((len(harray),len(niniArray)))
	M['Time']=np.zeros((len(harray),len(niniArray)))

	for k in range(len(harray)):
		h=harray[k]
		#for i in range(len(h)):
		for j in range(len(niniArray)):
			if M["fname"] == "SpectralF":
				tic=time.time()
				M["data"][k][j] = sum(FixedStepIntegrator_Spectral4(t_start, t_start+h, 2, Eq["true_sol"](np.array([t_start])),niniArray[j])[1][1,Index] for i in range(Repetitions))/Repetitions
				toc=time.time()
				M["Time"][k][j] = (toc-tic)/Repetitions
				#M["Time"][j] = sum(FixedStepIntegrator_Spectral4(t_start, t_start+h,2, Eq["true_sol"](np.array([t_start])),niniArray[j])[2] for i in range(Repetitions))/Repetitions

	
	######## Plotting ##################
	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames  
	MKR_size = 2	# marker size
	colours = ["r", 'g', 'y', 'b', 'm', 'c']
	m_facecolours = ["c", 'g', 'y', 'b', '1', 'r']
	markertypes = ['x', '.', '+', '+', '^', 'o']
	marker_size = [MKR_size, MKR_size, MKR_size, MKR_size, MKR_size, MKR_size]
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(1, 1, height_ratios=[1])
	gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1], sharex = ax0)
	
	fig = plt.gcf()
	fig.set_size_inches(6,3)
	font_size = 6
	title_font_size = 9
	label_size = 6
	legend_font_size = 5
	rc('xtick',labelsize=font_size)
	rc('ytick',labelsize=font_size)


	for i in range(0, len(harray)):
		line = M
		#t = h
		h=harray[i]
		t = np.log10(niniArray)
		x = line["data"][i].reshape(t.size, 1)
		x_true = (Eq["true_sol"](np.array([t_start+h]))[:,Index])
		#error = np.log10(np.abs((x - x_true)/x_true))
		print('x=',x)
		print('x_true=',x_true)
		error = np.log10(np.abs((x - x_true)/x_true))
		#T = line["data"][2]
		ax0.plot(t, error, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))
		ax1.plot(t, np.log10(line["Time"][i]), markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))


	ax0.minorticks_on()
	ax0.set_ylabel("Log(Relative Error)", fontsize=label_size)

	ax1.set_xlabel("Log(p)", fontsize=label_size)
	ax1.set_ylabel("Log(Runtime/s)", fontsize=label_size)


	ax0.set_title(Eq["title"]+" : Error vs Collocation Points used over one step", y=1.09, fontsize=title_font_size)
	if Eq == Airy:
		#lgnd = ax0.legend(fontsize=legend_font_size, loc='best', bbox_to_anchor=(0.27, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best', ncol = 2, shadow=False)
	else:
		#lgnd = ax0.legend(fontsize=legend_font_size, loc='best', bbox_to_anchor=(0.25, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best', ncol = 2, shadow=False)
	'''for i in range(0, len(lines)+1):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)'''
	

	ax1.minorticks_on()

	
	ax0.tick_params(axis='y', labelsize=label_size)
	

	
	ax1.tick_params(axis='x', labelsize=label_size)
	ax1.tick_params(axis='y', labelsize=label_size)
	
	plt.subplots_adjust(hspace=.0)
	plt.savefig("Triplet Collocation 1.png", bbox_inches='tight', dpi=1000)

	print("made plot")

	print(t)
	print(error)
	
def Plot_Error_StepSize(lines):
	# function for plotting a graph of the error of an integration method over one step against the size of the step.
	
	if Eq == Airy:
		MKR_size = 4	# marker size
	elif Eq == burst:
		MKR_size = 3	# marker size
	log_h = True
	log_t = False
		
	######## Integration ##################
	
	#t_start = Eq["t_start"]
	t_start=0
	nini=[10,16,25]
	h=np.logspace(-1, 1, 50)
	Repetitions = 1
	
	for M in lines:
		if M["fname"] != "SpectralF":
			M['data']=np.zeros(len(h))
			M['Time']=np.zeros(len(h))
		else:
			M['data']=np.zeros((len(nini),len(h)))

		for i in range(len(h)):
			if M["fname"] == "RKF45":
				M["data"][i] = RKF45_Integrator(t_start, t_start+h[i], h[i], Eq["x0"], Eq["A_num"],epsilon_RK=1e4)[1][1,Index]
				#M["Time"][i] = RKF45_Integrator(t_start, t_start+h[i], h[i], Eq["x0"], Eq["A_num"],epsilon_RK=0.005)[2]
			elif M["fname"] == "SpectralF":
				for k in range(len(nini)):
					#M["data"][i] = FixedStepIntegrator_Spectral(t_start, t_start+h[i],2, Eq["true_sol"](np.array([t_start])))[1][1,Index]
					M["data"][k][i] = FixedStepIntegrator_Spectral4(t_start, t_start+h[i], 2, Eq["true_sol"](np.array([t_start])),nini[k])[1][1,Index]
					#M["Time"][i] = FixedStepIntegrator_Spectral4(t_start, t_start+h[i], 2, Eq["true_sol"](np.array([t_start])),nini)[2]
			else:
				M["data"][i] = FixedStepIntegrator_Magnus(t_start, t_start+h[i],2, Eq["true_sol"](np.array([t_start])), M["M"])[1][1,Index]
				#M["Time"][i] = FixedStepIntegrator_Magnus(t_start, t_start+h[i],2, Eq["true_sol"](np.array([t_start])), M["M"])[2]
	
	
	######## Plotting ##################
	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames  
	
	colours = ["c", 'g', 'y', 'b', 'm', 'r']
	m_facecolours = ["c", 'g', 'y', 'b', '1', 'r']
	markertypes = ['.', 'x', '^', '+', '^', 'o']
	marker_size = [MKR_size, MKR_size, MKR_size, MKR_size, MKR_size+3, MKR_size]
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(1, 1, height_ratios=[1])
	gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1], sharex = ax0)
	
	fig = plt.gcf()
	fig.set_size_inches(21,12)
	font_size = 8
	title_font_size = 12
	label_size = 10
	legend_font_size = 8
	rc('xtick',labelsize=font_size)
	rc('ytick',labelsize=font_size)


	for i in range(0, len(lines)):
		line = lines[i]
		if line["fname"] != "SpectralF":
			t = np.log10(h)
			x = line["data"].reshape(t.size, 1)
			x_true = (Eq["true_sol"](np.array([t_start+h]))[:,Index]).reshape(t.size, 1)
			print('x_true=',x_true)
			#error = np.log10(np.abs((x - x_true)/x_true))
			error = np.log10(np.abs((x - x_true)))
			T = line["data"][2]
			#ax0.plot(t, error, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))
			#ax1.plot(t, np.log10(line["Time"]), markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))
			ax0.plot(t, error, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1,label="{:s}".format(line["name"]))
			#ax1.plot(t, np.log10(line["Time"]), markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1)
		else: 
			for k in range(len(nini)):
				t = np.log10(h)
				x = line["data"][k].reshape(t.size, 1)
				x_true = (Eq["true_sol"](np.array([t_start+h]))[:,Index]).reshape(t.size, 1)
				print('x_true=',x_true)
				#error = np.log10(np.abs((x - x_true)/x_true))
				error = np.log10(np.abs((x - x_true)))
				T = line["data"][2]
				#ax0.plot(t, error, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))
				#ax1.plot(t, np.log10(line["Time"]), markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, h={:.4g}".format(line["name"], h))
				ax0.plot(t, error, markertypes[i], markerfacecolor=m_facecolours[k], markeredgecolor=colours[k], markersize=marker_size[i], linewidth=1,label="{:s}, nini={:.4g}".format(line["name"], nini[k]))
				#ax1.plot(t, np.log10(line["Time"]), markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1)

	#ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	#ax0.set_xlim(0, t[-1])
	ax0.minorticks_on()
	ax0.set_ylabel("$log(Absolute Error)$", fontsize=label_size)
	#ax0.set_xlabel("$StepSize$", fontsize=label_size)
	ax0.set_xlabel("$log(h)$", fontsize=label_size)
	ax1.set_ylabel("$log(Runtime/s)$", fontsize=label_size)

	ax0.minorticks_on()
	ax0.set_title(Eq["title"]+" : comparing different methods", y=1.09, fontsize=title_font_size)
	if Eq == Airy:
		lgnd = ax0.legend(fontsize=legend_font_size, loc='upper right', bbox_to_anchor=(0.27, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
	elif Eq == burst:
		lgnd = ax0.legend(fontsize=legend_font_size, loc='upper right', bbox_to_anchor=(0.25, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
	'''for i in range(0, len(lines)+1):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)'''
	

	#ax1.minorticks_on()
	# remove last tick label for the second subplot
	#plt.setp(ax1.get_yticklabels()[-2], visible=False) 
	if Eq == Airy:	
		plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	ax0.tick_params(axis='y', labelsize=font_size)
	

	
	ax0.tick_params(axis='x', labelsize=font_size)
	


	print("made plot")
	plt.savefig("Figure StepSize.png", bbox_inches='tight', dpi=1000)
	print(t)
	print(error)
	

def plot_graph():
	# function for plotting a graph of the results.
	h0 = 0.5
	
	if Eq == Airy:
		MKR_size = 4	# marker size
	elif Eq == burst:
		MKR_size = 3	# marker size
	else:	
		MKR_size = 3
		
	log_h = True
	log_t = False
		
	######## Integration ##################
	
	t_start = Eq["t_start"]
	t_stop = Eq["t_stop"]
	#t_start =0
	#t_stop = 10
	
	for M in lines:
		if M["fname"] == "RKF45":
			M["data"] = RKF45_Integrator(t_start, t_stop, h0, Eq["true_sol"](np.array([t_start])), Eq["A_num"],epsilon_RK=0.005)
		elif M["fname"] == "SpectralF":
			#M["data"] = FixedStepIntegrator_Spectral(t_start, t_stop,100, Eq["x0"],epsres=1e-4,8,128)
			#M["data"] = FixedStepIntegrator_Spectral(t_start, t_stop,100, Eq["true_sol"](np.array([t_start])),1e-4,32,128)
			#M["data"] = FixedStepIntegrator_Spectral2(t_start, t_stop,100, Eq["true_sol"](np.array([t_start])),1e-8,8)
			#M["data"] = FixedStepIntegrator_Spectral3(t_start, t_stop,200, Eq["true_sol"](np.array([t_start])),1e-8,32,64)
			M["data"] = FixedStepIntegrator_Spectral4(t_start, t_stop,200, Eq["true_sol"](np.array([t_start])),12)
		elif M["fname"] == "SpectralA":
			#M["data"] = Spectral_Integrator2(t_start, t_start+1*(t_stop-t_start), h0, Eq["true_sol"](np.array([t_start])),8,1e-0,1e-1)
			M["data"] = Spectral_Integrator3(t_start, t_start+1*(t_stop-t_start), h0, Eq["true_sol"](np.array([t_start])),8,1e-2,1e-2)
		else:
			M["data"] = Magnus_Integrator(t_start, t_stop, h0, Eq["true_sol"](np.array([t_start])), M,epsilon=0.005)
		#	M["data"] = FixedStepIntegrator_Magnus(t_start, t_stop,1000, Eq["x0"], M["M"])	
	t_vec0 = np.linspace(t_start, t_stop, 500)
	x_true = Eq["true_sol"](t_vec0)
	
	######### Plotting ####################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames  
	

	MKR_size = 2	# marker size
	colours = ["c", 'g', 'y', 'b', 'm', 'r']
	m_facecolours = ["c", 'g', 'y', 'b', '1', 'r']
	markertypes = ['.', 'x', '^', '+', '^', 'o']
	marker_size = [MKR_size, MKR_size, MKR_size, MKR_size, MKR_size+1, MKR_size]
	
	# set height ratios for sublots
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
	
	################ Primary plot & error plot
	ax0 = plt.subplot(gs[0])
	fig = plt.gcf()
	fig.set_size_inches(8,5)
	font_size = 8
	title_font_size = 10
	label_size = 10
	legend_font_size = 8
	rc('xtick',labelsize=font_size)
	rc('ytick',labelsize=font_size)
	
	ax0.plot(t_vec0, x_true[:,Index], color="0", linewidth=1, linestyle="--", label="true soln")
	#ax0.plot(t_test, x_scipy_test[:,0], color="g", linewidth=1, linestyle="--", label="scipy.integrate.odeint()")
	#
	ax2 = plt.subplot(gs[2], sharex = ax0)
	ax2.plot(np.linspace(Eq["t_start"], t_stop, 20), np.log10(epsilon*np.ones(20)), color="k", linewidth=1, linestyle=":", label="$\epsilon$")
	#ax2.annotate("$\epsilon$", xy=(1.05*Eq["t_stop"], epsilon))
	#
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = (line["data"][1][:,Index]).reshape(t.size, 1)
		x_true = (Eq["true_sol"](t)[:,Index]).reshape(t.size, 1)
		error = np.log10(np.abs((x - x_true)/x_true))
		T = line["data"][2]
		if line==SpectralFixed:
			print(x)
		ax0.plot(t, x, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		ax2.plot(t, error, '--', color=colours[i], linewidth=1, alpha=1)
	##ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	ax0.set_xlim(t_start, t_stop)
	#ax2.set_xlim(Eq["t_start"], t_stop)
	#ax2.legend()
	ax0.set_ylabel("$x$", fontsize=label_size)
	ax2.set_ylabel("log$_{10}$(rel. error)", fontsize=font_size)
	ax2.set_xlabel("$t$", fontsize=label_size)
	ax2.minorticks_on()
	ymin, ymax = ax2.get_ylim()
	if ymax>1:
		ax2.set_ylim(top=1)
	ax0.minorticks_on()
	ax0.set_title(Eq["title"]+" : Comparing Integration Methods", y=1.09, fontsize=title_font_size)
	if Eq == Airy:
		#lgnd = ax0.legend(fontsize=legend_font_size, loc='upper right', bbox_to_anchor=(0.27, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best',  ncol = 2, shadow=False)
	elif Eq == burst:
		#lgnd = ax0.legend(fontsize=legend_font_size, loc='upper right', bbox_to_anchor=(0.25, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best', ncol = 2, shadow=False)
	else:
		#lgnd = ax0.legend(fontsize=legend_font_size, loc='upper right', bbox_to_anchor=(0.25, 0.88, 0.50, 0.25), ncol = 2, shadow=False)
		lgnd = ax0.legend(fontsize=legend_font_size, loc='best', ncol = 2, shadow=False)

	'''for i in range(0, len(lines)+1):
		lgnd.legendHandles[i]._legmarker.set_markersize(5)'''
	
	################ Stepsize plot
	# shared axis X
	ax1 = plt.subplot(gs[1], sharex = ax0)
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		h = t[1:] - t[:-1]
		t_av = 0.5*(t[1:] + t[:-1])
		if log_h:
			ax1.plot(t_av, np.log10(h), '-', color=colours[i], linewidth=1, label="{:s}".format(line["name"]))
		elif log_h == False:
			ax1.plot(t_av, h, '-', color=colours[i], linewidth=1, label="{:s}".format(line["name"]))
	if log_h:
		ax1.set_ylabel("$\\log_{10}$($h$)", fontsize=label_size)
		#savename = "Plots/" + filename + "_log_h.pdf"
	elif log_h == False:
		ax1.set_ylabel("h", fontsize=font_size)
		#savename = "Plots/" + filename + ".pdf"
	
	#ax1.set_xlim(Eq["t_start"], t_stop)
	
	if log_t:
		ax0.set_xscale('log')
		ax1.set_xscale('log')
		ax2.set_xscale('log')
		savename = "Plots/" + filename + "_log_t.pdf"
	elif not log_t:
		savename = "Plots/" + filename + ".pdf"

	ax1.minorticks_on()
	# remove last tick label for the second subplot
	#plt.setp(ax1.get_yticklabels()[-2], visible=False) 
	if Eq == Airy:	
		plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	ax0.tick_params(axis='y', labelsize=font_size)
	ax1.tick_params(axis='y', labelsize=font_size)
	ax2.tick_params(axis='y', labelsize=font_size)
	ax2.tick_params(axis='x', labelsize=font_size)
	
	plt.subplots_adjust(left=0.12, right=0.92)
	
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.subplots_adjust(hspace=.0)
	#plt.savefig(savename, transparent=True)
	#plt.clf()
	print("made plot")
	print("saved as " + savename)
	plt.savefig("Burst JM.png", bbox_inches='tight', dpi=1000)