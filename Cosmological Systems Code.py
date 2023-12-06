import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{pgf}']

# Magnus Solver

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

#import matplotlib
#matplotlib.use("Agg")
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

# -- PhotonCDM equation stuff -- #

eta_0 = 0.01
k = 200

Og0 = 0.5 
Ob0 = 0.5

variable = "Phi"

def Apcdm():
	A = sym.Matrix(
		[[-2*Og0/ts, -k, -Ob0/2, 0, -ts*(k**2)/3 - 1/ts], 
		 [k/3, 0, 0, 0, k/3],
		 [-6*Og0/ts, 0, -3*Ob0/2, -sym.I*k, -ts*k**2 - 3/ts],
		 [0, 0, 0, -1/(ts**2), -sym.I*k],
		[-2*Og0/ts, 0, -Ob0/2, 0, -ts*(k**2)/3 - 1/ts]])
	return A
	
A_sym_pcdm = Apcdm()
	
A_num_pcdm = sym.lambdify((ts), A_sym_pcdm, modules=array2mat_c)
	
PhotonCDM = {}
PhotonCDM["x0"] = np.array([1, 2, 1, 2, 1])
x0_string = str(PhotonCDM["x0"][0])
for i in range(1,5):
	x0_string = x0_string + "_" + str(PhotonCDM["x0"][i])

if variable == "Phi":
	Index = 4	# index of x_i variable to plot
	PhotonCDM["name"] = "PhotonCDM_Phi_k=" + str(k) + "_x0=" + x0_string
	PhotonCDM["title"] = "Radiation dominated photon and CDM system : $\\Phi$, $k = $" + str(k)
	PhotonCDM["ylim"] = (-0.2, 1.00)
elif variable == "Theta0":
	Index = 0	# index of x_i variable to plot
	PhotonCDM["name"] = "PhotonCDM_Theta0_k=" + str(k) + "_x0=" + x0_string
	PhotonCDM["title"] = "Radiation dominated photon and CDM system : $\\Theta_0$, $k = $" + str(k)
	PhotonCDM["ylim"] = (-5, 7)
	
PhotonCDM["A_sym"] = A_sym_pcdm
PhotonCDM["A_num"] = A_num_pcdm
PhotonCDM["t_start"] = eta_0
PhotonCDM["t_stop"] = 0.5

PhotonCDM["Theta0_sol_coeff"] = [0, -1]
PhotonCDM["Phi_sol_coeff"] = [0.5, -0.5]
	
def PhotonCDM_Theta0_sol(t, a, b):
	# define the solution for Theta0
	C = a*np.cos(k*t/np.sqrt(3)) + b*np.sin(k*t/np.sqrt(3))
	return C
	
def PhotonCDM_Phi_sol(t, a, b):
	t0 = PhotonCDM["t_start"]
	
	q = k*t/np.sqrt(3)
	q0 = k*t0/np.sqrt(3)
	
	J0 = special.spherical_jn(1,q0)/(q0)
	Y0 = special.spherical_yn(1,q0)/(q0)
	J = special.spherical_jn(1,q)/q
	Y = special.spherical_yn(1,q)/q
	
	C = a*(J/J0) + b*(Y/Y0)
	return C
	
def PhotonCDM_true_sol(t):
	x = np.ones((t.size, 5))
	a0, b0 = PhotonCDM["Theta0_sol_coeff"]
	a4, b4 = PhotonCDM["Phi_sol_coeff"]
	x[:,0] = PhotonCDM_Theta0_sol(t, a0, b0)
	x[:,4] = PhotonCDM_Phi_sol(t, a4, b4)
	return x

PhotonCDM["true_sol"] = PhotonCDM_true_sol
# ---------------------------- #


################### Choose equation #########################

Eq = PhotonCDM

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
	# Cotes quadrature rules using equidistant A(t) points
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

#	Symbolic manipulation using sympy

A_sym = Eq["A_sym"]

print("Symbolic A = ", A_sym)


A_num = Eq["A_num"]
print("Numeric A = ", A_num)	
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
			tic=time.process_time()
			Om = Omega(t0, t)
			return linalg.expm(Om)
		toc=time.process_time()		
		#print('Time to produce stepping matrix (exponentiation) = {}'.format(toc-tic))
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

def Pseudo_WKB(Use_numerics):
	# Pseudo-WKB method
	def Make_func():
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		Aprime = sym.diff(A, ts0) + A*A
		print("A' = ", Aprime)
		print()
		w1_sym = []
		for i in range(0, Ndim):
			w2 = -Aprime[i,i]
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
			M_sym = M11 + M12*A
			print()
			print("Pseudo-WKB matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
		elif Use_numerics == 1:
			# numerical version 
			Ap = sym.lambdify((ts0), Aprime, modules=array2mat)
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
				M_ = (M11 + M12 @ A_num(t0))
				return M_
		return Mf
	return Make_func

def Jordan_Magnus(Lambda_only, Use_numerics):
	def Make_func():
		# symbolics
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		P_, J_ = A.jordan_form()	# compute Jordan Normal form (next best thing to diagonalisation)
		P = sym.simplify(P_)
		J = sym.simplify(J_)
		print("J = ", J)
		print()
		print("P = ", P)
		print()
		Pinv = P
		P = P.inv()
		LK_ = J + sym.diff(Pinv, ts0)*P
		LK = sym.simplify(LK_)
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
		P_num = sym.lambdify((ts0), P, modules=array2mat_c)
		#
		def Mf(t0, t):
			M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ np.linalg.inv(P_num(t0))
			return M_.astype(np.float64)
		return Mf
	return Make_func


def Jordan_Magnus2(Lambda_only, Use_numerics):
	def Make_func():
		def Mf(t0, t):
			A=A_num
			dt=1e-8
			Ndim = A_sym.subs(ts, ts0).shape[0]
			def DecomposeSchur(t,dt,A):
				A00_np = np.array(A(t))
				J00_, P00_ = linalg.schur(A00_np)
				J00 = sym.Matrix(J00_)
				P00 = sym.Matrix(P00_)
				A00dt_np = np.array(A(t0+dt))
				J00dt_, P00dt_ = linalg.schur(A00dt_np)
				P00dt_ = sym.Matrix(P00dt_)
				dP00inv = (P00dt_.inv()-P00.inv())/dt
				return P00, dP00inv, J00
			#t0
			P0, dP0inv, J0 = DecomposeSchur(t0,dt,A)
	 		#t0_5
			P05, dP05inv, J05 = DecomposeSchur(t0+(t-t0)/2,dt,A)
	 		#t
			P, dPinv, J = DecomposeSchur(t,dt,A)
	 
			LK0_ = J0 + 2*dP0inv*P0
			LK0 = sym.simplify(LK0_)

			LK05_ = J05 + dP05inv*P05
			LK05 = sym.simplify(LK05_)
	 		
			LK_ = J + dPinv*P
			LK = sym.simplify(LK0_)
			print("LK = ", LK)

			if Lambda_only:
				# only use the diagonal elements
				LK0 = sym.eye(Ndim).multiply_elementwise(LK0)
				LK05 = sym.eye(Ndim).multiply_elementwise(LK05)
				LK = sym.eye(Ndim).multiply_elementwise(LK)
				print("L = ", LK)
			
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""
			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)

			M_ = P @ linalg.expm(np.array(Om1_num(t0,t)).astype(np.complex128)) @ P0.inv()

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

def DecomposeSchur(t,dt,A):

  tic=time.process_time()
  A00_np = A(t)
  A00dt_np = A(t+dt)

  toc1=time.process_time()
  J00_, P00_ = scipy.linalg.schur(A00_np)
 
  toc2=time.process_time()
  J00dt_, P00dt_ = scipy.linalg.schur(A00dt_np)
 
  toc3=time.process_time()
  a=np.linalg.inv(P00dt_)
  b=np.linalg.inv(P00_)
  toc4=time.process_time()
  dP00inv = (a-b)/dt
  toc5=time.process_time()


  print('Time to convert = {}'.format(toc1-tic))
  print('Time to decompose at t = {}'.format(toc2-toc1))
  print('Time to decompose at t+dt= {}'.format(toc3-toc2))
  print('Time to inverse = {}'.format(toc4-toc3))
  print('Time to calculate derivative = {}'.format(toc5-toc4))
  print('Total time = {}'.format(toc5-tic))
  return P00_, dP00inv, J00_
'''

def DecomposeSchur(t,dt,A):
	A00_np = np.array(A(t))
	J00_, P00_ = linalg.schur(A00_np)
	A00dt_np = np.array(A(t+dt))
	J00dt_, P00dt_ = linalg.schur(A00dt_np)
	dP00inv = (np.linalg.inv(P00dt_)-np.linalg.inv(P00_))/dt
	return P00_, dP00inv, J00_

def JMSchur(Lambda_only, Use_numerics):
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
				LK0 = np.diag(np.diag(LK0))
				LK05 = np.diag(np.diag(LK05))
				LK = np.diag(np.diag(LK))

			toc3=time.process_time()
			"""
			for the moment just use GL quadrature order 4 (?) here
			"""

			Om1_num = Omega_num2(LK0,LK05,LK, t0, t, alpha_SNC2, 4)
			Om1_num = np.array(Om1_num(t0,t)).astype(np.complex64)
			toc4=time.process_time()
	 
			M_ = P @ linalg.expm(Om1_num) @ np.linalg.inv(P0)
	 
			toc5=time.process_time()
	 
			print('Time to schur decompose = {}'.format(toc1-tic))
			print('Time to produce LK (Matrix multiplication) = {}'.format(toc2-toc1))
			print('Time to calculate stepping matrix (matrix multiplication and exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func



def Jordan_Magnus4(Lambda_only, Use_numerics):
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
			M_ = P @ linalg.expm(np.array(Om1_num(t0,t)).astype(np.complex128)) @ Pinv
			toc5=time.process_time()
			print('Time to produce LK (matrix multiplication) = {}'.format(toc1-tic))
			print('Time to calculate first maguns term = {}'.format(toc4-toc3))
			print('Time to calculate stepping matrix (matrix multiplication and exponential) = {}'.format(toc5-toc4))
			return np.array(M_).astype(np.complex128)
		return Mf
	return Make_func
	

def NPDiagonalise(t,dt,A):
	A00_np = np.array(A(t)).astype(np.complex64)
	W00_, V00_ = np.linalg.eig(A00_np)
	#J00 = np.linalg.inv(V00_) @ A00_np @ V00_
	J00 = np.diag(W00_)
	A00dt_np = np.array(A(t+dt)).astype(np.complex64)
	W00dt_, V00dt_ = np.linalg.eig(A00dt_np)
	dP00inv = (np.linalg.inv(V00dt_)-np.linalg.inv(V00_))/dt
	return V00_, dP00inv, J00

def JMNumpyDiag(Lambda_only, Use_numerics):
	def Make_func():
		def Mf(t0, t):
			tic=time.process_time()
			A=A_num
			dt=1e-8
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
	
def T_Jordan_Magnus(Lambda_only, Use_numerics):
	def Make_func():
		"""
		 The Jordan-Magnus method, but approximate P(t) as 
		 P(t) = P(0) + t*P'(0) + 0.5*t^2*P''(0) + ...
		"""
		global started_TJM	# have we started integrating?
		global count_TJM	# how many times have we recalculated P?
		global P_TJM 		# current P estimate
		global Mf_TJM		# current Mf
		#
		def get_P(tp):
			A_t_num = A_num(tp)
			Ndim = A_t_num.shape[0]
			# extract P matrix using np.linalg.
			w, v = linalg.eig(A_t_num)
			P = v
			print("found P")
			return P	
		dt = 0.001*0.001
		def get_diffP(t):
			P1 = get_P(t - 1.5*dt)
			P1inv = np.linalg.inv(P1)
			P2 = get_P(t - 0.5*dt)
			P2inv = np.linalg.inv(P2)
			P3 = get_P(t + 0.5*dt)
			P3inv = np.linalg.inv(P3)
			P4 = get_P(t + 1.5*dt)
			P4inv = np.linalg.inv(P4)
			#
			dP = (P3 - P2)/(dt)
			ddP = (P4 + P1 - P2 - P3)/(2*dt**2)
			dddP = (P4 - 3*P3 + 3*P2 - P1)/(dt**3)
			dPinv = (P3inv - P2inv)/(dt)
			ddPinv = (P4inv + P1inv - P2inv - P3inv)/(2*dt**2)
			dddPinv = (P4inv - 3*P3inv + 3*P2inv - P1inv)/(dt**3)
			P_list = [dP, ddP, dddP, dPinv, ddPinv, dddPinv]
			for P in P_list:
				P = sym.sympify(P).tomatrix()
			print("made P_list")
			return [dP, ddP, dddP, dPinv, ddPinv, dddPinv]
		#	
		def Make_M_from_new_P(tp):
			global P_TJM 	
			P0 = sym.sympify(get_P(tp)).tomatrix()
			Pinv0 = P0.inv()
			dP0, ddP0, dddP0, dPinv0, ddPinv0, dddPinv0 = get_diffP(tp)
			P_sym = P0 + (ts-tp)*dP0 + 0.5*((ts-tp)**2)*ddP0 + (1/6)*((ts-tp)**3)*dddP0
			
			print("P_sym = ", P_sym)
			
			Pinv_sym = Pinv0 + (ts-tp)*dPinv0 + 0.5*((ts-tp)**2)*ddPinv0 + (1/6)*((ts-tp)**3)*dddPinv0
			
			print("Pinv_sym = ", Pinv_sym)

			dPinv_sym = dPinv0 + (ts-tp)*ddPinv0 + 0.5*((ts-tp)**2)*dddPinv0
			
			print("dPinv_sym = ", dPinv_sym)
			
			LK_sym = Pinv_sym @ A_sym @ P_sym + dPinv_sym @ P_sym
			print("LK_sym = ", LK_sym)
			sysexit()
			if Use_numerics == 0:
				if Lambda_only:
					# only use the diagonal elements
					LK_sym = sym.eye(Ndim).multiply_elementwise(LK_sym)
					print("L = ", LK_sym)
					print()
				Om1 = sym.integrate(LK_sym.subs(ts, ts1), (ts1, ts0, ts)) 
				Om1_num = sym.lambdify((ts0, ts), Om1, modules=array2mat_c)
			elif Use_numerics == 1:
				LK_num = sym.lambdify((ts), LK_sym, modules=array2mat_c)
				def Omega_num(t0, t):
					Om = M_quad(LK_num, t0, t, MAXITER=quad_maxiter)
					return Om
				Om1_num = Omega_num
			P_num = sym.lambdify((ts), P_sym, modules=array2mat_c)
			Pinv_num = sym.lambdify((ts), Pinv_sym, modules=array2mat_c)
			P0_num = P_num(tp)
			P_TJM = P_num	# store the P_num function
			def Mf(t0, t):
				M_ = P_num(t) @ linalg.expm(Om1_num(t0, t)) @ Pinv_num(t0)
				return M_ #.astype(np.complex128)
			return Mf
		#
		started_TJM = False
		count_TJM = 0
		def M_func_adaptive(t0, t):
			global started_TJM, count_TJM, P_TJM, Mf_TJM 	
			t_mid = 0.5*(t + t0)
			# get first P matrix at t=t0
			if not started_TJM:
				Mf_TJM = Make_M_from_new_P(t_mid)
				started_TJM = True
			# check 10 times to see if P(t) need to be re-evaluated 
			count = np.floor(10*(t_mid - Eq["t_start"])/((Eq["t_stop"] - Eq["t_start"])))
			# do I need to check if we need to change P?
			if count > count_TJM:
				count_TJM = count
				# check to see if we need to change P
				P_old = P_TJM(t_mid)
				P_new = sym.matrix2numpy(get_P(t_mid), dtype=np.complex128)
				dP_norm = np.linalg.norm(P_new - P_old, ord='fro')
				#print("  count = " + str(count) + ", dP_norm = ", dP_norm)
				if dP_norm > 0:
					#print(P_new)
					#print(" making new P estimate, t_mid = ", t_mid)
					Mf_TJM = Make_M_from_new_P(t_mid)
			M_ = Mf_TJM(t0, t)
			return M_
		return M_func_adaptive
	return Make_func
	
def Ext_Pseudo_WKB(Use_numerics):
	# Extended Pseudo-WKB method
	def Make_func():
		A = A_sym.subs(ts, ts0)
		Ndim = A.shape[0]
		Id_sym = sym.eye(Ndim)
		Aprime = sym.diff(A, ts0) + A*A
		print("A' = ", Aprime)
		print()
		Ainv = A.inv()
		w1d_sym = []
		gamma_sym = []
		for i in range(0, Ndim):
			# extract diagonal elements of various matrices
			Ap_ = Aprime[i, i]
			A_ = A[i, i]
			Ainv_  = Ainv[i, i]
			ApAinv_ = (Aprime @ Ainv)[i, i]
			# 
			w2 = (ApAinv_*A_ - Ap_)/(1 - A_*Ainv_)
			gamma = (Ainv_*Ap_ - ApAinv_)/(1 - A_*Ainv_)
			#w1 = sym.sqrt(w2)
			print("w2 = ", w2)
			print("gamma = ", gamma)
			w1d = sym.sqrt(w2 - (gamma**2)/4)
			w1d_sym.append(w1d)
			gamma_sym.append(gamma)	
		if Use_numerics == 0:
			# symbolic version
			M11 = sym.eye(Ndim)
			M12 = sym.eye(Ndim)
			for i in range(0, Ndim):
				w1d = w1d_sym[i]
				gamma = gamma_sym[i]
				Int_gamma = sym.integrate(gamma.subs(ts0, ts1), (ts1, ts0, ts))
				C = sym.exp(-(1/2)*Int_gamma)*sym.cos(sym.integrate(w1d.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1d/w1d.subs(ts0, ts))
				S = sym.exp(-(1/2)*Int_gamma)*sym.sin(sym.integrate(w1d.subs(ts0, ts1), (ts1, ts0, ts)))*sym.sqrt(w1d/w1d.subs(ts0, ts))
				dw1d = sym.diff(w1d, ts0)
				M11[i,i] = C + S*(gamma/2*w1d + dw1d/(2*w1d**2))
				M12[i,i] = S/w1d
			M_sym = M11 + M12*A
			print()
			print("Ext-Pseudo-WKB matrix = ", M_sym)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym, modules=array2mat)
		elif Use_numerics == 1:
			# numerical version 
			Id = np.identity(Ndim)
			M11 = Id.astype(np.complex64)
			M12 = Id.astype(np.complex64)
			w1d_num = []
			gamma_num = []
			dw1d_num = []
			# convert symbolic forms into numerical functions
			for i in range(0, Ndim):
				w1d_num.append(sym.lambdify((ts0), w1d_sym[i], modules=array2mat_c))
				gamma_num.append(sym.lambdify((ts0), gamma_sym[i], modules=array2mat_c))
				dw1d_num.append(eg(w1d_num[i], 0.00001))
			def Mf(t0, t):
				# define a function to compute the M matrix
				Int_w1d = Id
				for i in range(Ndim):
					w1d = w1d_num[i](t)
					w1d0 = w1d_num[i](t0)
					dw1d0 = dw1d_num[i](t0)
					g0 = gamma_num[i](t0)
					Int_gamma = c_quad(gamma_num[i], t0, t)
					Int_w1d = c_quad(w1d_num[i], t0, t)
					C = np.exp(-0.5*Int_gamma)*np.cos(Int_w1d)*csqrt(w1d0/w1d)
					S = np.exp(-0.5*Int_gamma)*np.sin(Int_w1d)*csqrt(w1d0/w1d)
					M11[i,i] = C + S*(g0/(2*w1d0) + dw1d0/(2*(w1d0)**2))
					M12[i,i] = S/w1d0
				M_ = (M11 + M12 @ A_num(t0))
				return M_
		return Mf
	return Make_func
	
def Modified_M1(Use_numerics, alpha):
	# modified Magnus expansion from Iserles 2002a 
	# "ON THE GLOBAL ERROR OF DISCRETIZATION METHODS ... "
	def Make_func():
		A = A_sym.subs(ts, ts0)
		h = sym.Symbol("h", nonzero=True)
		t_half = sym.Symbol("t_half")
		A_half = A.subs(ts0, t_half)
		"""
		def B(t):
			A_t = A.subs(ts0, t)
			B = sym.exp((ts0 - t)*A_half)*(A_t - A_half)*sym.exp((t - ts0)*A_half)
			return B
		"""
		B_ = sym.exp(-h*A_half)*(A.subs(ts0, ts1) - A_half)*sym.exp(h*A_half)
		B_ = sym.nsimplify(B_)
		B_ = B_.rewrite(sym.cos)
		B_ = sym.simplify(B_)
		print("B = ", B_)
		print()
		if Use_numerics == 0:
			Om = sym.integrate(B_, (ts1, ts0, ts))
			Om_ = Om.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
			print("Om = ", Om_)
			print()
			M_sym = sym.exp(h*A_half)*sym.exp(Om)
			M_sym_ = M_sym.subs({h:ts - ts0, t_half: (1/2)*(ts + ts0)})
			print("Modified Magnus 1 matrix = ", M_sym_)
			print()
			Mf = sym.lambdify((ts0, ts), M_sym_, modules=array2mat_c)
		elif Use_numerics == 1:
			A_half_ = A_half.subs(t_half, (1/2)*(ts0 + ts)) 
			A_half_num = sym.lambdify((ts0, ts), A_half_, modules=array2mat)
			def B_num(t1, t0, t):
				A_t = Eq["A_num"](t1)
				A_h = A_half_num(t0, t)
				B = linalg.expm((t0 - t)*A_h) @ (A_t - A_h) @ linalg.expm((t - t0)*A_h)
				return B
			"""
			B_ = B(ts1)
			B_num = sym.lambdify((ts1, ts0, ts), B_, modules=array2mat_c)
			"""
			def Omega_B_num(t0, t):
				def A(t1):
					A_ = B_num(t1, t0, t)
					return A_
				a_1, a_2 = alpha(t0, t, A, 4)
				Om = a_1 - (1/12)*Com(a_1, a_2)
				return Om
			#	
			def Mf(t0, t):
				M_ = linalg.expm((t-t0)*A_half_num(t0, t)) @ linalg.expm(Omega_B_num(t0, t))
				return M_
		return Mf
	return Make_func


import numpy as np
import time
a = np.diagflat(np.array([[5, 3, 2, 1], [0, 1, -1, -1], [-1, -1, 100, 0], [1, 1, -1, 2]]))
from scipy.sparse import block_diag, spdiags
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
        self.m = np.shape(A_num(1))[0]
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
        m = np.shape(A_num(1))[0]
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


def solversetup(w, g, h0 = 0.1, nini = 64, nmax = 128, n = 4, p = 4):
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
    print("w2:")
    print(w2)
    print("D shape:",D.shape, "w2 shape:", w2.shape)

    D2 = 4/h**2*(D @ D) + w2

    print("D2 shape", (4/h**2*(D @ D)).shape)
    print("D2:", D2)

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

import numpy as np

def nonosc_step(info, x0, h, y0, dy0, epsres = 1e-3):
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
    while maxerr > epsres:
        N *= 2
        if N > Nmax:
            success = 0
            return y, dy, maxerr, success
        y, dy, x = spectral_cheb(info, x0, h, y0, dy0, int(np.log2(N/info.nini))) 
        maxerr = np.abs((yprev[0] - y[0])/y[0])
        if np.isnan(maxerr):
            maxerr = np.inf
        yprev = y
        dyprev = dy
        xprev = x
    info.increase(chebstep = 1)
    if info.denseout:
        # Store interp points
        info.yn = y
        info.dyn = dy
    return y, dy, maxerr, success


#Eq = Airy

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

M6_SNC = {
	"name" : "Magnus 6$^\\circ$, NC quad",
	"fname" : "M6SNC",
	"alpha" : alpha_SNC,
	"order" : 6,
	"Mfunc" : Magnus2(alpha_SNC)
}

WKB = {
	"name" : "WKB, analytic",
	"fname" : "WKB",
	"order" : 4,
	"Mfunc" : WKB_analytic
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

JMlnumSchur = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"M" : JMSchur(True, 1)
}

JMlknumSchur = {
	"name" : "JM ($\\Lambda$ and $K$)",
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
	"name" : "JM ($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : JMNumpyDiag(False, 1)
}

JMlnum2 = {
	"name" : "JM ($\\Lambda$ only)",
	"fname" : "JM1l_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus4(True, 1)
}

JMlknum2 = {
	"name" : "JM ($\\Lambda$ and $K$)",
	"fname" : "JM1lk_num",
	"order" : 4,
	"Use_numerics" : 1, 
	"Mfunc" : Jordan_Magnus4(False, 1)
}
TJMl = {
	"name" : "Taylor Jordan-Magnus (with $\\Lambda$ only)",
	"fname" : "TJMl",
	"order" : 2,
	"analytic" : True, 
	"Mfunc" : T_Jordan_Magnus(True, 0)
}

TJMlk = {
	"name" : "Taylor Jordan-Magnus (approx. P(t) to O($t^3$))",
	"fname" : "TJMlk",
	"order" : 2,
	"analytic" : True,
	"Mfunc" : T_Jordan_Magnus(False, 0)
}

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

lines = [RKF45,M6_GL,JMlknumSchur]

############### set up Numerics #################

Use_RK = False


for line in lines:
	if line["fname"] != "RKF45" :
		line["M"] = line["Mfunc"]()
	else: 
		Use_RK = True
		
'''
# correct line labels
for M in [JWKBnum, PWKBnum, JMlnum, JMlknum, MM1num, EPWKBnum]:
	M["name"] = M["name"] + " (scipy quad, maxiter=" + str(scipy_quad_maxiter) + ")"
'''
########## Integrator #################

# set error tolerance
epsilon	= 0.005
epsilon_RK = 0.005
rtol = 1		# rel. error tolerance for Magnus in units of ε
atol = 0.005	# abs. error tolerance for Magnus in units of ε
rtol_RK = 4		# rel. error tolerance for RKF4(5) in units of ε_RK
atol_RK = 2		# abs. error tolerance for RKF4(5) in units of ε_RK

def RKF45_Integrator(t_start, t_stop, h0, x0, A):
	# An integrator using a 4(5) RKF method
	T_0 = time.time()
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
	h_min = 0.01*h0 
	h_max = 10*h0
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
	return (t_, x_, T)
	
def Magnus_Integrator(t_start, t_stop, h0, x0, Method):
	# An integrator for all non-RKF4(5) methods
	T_0 = time.time()
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
	h_min = 0.01*h0
	h_max = 100*h0
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
			# compute error
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
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
	return (t_, x_, T)

def DecomposeDiagonal(t,A):
	A00_np = sym.Matrix(np.matrix(A(t)))
	P00_, J00_ = A00_np.diagonalize()
	J00 = sym.Matrix(J00_)
	P00 = sym.Matrix(P00_)
	return P00, J00
def Magnus_Integrator2(t_start, t_stop, h0, x0, Method):
	# An integrator for all non-RKF4(5) methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	"""
	A=A_num
	P,J = DecomposeDiagonal(t_start,A)
	Pinv = P.inv()
	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.01
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
			x_np1_0 = M(t, t+h,P, Pinv) @ x_n
			x_np1_l = M(t+0.5*h, t+h,P, Pinv) @ (M(t, t+0.5*h,P, Pinv) @ x_n)
			# compute error
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
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
	return (t_, x_, T)



		
def Integrator_1(t_start, t_stop,n_steps, x0, M):
	t_vec = np.linspace(t_start, t_stop, n_steps)
	t_vec0 = np.linspace(t_start, t_stop, 500)
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	Ndim = x0.size
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(Ndim)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		M_ = M(t0, t)
		x[n,:] = (M_ @ x0).reshape(Ndim)
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_vec, x, T)

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
	Ndim = x0.size
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(Ndim)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		M_ = M(t0, t)
		x[n,:] = (M_ @ x0).reshape(Ndim)
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_vec, x, T)

def FixedStepIntegrator_Spectral(t_start, t_stop,n_steps, x0):
	#A Fixed step integrator method for spectral methods
	t_vec = np.linspace(t_start, t_stop, n_steps)
	t_vec0 = np.linspace(t_start, t_stop, 500)
	T_0 = time.time()
	# We'll solve the Airy equation, where w(t) = sqrt(t) and g(t) = 0:
	w = W2_from_A(A_num)
	g = np.zeros(np.shape(A_num))
	#length = np.shape(A_num)[0]
	w = lambda t: repeat_along_diagonal(np.diag(t), 5)

	# Integration range, with a switching point at t = tswitch
	tswitch = t_start
	h=(t_stop-t_start)/n_steps
	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.
	info = solversetup(w, g, 5)
	print(info.nini, info.nmax, info.n, info.p)
	print(info.Ds)

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12
	# What's this?

	#First, and second derivative of solution initial condition
	import scipy.special as sp
	PhotonCDM["dx0"] = PhotonCDM["A_num"](PhotonCDM["t_start"]) @ PhotonCDM["x0"].reshape(5, 1)
	ui = PhotonCDM["x0"]
	dui = PhotonCDM["dx0"]
	uprev=ui
	duprev=dui
	"""
	x0 = initial conditions
	t_vec = vector of times  (N,) shape array
	"""
	Ndim = x0.size
	x = np.zeros((len(t_vec), x0.shape[0])) # set up the array of x values
	x[0, :] = x0.reshape(Ndim)
	for n in range(1,len(t_vec)):
		t0 = float(t_vec[0])
		t = float(t_vec[n])
		tprev = float(t_vec[n-1])
		x[n,:], duprev, maxerr, status, *misc = nonosc_step(info, tprev, h, uprev, duprev)
		uprev=x[n,:]
		print("\r" + "integrated {:.0%}".format(n/(len(t_vec)-1)), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_vec, x, T)

def Spectral_Integrator(t_start, t_stop, h0, x0):
	# An integrator for spectral methods
	T_0 = time.time()
	"""
	x0 = initial conditions
	t_start = start time
	t_stop = end time
	h0 = initial step size
	M = stepping function x(t) = M(t0, t) x(t0)
	
	"""
	
	epsilon	= 0.05
	rtol = 1		# rel. error tolerance for spectral in units of ε
	atol = 0.05	# abs. error tolerance for spectral in units of ε

	Ndim = x0.size
	x_ = np.zeros((1, Ndim)) # set up the array of x values
	t_ = np.zeros(1)			# set up the array of t values
	t_[0] = t_start
	x_[0,:] = x0
	h = h0
	h_min = 0.01
	h_max = 1
	n = 0
	t = t_start
	#
	S = 0.98				# safety factor

	#length = np.shape(A_num)[0]
	w = W2_from_A(A_num)
	g = np.zeros(np.shape(A_num))
	w = lambda t: repeat_along_diagonal(np.diag(t), 5)

	# Integration range, with a switching point at t = tswitch
	tswitch = t_start

	# One setup of the solver is necessary at the very start, the resulting info object stores differentiation and integration matrices, Chebyshev nodes, etc.
	info = solversetup(w, g, 5)
	print(info.nini, info.nmax, info.n, info.p)
	print(info.Ds)

	# From t = ti to t = tswitch, we pretend to have used some alternative solver.
	# It produced the solution uswitch, duswitch
	dt=1e-12
	# What's this?

	#First, and second derivative of solution initial condition
	import scipy.special as sp
	PhotonCDM["dx0"] = PhotonCDM["A_num"](PhotonCDM["t_start"]) @ PhotonCDM["x0"].reshape(5, 1)
	ui = PhotonCDM["x0"]
	dui = PhotonCDM["dx0"]
	uprev=ui
	duprev=dui

	
	#
	while t <= t_stop:
		x_n = x_[n,:].reshape(Ndim, 1)
		Err_small = False
		h_new = h
		while Err_small == False:
			# compute the predictions using one step of h & two steps of h/2
			uprev, duprev, maxerr, status, *misc = nonosc_step(info, t, h, uprev, duprev)

			unext0_5, dunext0_5, maxerr0_5, status0_5, *misc0_5 = nonosc_step(info, t, h/2, uprev, duprev)
			x_np1_l, dunext1, maxerr1, status1, *misc1 = nonosc_step(info, t+h/2, h/2, unext0_5, dunext0_5)
			order=1
			x_np1_0=uprev
			x_np1_l=duprev
			# compute error
			Err =  ferr(x_np1_0, x_np1_l)
			Err_max = epsilon*(rtol*np.abs(x_np1_l) + atol) #h*(A(t) @ x_n)[0,0]) # maximum error allowed
			Err_ratio = np.abs(np.std(Err / Err_max))
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
			print("\r" + "spectral" + "\t" + "integrated {:.1%}".format(float((t-t_start)/(t_stop-t_start))), end='')
	T = time.time() - T_0
	print(" done in {:.5g}s".format(T))
	return (t_, x_, T)


###### plot graph #####################

###### plot graph #####################

def plot_graph():
	# function for plotting a graph of the results.
	h0 = 0.025
	
	MKR_size = 2	# marker size
	log_h = True
	log_t = False
		
	######## Integration ##################
	
	t_start = Eq["t_start"]
	t_stop = Eq["t_stop"]
	
	for M in lines:
		if M["fname"] == "RKF45":
			M["data"] = RKF45_Integrator(t_start, t_stop, h0, Eq["x0"], Eq["A_num"])
		#if M["fname"] == "SpectralF":
		#	M["data"] = FixedStepIntegrator_Spectral(t_start, t_stop,500, Eq["x0"])
		#if M["fname"] == "SpectralA":
		#	M["data"] = Spectral_Integrator(t_start, t_stop, h0, Eq["x0"])
		else:
			M["data"] = Magnus_Integrator(t_start, t_stop, h0, Eq["x0"],  M)
			#M["data"] = FixedStepIntegrator_Magnus(t_start, t_stop,5000, Eq["x0"], M["M"])
	
	######################################
	
	T_num = time.time()
	print("Done numerical stepping, time taken = {:.5f}".format(T_num - T_start))
	
	linenames = ""
	for data in lines:
		linenames = linenames + data["fname"] + "_"
	filename = Eq["name"] + "_" + linenames  
	
	"""
	filename = filename + "rtol=" + str(rtol) + "_atol=" + str(atol) + "_epsil=" + str(epsilon) 
	if Use_RK:
		filename = filename + "_rtolRK=" + str(rtol_RK) + "_atolRK=" + str(atol_RK) + "_epsilRK=" + str(epsilon_RK)
	filename = filename + "scipy_quad_MI=" + str(scipy_quad_maxiter)
	"""
	
	colours = ['c', 'b', 'g', 'y', 'g']
	m_facecolours = ['c', '1', 'g', 'y', 'g']
	markertypes = ['.', 'o', 'x', '^', 'x']
	if variable == "Theta0":
		marker_size = [MKR_size, MKR_size, MKR_size+2, MKR_size+1,MKR_size]
	elif variable == "Phi":
		marker_size = [MKR_size, MKR_size, MKR_size, MKR_size,MKR_size]
	# set height ratios for sublots
	gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
	
	################ Primary plot & error plot
	
	### fit analytic solutions for Phi to initial conditions 
	
	# fit solution curve to the data from the first line
	t_vec0_start = 0.015
	line = lines[0]
	t_fit = line["data"][0]
	x_fit = np.real((line["data"][1][:,0]))
	x_fit = np.array(x_fit, dtype=np.float64).reshape(t_fit.size)
	popt, pcov = optimize.curve_fit(PhotonCDM_Theta0_sol, t_fit, x_fit, p0 = [1, 0])
	Eq["Theta0_sol_coeff"] = popt
	# fit to Phi line
	start_index = 0
	for i in range(0, t_fit.size):
		if t_fit[i] > t_vec0_start:
			start_index = i
			break
	t_fit = t_fit[start_index:]
	x_fit = np.real((line["data"][1][start_index:,4]))
	x_fit = np.array(x_fit, dtype=np.float64).reshape(t_fit.size)
	popt, pcov = optimize.curve_fit(PhotonCDM_Phi_sol, t_fit, x_fit, p0 = [1, 0])
	Eq["Phi_sol_coeff"] = popt
	print("Eq[Phi_sol_coeff] = ",Eq["Phi_sol_coeff"])
	
	# fit to initial conditions
	t0 = PhotonCDM["t_start"]
	
	PhotonCDM["dx0"] = PhotonCDM["A_num"](PhotonCDM["t_start"]) @ PhotonCDM["x0"].reshape(5, 1)
	
	q0 = k*t0/np.sqrt(3)
	J0 = special.spherical_jn(1,q0)/(q0)
	Y0 = special.spherical_yn(1,q0)/(q0)
	dJ0 = (k/np.sqrt(3))*(special.spherical_jn(1,q0,derivative=True)/(q0) - special.spherical_jn(1,q0)/(q0**2))
	dY0 = (k/np.sqrt(3))*(special.spherical_yn(1,q0,derivative=True)/(q0) - special.spherical_yn(1,q0)/(q0**2))	
	M0 = np.matrix([[1, 1], [dJ0/J0, dY0/Y0]])
	print(np.shape(PhotonCDM["x0"][4]))
	print(np.shape(PhotonCDM["dx0"][4][0,0]))
	x0_vec = np.array([PhotonCDM["x0"][4], PhotonCDM["dx0"][4][0,0]])



	AB = np.linalg.inv(M0) @ x0_vec.reshape(2, 1)
	print("AB = ", AB)
	
	#
	t_vec0 = np.linspace(t_start, t_stop, 500)
	
	if variable == "Phi":
		x_neumann = PhotonCDM_Phi_sol(t_vec0, AB[0,0], AB[1,0])
		x_bessel = PhotonCDM_Phi_sol(t_vec0, Eq["Phi_sol_coeff"][0], Eq["Phi_sol_coeff"][1])
		def x_true_func(t):
			start_index = 0
			for i in range(0, t.size):
				if t[i] > t_vec0_start:
					start_index = i
					break
			x_init = PhotonCDM_Phi_sol(t, AB[0,0], AB[1,0])
			x_late = PhotonCDM_Phi_sol(t, Eq["Phi_sol_coeff"][0], Eq["Phi_sol_coeff"][1])
			x_true_ = np.append(x_init[:start_index], x_late[start_index:])
			return x_true_
	elif variable == "Theta0":
		def x_true_func(t):		
			x_true_ = PhotonCDM_Theta0_sol(t, Eq["Theta0_sol_coeff"][0], Eq["Theta0_sol_coeff"][1])
			return x_true_
	
	### create first subplot
	ax0 = plt.subplot(gs[0])
	fig = plt.gcf()
	fig.set_size_inches(21,12)
	font_size = 8
	title_font_size = 10
	label_size = 10
	legend_font_size = 8
	rc('xtick',labelsize=font_size)
	rc('ytick',labelsize=font_size)

	# plot theoretical solutions
	if variable == "Phi":
		ax0.plot(t_vec0, x_bessel, color="k", linewidth=1, linestyle="--", label="adiabatic soln. fitted to data for $\\eta>"+str(t_vec0_start)+"$")
		#ax0.plot(t_vec0, x_neumann, color="k", linewidth=1, linestyle=":", label="adiabatic soln. fitted to init. cond.")
		ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	elif variable == "Theta0":
		x_true0 = x_true_func(t_vec0)
		ax0.plot(t_vec0, x_true0, color="k", linewidth=1, linestyle="--", label="sinusoid soln. fitted to data")
		ax0.set_ylim(Eq["ylim"][0], Eq["ylim"][1])
	#
	#ax2 = plt.subplot(gs[2], sharex = ax0)
	#ax2.plot(np.linspace(Eq["t_start"], t_stop, 20), np.log10(epsilon*np.ones(20)), color="k", linewidth=1, linestyle=":", label="$\epsilon$")
	#ax2.annotate("$\epsilon$", xy=(1.05*Eq["t_stop"], epsilon))
	#
	for i in range(0, len(lines)):
		line = lines[i]
		t = line["data"][0]
		x = (line["data"][1][:,Index]).reshape(t.size, 1)
		#x_true = x_true_func(t).reshape(t.size, 1)
		#x_true = (Eq["true_sol"](t)[:,Index]).reshape(t.size, 1)
		#error = np.log10(np.abs((x - x_true)/x_true))
		T = line["data"][2]
		ax0.plot(t, x, markertypes[i], markerfacecolor=m_facecolours[i], markeredgecolor=colours[i], markersize=marker_size[i], linewidth=1, label="{:s}, T={:.4g}s".format(line["name"], T))
		#ax2.plot(t, error, colours[i] + '--', linewidth=1, alpha=1)
	ax0.set_ylabel("$x$", fontsize=label_size)
	ax0.set_xlim(Eq["t_start"], t_stop)
	'''ax2.set_xlim(Eq["t_start"], t_stop)
	ax2.set_ylabel("log$_{10}$(rel. error)", fontsize=label_size)
	ax2.legend()
	ax2.set_xlabel("$\eta$", fontsize=label_size, labelpad=2)
	ax2.minorticks_on()'''
	#ymin, ymax = ax2.get_ylim()
	#if ymax>1:
	#	ax2.set_ylim(top=1)
	#ax2.set_ylim(-3, 1)
	ax0.minorticks_on()
	
	if variable == "Theta0":
		ax0.set_title(Eq["title"], y=1.08, fontsize=title_font_size)
		lgnd = ax0.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.25, 0.88, 0.50, 0.25), ncol = 2, shadow=False) # 
		for i in range(0, len(lines)+1):
			lgnd.legendHandles[i]._legmarker.set_markersize(5)
	elif variable == "Phi":
		ax0.set_title(Eq["title"], y=1.00, fontsize=title_font_size)
		lgnd = ax0.legend(fontsize=8, loc='upper center', ncol = 2, shadow=False) # 
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
			ax1.plot(t_av, np.log10(h), colours[i] + '-', linewidth=1, label="{:s}".format(line["name"]))
		elif log_h == False:
			ax1.plot(t_av, h, colours[i] + '-', linewidth=1, label="{:s}".format(line["name"]))
	if log_h:
		ax1.set_ylabel("$\\log_{10}$($h$)", fontsize=label_size)
		#savename = "Plots/" + filename + "_log_h.pdf"
	elif log_h == False:
		ax1.set_ylabel("h", fontsize=font_size)
		#savename = "Plots/" + filename + ".pdf"
	
	ax1.set_xlim(Eq["t_start"], t_stop)
	
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
	#plt.setp(ax0.get_yticklabels()[0], visible=False) 
	
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)
	#### set figure size and font size
	#
	ax0.tick_params(axis='y', labelsize=font_size)
	ax1.tick_params(axis='y', labelsize=font_size)
	#ax2.tick_params(axis='y', labelsize=font_size)
	#ax2.tick_params(axis='x', labelsize=font_size)
	plt.subplots_adjust(hspace=.0)
	#plt.savefig(savename, transparent=True)
	#plt.clf()
	print("made plot")
	print("saved as " + savename)
	


#################################################################
plot_graph()

#get_times()