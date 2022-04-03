import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	m,xa,ya,xb,yb,k1,k2,r1eq,r2eq = p
	r,v,theta,omega = ic

	sub = {M:m, Xa:xa, Ya:ya, Xb:xb, Yb:yb, K1:k1, K2:k2, R1eq:r1eq, R2eq:r2eq, R:r, Rdot:v, THETA:theta, THETAdot:omega}

	print(ti)

	return [v,A.subs(sub),omega,ALPHA.subs(sub)]


M, K1, K2, R1eq, R2eq, Xa, Ya, Xb, Yb, t = sp.symbols('M K1 K2 R1eq R2eq Xa Ya Xb Yb t')
R, THETA = dynamicsymbols('R THETA')

X = Xa + R * sp.cos(THETA)
Y = Ya + R * sp.sin(THETA)

#dR2s = (X - Xb + R2eq * sp.cos(sp.pi - THETA))**2 + (Y - Yb + R2eq * sp.sin(sp.pi - THETA))**2
dR2 = sp.simplify(sp.sqrt((X - Xb)**2 + (Y - Yb)**2) - R2eq)

Xdot = X.diff(t,1)
Ydot = Y.diff(t,1)
Rdot = R.diff(t,1)
THETAdot = THETA.diff(t,1)

T = sp.Rational(1,2) * M * (Xdot**2 + Ydot**2)

V = sp.Rational(1,2) * (K1 * (R - R1eq)**2 + K2 * dR2**2)

L = T - sp.simplify(V)

dLdR = L.diff(R,1)
dLdRdot = L.diff(Rdot,1)
ddtdLdRdot = dLdRdot.diff(t,1)

dLdTHETA = L.diff(THETA,1)
dLdTHETAdot = L.diff(THETAdot,1)
ddtdLdTHETAdot = dLdTHETAdot.diff(t,1)

dLR = sp.simplify(ddtdLdRdot - dLdR)
dLTHETA = sp.simplify(ddtdLdTHETAdot - dLdTHETA)

Rddot = R.diff(t,2)
THETAddot = THETA.diff(t,2)

sol = sp.solve([dLR,dLTHETA],(Rddot,THETAddot))

A = sp.simplify(sol[Rddot])
ALPHA = sol[THETAddot]

#--------------------------------------------------

m = 1
xa, ya = [0, 0]
xb, yb = [5, 5]
k1, k2 = [25, 25]
r1eq, r2eq = [5*np.sqrt(2)/2, 5*np.sqrt(2)/2]
ro = 15 
vo = 0
thetao = 175 
omegao = 0
cnvrt = np.pi/180
thetao *= cnvrt
omegao *= cnvrt
mr = 0.5
tf = 60 

p = m,xa,ya,xb,yb,k1,k2,r1eq,r2eq
ic = ro,vo,thetao,omegao

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args=(p,))

x = np.asarray([X.subs({Xa:xa, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)
y = np.asarray([Y.subs({Ya:ya, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)],dtype=float)

ke = np.asarray([T.subs({M:m, Xa:xa, Ya:ya, R:rth[i,0], Rdot:rth[i,1], THETA:rth[i,2], THETAdot:rth[i,3]}) for i in range(nframes)])
pe = np.asarray([V.subs({K1:k1, K2:k2, R1eq:r1eq, R2eq:r2eq, Xa:xa, Ya:ya, Xb:xb, Yb:yb, R:rth[i,0], THETA:rth[i,2]}) for i in range(nframes)])
E = ke + pe

#-----------------------------------------------------

xp = [xa,xb]
yp = [ya,yb]

xmax = max(x) + 2 * mr if max(x) > max(xp) else max(xp) + 2 * mr
xmin = min(x) - 2 * mr if min(x) < min(xp) else min(xp) - 2 * mr
ymax = max(y) + 2 * mr if max(y) > max(yp) else max(yp) + 2 * mr
ymin = min(y) - 2 * mr if min(y) < min(yp) else min(yp) - 2 * mr

theta2 = np.arccos((yb-y)/((xb - x)**2 + (yb - y)**2)**0.5)

r1max = max(np.abs(rth[:,0]))
r2max = max(((xb - x)**2 + (yb - y)**2)**0.5)
nl1 = int(np.ceil(r1max/(2*mr)))
nl2 = int(np.ceil(r2max/(2*mr)))
l1 = (np.asarray(np.abs(rth[:,0]))-mr)/nl1
l2 = (np.sqrt((xb - x)**2 + (yb - y)**2)-mr)/nl2
h1 = np.sqrt(mr**2 - (0.5*l1)**2)
h2 = np.sqrt(mr**2 - (0.5*l2)**2)
xl1o = x - mr*np.sin(np.pi/2 - np.asarray(rth[:,2]))
yl1o = y - mr*np.cos(np.pi/2 - np.asarray(rth[:,2]))
flipa = np.asarray([-1 if x[i]>xb and y[i]<yb else 1 for i in range(nframes)])
flipb = np.asarray([-1 if x[i]<xb and y[i]>yb else 1 for i in range(nframes)])
flipc = np.asarray([-1 if x[i]<xb else 1 for i in range(nframes)])
xl2o = x + np.sign((yb-y)*flipa*flipb) * mr*np.sin(theta2)
yl2o = y + mr*np.cos(theta2)
xl1 = np.zeros((nl1,nframes))
yl1 = np.zeros((nl1,nframes))
xl2 = np.zeros((nl2,nframes))
yl2 = np.zeros((nl2,nframes))
for i in range(nframes):
	flip1 = -1 if x[i]>xb and y[i]<yb else 1
	flip2 = -1 if x[i]<xb and y[i]>yb else 1
	flip3 = -1 if x[i]<xb else 1
	xl1[0][i] = xl1o[i] - 0.5 * l1[i] * np.sin(np.pi/2 - rth[i,2]) - h1[i] * np.sin(rth[i,2])
	yl1[0][i] = yl1o[i] - 0.5 * l1[i] * np.cos(np.pi/2 - rth[i,2]) + h1[i] * np.cos(rth[i,2])
	xl2[0][i] = xl2o[i] + np.sign((yb-y[i])*flip1*flip2) * 0.5 * l2[i] * np.sin(theta2[i]) - np.sign((yb-y[i])*flip1*flip2) * flip3 * h2[i] * np.sin(np.pi/2 - theta2[i])
	yl2[0][i] = yl2o[i] + 0.5 * l2[i] * np.cos(theta2[i]) + flip3 * h2[i] * np.cos(np.pi/2 - theta2[i])
for j in range(nframes):
	for i in range(1,nl1):
		xl1[i][j] = xl1o[j] - (0.5 + i) * l1[j] * np.sin(np.pi/2 - rth[j,2]) - (-1)**i * h1[j] * np.sin(rth[j,2])
		yl1[i][j] = yl1o[j] - (0.5 + i) * l1[j] * np.cos(np.pi/2 - rth[j,2]) + (-1)**i * h1[j] * np.cos(rth[j,2])
	for i in range(1,nl2):
		flip1 = -1 if x[j]>xb and y[j]<yb else 1
		flip2 = -1 if x[j]<xb and y[j]>yb else 1
		flip3 = -1 if x[j]<xb else 1
		xl2[i][j] = xl2o[j] + np.sign((yb-y[j])*flip1*flip2) * (0.5 + i) * l2[j] * np.sin(theta2[j]) - np.sign((yb-y[j])*flip1*flip2) * flip3 * (-1)**i * h2[j] * np.sin(np.pi/2 - theta2[j])
		yl2[i][j] = yl2o[j] + (0.5 + i) * l2[j] * np.cos(theta2[j]) + flip3 * (-1)**i * h2[j] * np.cos(np.pi/2 - theta2[j])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x[frame],y[frame]),radius=mr,fc='xkcd:red')
	plt.gca().add_patch(circle)
	circle=plt.Circle((xa,ya),radius=0.5*mr,fc='xkcd:cerulean')
	plt.gca().add_patch(circle)
	circle=plt.Circle((xb,yb),radius=0.5*mr,fc='xkcd:cerulean')
	plt.gca().add_patch(circle)
	plt.plot([xl1o[frame],xl1[0][frame]],[yl1o[frame],yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[nl1-1][frame],xa],[yl1[nl1-1][frame],ya],'xkcd:cerulean')
	plt.plot([xl2o[frame],xl2[0][frame]],[yl2o[frame],yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[nl2-1][frame],xb],[yl2[nl2-1][frame],yb],'xkcd:cerulean')
	for i in range(nl1-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	for i in range(nl2-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.title("A Springy Thingy")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('springy_thingy.mp4', writer=writervideo)
plt.show()





