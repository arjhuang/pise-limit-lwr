"""
@author: Archie Huang
Built upon Dr. Maziar Raissi's PINNs - https://github.com/maziarraissi/PINNs

Use Tensorflow 1.x

code ref-
1. boundary01_pidl_fixed_input.py
2. boundary02_pidl_var_bud.py for selecting boundary data
3. ojits01_synthetic.py for processing synthetic dataset
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from plotting import newfig
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N_u = 216
N_f = 10000

se = 1234
np.random.seed(se)
tf.set_random_seed(se)

# PINN Class
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        self.u = u

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        ##################### Addition ##############################
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, seed=se), dtype=tf.float32)
        #############################################################

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        # u_xx = tf.gradients(u_x, x)[0] ## Shi, physics-cost with diffusion term
        # f = u_t + u_x - 2 * u * u_x - 0.005 * u_xx ## Shi, physics-cost with diffusion term
        f = u_t + u_x - 2 * u * u_x ## Animesh, physics-cost without diffusion term
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
        return u_star, f_star



layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# data = scipy.io.loadmat('../data/rho_bellshape_10grid_DS10_gn_eps005_solver2_ring.mat')
data = scipy.io.loadmat('../../data/ring_nodiff.mat')
uxn = 241
xlo = 0.
xhi = 1.0042

utn = 960
tlo = 0.
thi = 3.
x = np.linspace(xlo, xhi, uxn)
t = np.linspace(tlo, thi, utn)
# Exact = np.real(data['rho']).T # Shi, with diffusion
Exact = np.real(data['compressed_u']).T # Animesh, without diffusion

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

######################## Eulerian Training Data #################################

xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

X_u_train1 = np.vstack([xx1, xx2, xx3])
u_train1 = np.vstack([uu1, uu2, uu3])
idx1 = np.random.choice(X_u_train1.shape[0], N_u, replace=False)
X_u_train = X_u_train1[idx1, :]
u_train = u_train1[idx1, :]
#
# ######################### Collocation Points #################################

lb = X_star.min(0)
ub = X_star.max(0)
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))

############################### Model Training ###################################

# PINN Model
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed)
u_pred, f_pred = model.predict(X_star)
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Error u: %e' % error_u)
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)


fig, ax = newfig(1.4, 1.8)
ax.axis('off')

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=0.9, bottom=0.7, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

ax.tick_params(axis='both', which='major', labelsize=10)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='RdBu',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.tick_params(labelsize=10)
fig.colorbar(h, cax=cax, ticks=[0., .25, .5, .75, 1.])

ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', markersize=4, clip_on=False)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[0] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[160] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[320] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[480] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[640] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[800] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('Time $t$', fontsize=10)
ax.set_ylabel('Location $x$', fontsize=10)
ax.legend(frameon=False, loc='best', fontsize=10)
ax.set_title('Density Reconstruction $\\rho (x,t)$, [L-BFGS-B, Hyperbolic]', fontsize=10)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=0.6, bottom=0.4, left=0.12, right=0.92, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.set_title('$t = 0$', fontsize=10)
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.yaxis.set_major_locator(MultipleLocator(0.25))

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[160, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[160, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.set_title('$t = 0.5$', fontsize=10)
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.yaxis.set_major_locator(MultipleLocator(0.25))

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[320, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[320, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = 1.0$', fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.25))


####### Row 2: u(t,x) slices ##################
gs2 = gridspec.GridSpec(1, 3)
gs2.update(top=0.3, bottom=0.1, left=0.12, right=0.92, wspace=0.5)

ax = plt.subplot(gs2[0, 0])
ax.plot(x, Exact[480, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[480, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = 1.5$', fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.25))

ax = plt.subplot(gs2[0, 1])
ax.plot(x, Exact[640, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[640, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = 2.0$', fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.25))

ax = plt.subplot(gs2[0, 2])
ax.plot(x, Exact[800, :], 'b-', linewidth=2, label='Ground Truth')
ax.plot(x, U_pred[800, :], 'r--', linewidth=2, label='Reconstruction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (x,t)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = 2.5$', fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.legend(loc='lower center', bbox_to_anchor=(-1.0, -0.5), ncol=5, frameon=False)

plt.savefig('../res_bund/ring_bfgs_bund_hype_{}_{:.4f}.pdf'.format(N_u, error_u))
plt.savefig('../res_bund/ring_bfgs_bund_hype_{}_{:.4f}.eps'.format(N_u, error_u))
plt.show()