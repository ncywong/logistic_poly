from equadratures import *
import numpy as np

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
from scipy.optimize import minimize

class logistic_poly():
    def __init__(self, X_train, f_train, n=2, dummy_poly=None, M_init=None, c_init=None, tol=1e-7, cauchy_tol=1e-5,
                  cauchy_length = 3, verbosity=2, order=2, C=1.0, max_M_iters=10, restarts = 5):
        self.X = X_train
        self.f = f_train
        self.N_train, self.d = X_train.shape
        self.n = n
        self.tol = tol
        self.cauchy_tol = cauchy_tol
        self.verbosity = verbosity
        self.cauchy_length = cauchy_length
        self.C = C
        self.max_M_iters = max_M_iters
        self.restarts = restarts
        self.order = order

        if M_init is None:
            self.M = np.linalg.qr(np.random.randn(self.d, self.n))[0]
        else:
            self.M = M_init

        if dummy_poly is None:
            my_params = [Parameter(order=order, distribution='uniform', lower=-np.sqrt(self.d), upper=np.sqrt(self.d)) for _ in range(n)]
            my_basis = Basis('total-order')
            self.dummy_poly = Poly(parameters=my_params, basis=my_basis, method='least-squares',
                                               sampling_args={'mesh':'user-defined',
                                                               'sample-points': self.X @ self.M,
                                                               'sample-outputs': self.f})
        else:
            self.dummy_poly = dummy_poly

        if c_init is None:
            my_poly_init = Poly(parameters=my_params, basis=my_basis, method='least-squares',
                                               sampling_args={'mesh':'user-defined',
                                                               'sample-points': self.X @ self.M,
                                                               'sample-outputs': self.f})
            my_poly_init.set_model()
            self.c = my_poly_init.coefficients
        else:
            self.c = c_init

    @staticmethod
    def sigmoid(U):
        return 1.0 / (1.0 + np.exp(-U))

    def p(self, X, M, c):
        self.dummy_poly.coefficients = c
        return self.dummy_poly.get_polyfit(X @ M).reshape(-1)

    def phi(self, X, M, c):
        pW = self.p(X, M, c)
        return self.sigmoid(pW)

    def cost(self, f, X, M, c):
        this_phi = self.phi(X, M, c)
        return -np.sum(f * np.log(this_phi + 1e-15) + (1.0 - f) * np.log(1 - this_phi + 1e-15)) \
               + 0.5 * self.C * np.linalg.norm(c)**2

    def dcostdc(self, f, X, M, c):
        W = X @ M
        self.dummy_poly.coefficients = c

        V = self.dummy_poly.get_poly(W)
        U = self.dummy_poly.get_polyfit(W).reshape(-1)
        diff = f - self.phi(X, M, c)

        return -np.dot(V, diff) + self.C * c

    def dcostdM(self, f, X, M, c):
        self.dummy_poly.coefficients = c

        W = X @ M
        U = self.dummy_poly.get_polyfit(W).reshape(-1)
        J = np.array(self.dummy_poly.get_poly_grad(W))
        if len(J.shape) == 2:
            J = J[np.newaxis,:,:]

        diff = f - self.phi(X, M, c)

        Jt = J.transpose((2,0,1))
        XJ = X[:, :, np.newaxis] * np.dot(Jt[:, np.newaxis, :, :], c)

        result = -np.dot(XJ.transpose((1,2,0)), diff)
        return result

    def fit(self):
        f = self.f
        X = self.X
        tol = self.tol
        d = self.d
        n = self.n

        current_best_residual = np.inf
        for r in range(self.restarts):
            print('restart %d' % r)
            M0 = np.linalg.qr(np.random.randn(self.d, self.n))[0]
            my_params = [Parameter(order=self.order, distribution='uniform', lower=-5, upper=5) for _ in range(n)]
            my_basis = Basis('total-order')
            my_poly_init = Poly(parameters=my_params, basis=my_basis, method='least-squares',
                                sampling_args={'mesh': 'user-defined',
                                               'sample-points': X @ M0,
                                               'sample-outputs': f})
            my_poly_init.set_model()
            c0 = my_poly_init.coefficients.copy()

            residual = self.cost(f, X, M0, c0)

            cauchy_length = self.cauchy_length
            residual_history = []
            iter_ind = 0
            M = M0.copy()
            c = c0.copy()
            while residual > tol:
                if self.verbosity == 2:
                    print(residual)
                residual_history.append(residual)
                # Minimize over M
                func_M = lambda M_var: self.cost(f, X, M_var, c)
                grad_M = lambda M_var: self.dcostdM(f, X, M_var, c)

                manifold = Stiefel(d, n)
                solver = ConjugateGradient(maxiter=self.max_M_iters)

                problem = Problem(manifold=manifold, cost=func_M, egrad=grad_M, verbosity=0)

                M = solver.solve(problem, x=M)

                # Minimize over c
                func_c = lambda c_var: self.cost(f, X, M, c_var)
                grad_c = lambda c_var: self.dcostdc(f, X, M, c_var)

                res = minimize(func_c, x0=c, method='CG', jac=grad_c)
                c = res.x
                residual = self.cost(f, X, M, c)
                if iter_ind < cauchy_length:
                    iter_ind += 1
                elif np.abs(np.mean(residual_history[-cauchy_length:]) - residual)/residual < self.cauchy_tol:
                    break

            if self.verbosity > 0:
                print('final residual on training data: %f' % self.cost(f, X, M, c))
            if residual < current_best_residual:
                self.M = M
                self.c = c
                current_best_residual = residual




    def predict(self, X):
        return self.phi(X, self.M, self.c)


