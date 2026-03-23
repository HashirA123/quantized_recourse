from sklearn.linear_model import LogisticRegression
import lime
import lime.lime_tabular
import torch
import numpy as np
from torch import optim
from torch.autograd import grad
from torch.autograd import Variable
from scipy.optimize import linprog

# code adapted from https://github.com/AI4LIFE-GROUP/ROAR

def lime_explanation(model_pred_proba, X_train, x):
	explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train, 
		discretize_continuous=False,
		feature_selection='none')
	exp = explainer.explain_instance(x,
						model_pred_proba,
						num_features=X_train.shape[1],
						model_regressor=LogisticRegression()
	)
	coefficients = exp.local_exp[1][0][1]
	# coefficients = np.zeros(X_train.shape[1])

	# for tpl in exp.local_exp[1]:
	# 	coefficients[tpl[0]] = tpl[1]
	intercept = exp.intercept[1]
	return coefficients, np.array(intercept)

class ROAR():
    def __init__(self, W=None, W0=None, y_target=1,
                 delta_max=0.1, feature_costs=None,
                 pW=None, pW0=None):
        self.set_W(W)
        self.set_W0(W0)

        self.set_pW(pW)
        self.set_pW0(pW0)

        self.y_target = torch.tensor(y_target).float()
        self.delta_max = delta_max
        self.feature_costs = feature_costs
        if self.feature_costs is not None:
            self.feature_costs = torch.from_numpy(feature_costs).float()

    def set_W(self, W):
        self.W = W
        if W is not None:
            self.W = torch.from_numpy(W).float()

    def set_W0(self, W0):
        self.W0 = W0
        if W0 is not None:
            self.W0 = torch.from_numpy(W0).float()

    def set_pW(self, pW):
        self.pW = pW
        if pW is not None:
            self.pW = torch.from_numpy(pW).float()

    def set_pW0(self, pW0):
        self.pW0 = pW0
        if pW0 is not None:
            self.pW0 = torch.from_numpy(pW0).float()

    def l1_cost(self, x_new, x):
        cost = torch.dist(x_new, x, 1)
        return cost

    def pfc_cost(self, x_new, x):
        cost = torch.norm(self.feature_costs * (x_new - x), 1)
        return cost

    def calc_delta_opt(self, recourse):
        """
		calculate the optimal delta using linear program
		:returns: torch tensor with optimal delta value
		"""
        W = torch.cat((self.W, self.W0), 0)  # Add intercept to weights
        recourse = torch.cat((recourse, torch.ones(1)), 0)  # Add 1 to the feature vector for intercept

        loss_fn = torch.nn.BCELoss()

        A_eq = np.empty((0, len(W)), float)

        b_eq = np.array([])

        W.requires_grad = True
        f_x_new = torch.nn.Sigmoid()(torch.matmul(W, recourse))
        w_loss = loss_fn(f_x_new, self.y_target)
        gradient_w_loss = grad(w_loss, W)[0]

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.delta_max, self.delta_max)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='highs')
        delta_opt = res.x  # the delta value that maximizes the function
        delta_W, delta_W0 = np.array(delta_opt[:-1]), np.array([delta_opt[-1]])
        return delta_W, delta_W0

    def get_recourse(self, x, lamb=0.1) -> np.ndarray:
        # torch.manual_seed(0)

        # returns x'
        x = torch.from_numpy(x).float()
        # print(f"this is x: {x}")
        lamb = torch.tensor(lamb).float()

        x_new = Variable(x.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new])

        loss_fn = torch.nn.BCELoss()

        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1
        f_x_new = 0

        while loss_diff > 1e-4:
            loss_prev = loss.clone().detach()

            delta_W, delta_W0 = self.calc_delta_opt(x_new)
            delta_W, delta_W0 = torch.from_numpy(delta_W).float(), torch.from_numpy(delta_W0).float()

            optimizer.zero_grad()
            if self.pW is not None:
                dec_fn = torch.matmul(self.W + delta_W, x_new) + self.W0
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.pW, dec_fn.unsqueeze(0)) + self.pW0)[0]
            else:
                f_x_new = torch.nn.Sigmoid()(torch.matmul(self.W + delta_W, x_new) + self.W0 + delta_W0)[0]

            if self.feature_costs is not None:
                cost = self.pfc_cost(x_new, x)
            else:
                cost = self.l1_cost(x_new, x)

            loss = loss_fn(f_x_new, self.y_target) + lamb * cost
            loss.backward()
            optimizer.step()

            loss_diff = torch.dist(loss_prev, loss, 2)
        # print(f"this is x_new: {x_new}")
        return x_new.detach().numpy()
