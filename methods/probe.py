import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.distributions.normal as normal_distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
from torch.autograd import Variable
import logging

# code adapted from https://github.com/MartinPawelczyk/ProbabilisticallyRobustRecourse

DECISION_THRESHOLD = 0.5

# Mean and variance for rectified normal distribution:
# see in here : http://journal-sfds.fr/article/view/669

class PROBE():
    def __init__(self, 
            cat_feature_indices: List[int],
            binary_cat_features: bool = True,
            feature_costs: Optional[List[float]] = None,
            lr: float = 0.001,
            lambda_param: float = 0.01,
            y_target: List[int] = [0.45, 0.55],
            n_iter: int = 100,
            t_max_min: float = 0.5,
            norm: int = 1,
            clamp: bool = True,
            loss_type: str = "BCE",
            invalidation_target: float = 0.35,
            inval_target_eps: float = 0.001,
            noise_variance: float = 0.01
        ):
        self.cat_feature_indices = cat_feature_indices
        self.binary_cat_features = binary_cat_features
        self.feature_costs = feature_costs
        self.lr = lr
        self.lambda_param = lambda_param
        self.y_target = y_target
        self.n_iter = n_iter
        self.t_max_min = t_max_min
        self.norm = norm
        self.clamp = clamp
        self.loss_type = loss_type
        self.invalidation_target = invalidation_target
        self.inval_target_eps = inval_target_eps
        self.noise_variance = noise_variance

    def compute_jacobian(self, inputs, output, num_classes=1):
        """
        :param inputs: Batch X Size (e.g. Depth X Width X Height)
        :param output: Batch X Classes
        :return: jacobian: Batch X Classes X Size
        """
        assert inputs.requires_grad
        grad = self.gradient(output, inputs)
        return grad


    def gradient(self, y, x, grad_outputs=None):
        """Compute dy/dx @ grad_outputs"""
        if grad_outputs is None:
            grad_outputs = torch.tensor(1)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad


    def compute_invalidation_rate_closed(self, torch_model, x, sigma2):
        # Compute input into CDF
        prob = torch_model(x)
        logit_x = torch.log(prob[0][1] / prob[0][0])
        Sigma2 = sigma2 * torch.eye(x.shape[0])
        jacobian_x = self.compute_jacobian(x, logit_x, num_classes=1).reshape(-1)
        denom = torch.sqrt(sigma2) * torch.norm(jacobian_x, 2)
        arg = logit_x / denom
        
        # Evaluate Gaussian cdf
        normal = normal_distribution.Normal(loc=0.0, scale=1.0)
        normal_cdf = normal.cdf(arg)
        
        # Get invalidation rate
        ir = 1 - normal_cdf
        
        return ir


    def perturb_sample(self, x, n_samples, sigma2):
        # stack copies of this sample, i.e. n rows of x.
        X = x.repeat(n_samples, 1)
        # sample normal distributed values
        Sigma = torch.eye(x.shape[1]) * sigma2
        eps = MultivariateNormal(
            loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
        ).sample((n_samples,))
        
        return X + eps

    def reparametrization_trick(self, mu, sigma2, n_samples):
        
        #var = torch.eye(mu.shape[1]) * sigma2
        std = torch.sqrt(sigma2).to(mu.device)
        epsilon = MultivariateNormal(loc=torch.zeros(mu.shape[1]), covariance_matrix=torch.eye(mu.shape[1]))
        epsilon = epsilon.sample((n_samples,)).to(mu.device)  # standard Gaussian random noise
        ones = torch.ones_like(epsilon).to(mu.device)
        random_samples = mu.reshape(-1) * ones + std * epsilon
        
        return random_samples


    def compute_invalidation_rate(self, torch_model, random_samples):
        yhat = torch_model(random_samples)[:, 1]
        hat = (yhat > 0.5).float()
        ir = 1 - torch.mean(hat, 0)
        return ir


    def get_recourse(
        self,
        torch_model,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Generates counterfactual example according to Wachter et.al for input instance x

        Parameters
        ----------
        torch_model: black-box-model to discover
        x: factual to explain

        Returns
        -------
        Counterfactual example as np.ndarray
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # returns counterfactual instance
        # torch.manual_seed(0)
        noise_variance = torch.tensor(noise_variance)

        if feature_costs is not None:
            feature_costs = torch.from_numpy(feature_costs).float().to(device)

        x = torch.from_numpy(x).float().to(device)
        y_target = torch.tensor(y_target).float().to(device)
        lamb = torch.tensor(self.lambda_param).float().to(device)
        # x_new is used for gradient search in optimizing process
        x_new = Variable(x.clone(), requires_grad=True)

        optimizer = optim.Adam([x_new], self.lr, amsgrad=True)
        softmax = nn.Softmax()

        if self.loss_type == "MSE":
            loss_fn = torch.nn.MSELoss()
            f_x_new = softmax(torch_model(x_new))[1]
        else:
            loss_fn = torch.nn.BCELoss()
            f_x_new = torch_model(x_new)[:, 1]

        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=self.t_max_min)

        costs = []
        ces = []

        random_samples = self.reparametrization_trick(x_new, noise_variance, n_samples=1000)
        invalidation_rate = self.compute_invalidation_rate(torch_model, random_samples)
        
        while (f_x_new <= DECISION_THRESHOLD) or (invalidation_rate > self.invalidation_target + self.inval_target_eps):

            for it in range(self.n_iter):                
                optimizer.zero_grad()

                f_x_new_binary = torch_model(x_new).squeeze(axis=0)

                cost = (
                    torch.dist(x_new, x, self.norm)
                    if feature_costs is None
                    else torch.norm(feature_costs * (x_new - x), self.norm)
                )

                # invalidation_rate = compute_invalidation_rate(torch_model, random_samples)
                invalidation_rate_c = self.compute_invalidation_rate_closed(torch_model, x_new, noise_variance)
                
                # Compute & update losses
                loss_invalidation = invalidation_rate_c - self.invalidation_target

                # Hinge loss
                loss_invalidation[loss_invalidation < 0] = 0

                loss = 3 * loss_invalidation + loss_fn(f_x_new_binary, y_target) + lamb * cost
                loss.backward()
                optimizer.step()

                random_samples = self.reparametrization_trick(x_new, noise_variance, n_samples=10000)
                invalidation_rate = self.compute_invalidation_rate(torch_model, random_samples)
                
                # clamp potential CF
                if self.clamp:
                    x_new.clone().clamp_(0, 1)

                f_x_new = torch_model(x_new)[:, 1]

            if (f_x_new > self.DECISION_THRESHOLD) and (invalidation_rate < self.invalidation_target + self.inval_target_eps):
                    costs.append(cost)
                    ces.append(x_new)
                    
                    break
                    
            lamb -= 0.10
            
            if datetime.datetime.now() - t0 > t_max:
                logging.info("Timeout")
                break

        if not ces:
            logging.info("No Counterfactual Explanation Found at that Target Rate - Try Different Target")
        else:
            logging.info("Counterfactual Explanation Found")
            costs = torch.tensor(costs)
            min_idx = int(torch.argmin(costs).numpy())
            x_new_enc = ces[min_idx]
                
        return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
