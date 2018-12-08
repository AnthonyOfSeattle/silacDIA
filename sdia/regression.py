import numpy as np
import scipy as sp

class GradGL:
    def __init__(self, l2_penalty, step_size=5e-10, max_iter=100):

        self.l2_penalty = l2_penalty
        self.step_size = step_size
        self.max_iter = max_iter
        self.X_ = None
        self.y_ = None
        self.group_map_ = None
        self.coef_ = None
        self.loss_ = None
        
    def preprocess_groups(self, groups):
        idx = np.arange(len(groups))
        group_map = {g: idx[groups == g] for g in np.unique(groups)}
        return group_map
    
    def calculate_loss(self):
        loss = np.sum(np.power(self.y_ - np.matmul(self.X_, self.coef_), 2))
        for g in self.groups_:
            group_select = self.groups_[g]
            group_size = len(group_select)
            group_norm = np.linalg.norm(self.coef_[group_select])
            loss += self.l2_penalty * np.sqrt(group_size) * group_norm
            
        return loss
        
    def fit(self, X, groups, y):
        # Store groups
        self.groups_ = self.preprocess_groups(groups)
        
        # Check X and y
        self.X_, self.y_ = X, np.atleast_2d(y).T

        # Set coefficients to zero
        self.coef_ = 10 * np.abs(np.random.randn(self.X_.shape[1], 1)) #np.zeros([self.X_.shape[1], 1])
        best_loss = np.inf
        best_coef = None
        tracker = 2
        for i in range(self.max_iter):
            cur_loss = self.calculate_loss()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coef = self.coef_.copy()
                tracker = 2
            else:
                tracker -= 1
                if tracker == 0:
                    break
            
            loss_grad = -1. * np.matmul(X.T, (self.y_ - np.matmul(self.X_, self.coef_)))
            
            for g in self.groups_:
                group_select = self.groups_[g]
                group_size = len(group_select)
                if np.any(self.coef_[group_select] > 0.):
                    penalty_grad = self.coef_[self.groups_[g]] / np.linalg.norm(self.coef_[self.groups_[g]])
                    loss_grad[group_select] += self.l2_penalty * np.sqrt(group_size) * penalty_grad
                else:
                    grad_norm = np.linalg.norm(loss_grad[group_select])
                    grad_scale = max(grad_norm - self.l2_penalty * np.sqrt(group_size), 0.) / grad_norm
                    loss_grad[group_select] = grad_scale * loss_grad[group_select]
                    
            self.coef_ -= self.step_size * loss_grad
            self.coef_ = np.maximum(self.coef_, 0.)
            
        self.coef_ = best_coef.copy()
        self.loss_ = best_loss
        
    def predict(self):
        return np.matmul(self.X_, self.coef_) 

    
class GradSGL:
    def __init__(self, l2_penalty, l1_penalty, step_size=5e-10, max_iter=100):

        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.step_size = step_size
        self.max_iter = max_iter
        self.X_ = None
        self.y_ = None
        self.group_map_ = None
        self.coef_ = None
        self.loss_ = None
        
    def preprocess_groups(self, groups):
        idx = np.arange(len(groups))
        group_map = {g: idx[groups == g] for g in np.unique(groups)}
        return group_map
    
    def calculate_loss(self):
        loss = np.sum(np.power(self.y_ - np.matmul(self.X_, self.coef_), 2))
        for g in self.groups_:
            group_select = self.groups_[g]
            group_size = len(group_select)
            group_norm = np.linalg.norm(self.coef_[group_select])
            loss += self.l2_penalty * np.sqrt(group_size) * group_norm
        loss += self.l1_penalty * np.sum(self.coef_)
            
        return loss
        
    def fit(self, X, groups, y):
        # Store groups
        self.groups_ = self.preprocess_groups(groups)
        
        # Check X and y
        self.X_, self.y_ = X, np.atleast_2d(y).T

        # Set coefficients to zero
        self.coef_ = 10 * np.abs(np.random.randn(self.X_.shape[1], 1)) #np.zeros([self.X_.shape[1], 1])
        best_loss = np.inf
        best_coef = None
        tracker = 2
        for i in range(self.max_iter):
            cur_loss = self.calculate_loss()
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coef = self.coef_.copy()
                tracker = 2
            else:
                tracker -= 1
                if tracker == 0:
                    break
            
            loss_grad = -1. * np.matmul(X.T, (self.y_ - np.matmul(self.X_, self.coef_)))
            
            for g in self.groups_:
                group_select = self.groups_[g]
                group_size = len(group_select)
                if np.any(self.coef_[group_select] > 0.):
                    penalty_grad = self.coef_[self.groups_[g]] / np.linalg.norm(self.coef_[self.groups_[g]])
                    loss_grad[group_select] += self.l2_penalty * np.sqrt(group_size) * penalty_grad
                else:
                    grad_norm = np.linalg.norm(loss_grad[group_select])
                    grad_scale = max(grad_norm - self.l2_penalty * np.sqrt(group_size), 0.) / grad_norm
                    loss_grad[group_select] = grad_scale * loss_grad[group_select]
            
            # l1 penalty
            coef_zero = self.coef_ == 0
            grad_zero = loss_grad == 0
            defined = np.array(np.logical_and(~coef_zero, ~grad_zero)).flatten()
            if np.any(defined):
                signs = np.sign(self.coef_[defined])
                loss_grad[defined] += signs * self.l1_penalty
                
            undefined = np.array(np.logical_and(coef_zero, ~grad_zero)).flatten()
            if np.any(undefined):
                zero_pos = np.array(loss_grad[undefined])
                scale = np.maximum(np.abs(zero_pos) - self.l1_penalty, 0.) / np.abs(zero_pos)
                loss_grad[undefined] = scale * zero_pos
                    
            self.coef_ -= self.step_size * loss_grad
            self.coef_ = np.maximum(self.coef_, 0.)
            
        self.coef_ = best_coef.copy()
        self.loss_ = best_loss
        
    def predict(self):
        return np.matmul(self.X_, self.coef_)  
