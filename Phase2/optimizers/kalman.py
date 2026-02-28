import torch
from torch.optim import Optimizer
import torch.autograd as autograd

class FDEKF(Optimizer):
    def __init__(self, params, p0_var=0.1, r_var=0.01, q_var=1e-6):
        defaults = dict(p0_var=p0_var, r_var=r_var, q_var=q_var)
        super(FDEKF, self).__init__(params, defaults)
        
        # Initialize P (Error Covariance)
        for group in self.param_groups:
            p0 = group['p0_var']
            for p in group['params']:
                state = self.state[p]
                state['P'] = torch.full_like(p.real, p0, dtype=torch.float32)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("FDEKF requires a closure that returns (output, target)")

        with torch.enable_grad():
            output, target = closure()
            innovation = target - output 
            out_re = output.real.sum()
            out_im = output.imag.sum()
            
        # Collect all parameters requiring gradients
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    all_params.append(p)
                    
        # Compute Jacobians for all parameters in one go (Efficiency!)
        # H_re = d(y_re)/d(theta), H_im = d(y_im)/d(theta)
        grads_re = autograd.grad(out_re, all_params, retain_graph=True, allow_unused=True)
        grads_im = autograd.grad(out_im, all_params, retain_graph=False, allow_unused=True)
        
        param_idx = 0
        for group in self.param_groups:
            R = group['r_var']
            Q = group['q_var']
            
            for p in group['params']:
                if not p.requires_grad:
                    continue
                
                h_re = grads_re[param_idx]
                h_im = grads_im[param_idx]
                param_idx += 1
                
                if h_re is None: h_re = torch.zeros_like(p)
                if h_im is None: h_im = torch.zeros_like(p)
                
                state = self.state[p]
                P = state['P']
                h = torch.complex(h_re, h_im)
                
                self._kalman_update_scalar(p, P, h, innovation, R, Q)
                    
    def _kalman_update_scalar(self, param, P, h, innovation, R, Q):
        # param: tensor (modified in place)
        # P: tensor (variance, modified in place)
        # h: tensor (Jacobian, complex)
        # innovation: scalar (complex)
        # R, Q: scalars
        
        # 1. Predict P
        P_pred = P + Q
        
        # 2. Innovation Covariance S
        # S = H P H* + R
        # h is complex, P is real variance
        S = (torch.abs(h) ** 2) * P_pred + R
        
        # 3. Kalman Gain K
        # K = P H* / S
        K = P_pred * torch.conj(h) / S
        
        # 4. Update State
        # theta = theta + K * y_tilde
        update = K * innovation
        
        # Clipping for stability in nonlinear environments
        max_delta = 0.001
        with torch.no_grad():
            update_delta = torch.clamp(update.real, -max_delta, max_delta)
            param.add_(update_delta) # param is real here
            
        # 5. Update Covariance
        # P = (1 - K H) P
        # K H = (P H* / S) H = P |H|^2 / S
        # This is real.
        KH = K * h
        # Theoretically real, but numerically might have small imag. Take real.
        # Ensure KH is not too large
        KH_real = torch.clamp(KH.real, 0, 0.99)
        P_new = (1 - KH_real) * P_pred
        
        with torch.no_grad():
            P.copy_(P_new)

class RRFDEKF(FDEKF):
    """
    Round-Robin FDEKF.
    Updates parameters sequentially? 
    In PyTorch, 'step' iterates sequentially over parameters anyway.
    The difference is that RR-FDEKF uses the *updated* values of previous params 
    to compute the residual for the next param?
    That requires re-running the model forward pass after EACH parameter update.
    That is O(N^2) relative to forward pass cost. Very slow for Deep Learning.
    
    Paper says: "Updates only a single parameter theta_i using per measurement".
    "We structure ... around logical processing cycles."
    "RR-FDEKF which only updates a single parameter theta_i using per measurement."
    
    Ah! It means for measurement k, we ONLY update parameter i.
    Next measurement k+1, we update parameter i+1.
    This reduces complexity to O(1) per sample!
    
    This is very different.
    """
    def __init__(self, params, p0_var=0.1, r_var=0.01, q_var=1e-6, num_params=None):
        super(RRFDEKF, self).__init__(params, p0_var, r_var, q_var)
        self.param_list = []
        for group in self.param_groups:
            for p in group['params']:
                self.param_list.append(p)
        self.curr_idx = 0
        
    def step(self, closure=None):
        if closure is None: raise RuntimeError("Closure required")
        
        # 1. Forward & Jacobian (Same as FDEKF)
        # But we only need Jacobian for the CURRENT parameter index.
        with torch.enable_grad():
            output, target = closure()
            innovation = target - output
            out_re = output.real.sum()
            out_im = output.imag.sum()
            
        # Select current parameter to update
        # Need to handle flattening? Or just iterate list of tensors?
        # A tensor might have many elements. 
        # "Parameter" in paper usually means scalar weight.
        # In NN, "Parameter" is a tensor (W matrix).
        # Should we update the whole tensor? Or one element of tensor?
        # Efficient implementation: Update the whole Tensor block (Layer) 
        # or iterate scalar elements?
        # Paper implies scalar round robin.
        # For efficiency in PyTorch, maybe Round-Robin over LAYERS or Groups?
        # Or just update ONE tensor from the list per step?
        # Let's update ONE Tensor per step.
        
        p = self.param_list[self.curr_idx]
        self.curr_idx = (self.curr_idx + 1) % len(self.param_list)
        
        # Update logic for this tensor p
        # (Same as FDEKF but only for p)
        group = self.param_groups[0] # Assume 1 group
        R = group['r_var']
        Q = group['q_var']
        
        if p.grad is None: 
            # Force grad computation for this p
             # ...
             pass
             
        # Actually we need to call autograd.grad for this specific p
        grad_re = autograd.grad(out_re, p, retain_graph=False, allow_unused=True)[0]
        grad_im = autograd.grad(out_im, p, retain_graph=False, allow_unused=True)[0]
        
        if grad_re is None: grad_re = torch.zeros_like(p)
        if grad_im is None: grad_im = torch.zeros_like(p)
        
        state = self.state[p]
        P = state['P']
        
        if p.is_complex():
             # ... (Similar logic, need separate P for real/imag or combined)
             # Let's implement real-only support for now, or assume model uses Real/Imag parameters explicitly
             # Our ComplexLinear uses Real and Imag parameters separately!
             # So p is always real (weight_re, weight_im).
             pass
        
        # Apply update
        h = grad_re + 1j * grad_im
        self._kalman_update_scalar(p, P, h, innovation, R, Q)

