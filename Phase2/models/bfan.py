import torch
import torch.nn as nn
import numpy as np

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and Imaginary weights
        self.weight_re = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_im = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_re = nn.Parameter(torch.Tensor(out_features))
            self.bias_im = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_re', None)
            self.register_parameter('bias_im', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # Scale to ensure mixed signal reaches nonlinear basis [0, 0.2]
        scale = 0.5 
        nn.init.kaiming_uniform_(self.weight_re, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_im, a=np.sqrt(5))
        with torch.no_grad():
            self.weight_re.mul_(scale)
            self.weight_im.mul_(scale)
            
        if self.bias_re is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_re)
            bound = (1 / np.sqrt(fan_in)) * scale
            nn.init.uniform_(self.bias_re, -bound, bound)
            nn.init.uniform_(self.bias_im, -bound, bound)

    def forward(self, x):
        # x is (Batch, In, 2) or (Batch, In) complex
        if x.is_complex():
            input_re, input_im = x.real, x.imag
        else:
            # Assume last dim is 2 (Re, Im)
            input_re = x[..., 0]
            input_im = x[..., 1]
            
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # W * x = (Wr + iWi)(xr + ixi) = (Wrxr - Wixi) + i(Wrxi + Wixr)
        
        out_re = torch.nn.functional.linear(input_re, self.weight_re, self.bias_re) - \
                 torch.nn.functional.linear(input_im, self.weight_im, None)
                 
        out_im = torch.nn.functional.linear(input_re, self.weight_im, self.bias_im) + \
                 torch.nn.functional.linear(input_im, self.weight_re, None) # bias added once per component
                 
        # If bias exists, we added Re bias to Re part, Im bias to Im part.
        # But logic above:
        # Re: Wr*xr + br - Wi*xi = Wr*xr - Wi*xi + br. Correct.
        # Im: Wr*xi + bi + Wi*xr = Wr*xi + Wi*xr + bi. Correct.
        
        return torch.complex(out_re, out_im)

class SymmetrizedBasis(nn.Module):
    def __init__(self, num_knots=20, max_val=1.0):
        super(SymmetrizedBasis, self).__init__()
        # Knots are fixed parameters (or learnable?)
        # For simplicity, fixed uniform knots initially to match paper's likely LUT approach
        # Paper uses odd symmetry: h(x) - h(-x)
        # We only need positive knots because negative ones are mirrored.
        
        self.num_knots = num_knots
        # Use dense knots in the low-power region where PIM lives
        # Spacing: 0.2 / 50 = 0.004 (resolves 0.05 amplitude well)
        knots = torch.linspace(0.0, 0.2, num_knots)
        self.register_buffer('knots', knots)
        
    def forward(self, x):
        # x is real-valued inputs (from the rotated projection)
        # x shape: (Batch, Num_Features)
        # Output shape: (Batch, Num_Features * Num_Knots)
        
        # We need to apply basis to EACH feature channel.
        # Input x: (B, F)
        # Knots: (K,)
        
        B, F = x.shape
        K = self.num_knots
        
        x_expanded = x.unsqueeze(-1) # (B, F, 1)
        knots_expanded = self.knots.view(1, 1, K) # (1, 1, K)
        
        # Basis: f_k(u) = ReLU(u - xi) - ReLU(-u - xi)
        
        # Term 1: ReLU(u - xi)
        term1 = torch.relu(x_expanded - knots_expanded)
        
        # Term 2: ReLU(-u - xi)
        term2 = torch.relu(-x_expanded - knots_expanded)
        
        basis_out = term1 - term2 # (B, F, K)
        
        # Flatten feature and knot dimensions for linear combination
        return basis_out.view(B, F * K)

class SymmetrizedBFAN(nn.Module):
    def __init__(self, memory_depth=5, hidden_dim=4, num_knots=20):
        super(SymmetrizedBFAN, self).__init__()
        
        # 1. Linear Mixing (Phi)
        # Input is a memory vector of complex samples.
        # We mix them into 'hidden_dim' signals.
        # This rotation/mixing allows the model to find the 'Principal nonlinear axes'
        
        self.mixing = ComplexLinear(memory_depth, hidden_dim, bias=False)
        
        # 2. Nonlinearity (Basis Expansion)
        # We treat Real and Imag parts of the mixed signal as separate scalar inputs to the basis
        # (Canonical approach for complex nonlinearity: act on components)
        self.basis = SymmetrizedBasis(num_knots=num_knots)
        
        # 3. Readout (Linear Combination)
        # Input dim = hidden_dim * 2 (Re/Im) * num_knots
        self.readout_dim = hidden_dim * 2 * num_knots
        self.readout = ComplexLinear(self.readout_dim, 1, bias=False) # Output is scalar PIM estimate
        
    def forward(self, x):
        # x: (Batch, Memory_Depth) Complex
        
        # 1. Mix/Rotate
        hidden = self.mixing(x) # (Batch, hidden_dim) Complex
        
        # 2. Split Re/Im
        h_re = hidden.real
        h_im = hidden.imag
        
        # 3. Apply Basis (Symmetrized ReLU)
        # The basis expansion is done on real components
        feat_re = self.basis(h_re) # (Batch, hidden_dim * K)
        feat_im = self.basis(h_im) # (Batch, hidden_dim * K)
        
        # 4. Concatenate for readout
        # Readout is a Complex Linear Layer.
        # It expects a complex input vector.
        # We have real basis features.
        # We can treat them as the Real part of a complex vector (Imag=0)?
        # Or we construct a "Complex Feature Vector"
        # The paper says: y = Sum w_k * f_k(a)
        # w_k are complex weights.
        # f_k(a) are real values (ReLU outputs).
        # So we can feed `features` as Real-valued complex numbers (Imag=0) to ComplexLinear?
        # Or just implement custom readout: y = (W_re + jW_im) * Features
        
        features = torch.cat([feat_re, feat_im], dim=1) # (Batch, hidden_dim * 2 * K)
        # Treat as complex with zero imaginary part
        features_complex = torch.complex(features, torch.zeros_like(features))
        
        out = self.readout(features_complex)
        
        return out
