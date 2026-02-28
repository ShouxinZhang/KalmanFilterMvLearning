import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy.signal import correlate
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.bfan import SymmetrizedBFAN
from optimizers.kalman import FDEKF, RRFDEKF

# ==========================================
# Data Loading & Sync (From Phase 1)
# ==========================================
def load_data():
    mat_path = '/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/FDEKF/SISO.mat'
    mat = sio.loadmat(mat_path)
    
    # 1. Load Raw
    tx = mat['txa'].flatten() # (N,) Complex
    
    # Use NFA as Reference PIM
    nfa = mat['nfa'].flatten()
    
    # Truncate to min length
    n = min(len(tx), len(nfa))
    tx = tx[:n]
    nfa = nfa[:n]
    # Calculated Freq Diff: 1842.48 - 1750 = 92.48 MHz
    # Sampling: 245.76 MHz
    # Normalized: 0.3763
    freq_rot = 0.37630208
    t = np.arange(len(tx))
    tx_rotated = tx * np.exp(-1j * 2 * np.pi * freq_rot * t)
    
    # 3. Lag Alignment
    # Determine lag
    # Envelope correlation
    env_tx = np.abs(tx_rotated)
    env_rx = np.abs(nfa)
    corr = correlate(env_rx - np.mean(env_rx), env_tx - np.mean(env_tx), mode='full')
    lags = np.arange(-len(env_tx) + 1, len(env_tx))
    best_lag = lags[np.argmax(corr)]
    
    # Apply lag
    if best_lag > 0:
        tx_aligned = np.roll(tx_rotated, best_lag)
        tx_aligned[:best_lag] = 0
    else:
        tx_aligned = np.roll(tx_rotated, best_lag)
        tx_aligned[best_lag:] = 0
        
    # 4. Phase Synchronization
    # De-rotate TX to match NFA phase
    # Smooth phase difference
    phase_diff = np.angle(nfa) - np.angle(tx_aligned)
    phase_unwrapped = np.unwrap(phase_diff)
    # Moving average
    win_len = 1000
    phase_smooth = np.convolve(phase_unwrapped, np.ones(win_len)/win_len, mode='same')
    
    tx_sync = tx_aligned * np.exp(1j * phase_smooth)
    
    # Normalize TX
    scale_tx = 1.0 / np.max(np.abs(tx_sync))
    tx_sync *= scale_tx
    
    # Normalize RX independently (Target Normalization)
    scale_rx = 1.0 / np.max(np.abs(nfa))
    nfa_train = nfa * scale_rx
    
    return tx_sync, nfa_train, scale_rx

# ==========================================
# Training
# ==========================================
def prepare_input_features(tx, memory_depth):
    # Create (N, Memory) tensor
    # x[n] = [tx[n], tx[n-1], ..., tx[n-M+1]]
    N = len(tx)
    # Pad beginning
    padded = np.pad(tx, (memory_depth, 0), mode='constant')
    
    # Window
    # Shape: (N, Memory)
    # Stride tricks for efficiency (numpy)
    # strides = (itemsize, itemsize)
    itemsize = padded.itemsize
    shape = (N, memory_depth)
    strides = (itemsize, itemsize)
    
    # Actually stride should be reversed for [n, n-1, ...] or standard?
    # Usual sliding window: [0, 1, 2], [1, 2, 3] -> This is [n-M, ..., n]
    # We want [n, n-1, ..., n-M] usually for FIR?
    # ComplexLinear doesn't care about order as long as consistent.
    # Let's use standard sliding window [n, n+1, ...] of the padded array?
    # Let's generate [tx[i], tx[i+1]... tx[i+M]] from padded (offset).
    # Easier: Use torch.unfold later or simple list comprehension if N is small (64k is small)
    
    features = []
    for i in range(N):
        # We want window ending at i.
        # Index in padded: i + memory_depth
        # Window: padded[i : i+memory_depth] -> [tx[i-M] ... tx[i]]
        # We need to reverse it to have tx[i] first? Doesn't matter for learnable weights.
        win = padded[i : i+memory_depth]
        features.append(win)
        
    return np.array(features, dtype=np.complex64) # Float32 complex

def train():
    # 1. Load
    tx, rx, scale_rx = load_data()
    print(f"Data Loaded: TX {tx.shape}, RX {rx.shape}")
    
    MEMORY = 15
    KNOTS = 50
    HIDDEN = 10
    
    # 2. Features
    X_np = prepare_input_features(tx, MEMORY)
    
    # Convert to Tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.from_numpy(X_np).to(device) # (N, Mem)
    y_tensor = torch.from_numpy(rx).to(device).to(torch.complex64)
    # rx is np.complex128 usually
    
    N = len(tx)
    
    # 3. Model
    model = SymmetrizedBFAN(memory_depth=MEMORY, hidden_dim=HIDDEN, num_knots=KNOTS).to(device)
    
    # 4. Adam Warm-up
    print("Starting Adam Warm-up (50 epochs)...")
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    indices = np.arange(len(tx))
    BATCH_SIZE = 1024
    for epoch in range(50):
        np.random.shuffle(indices)
        for i in range(0, len(tx), BATCH_SIZE):
            batch_idx = indices[i : i+BATCH_SIZE]
            x_b = X_tensor[batch_idx]
            y_b = y_tensor[batch_idx]
            
            adam_optimizer.zero_grad()
            pred = model(x_b).squeeze()
            loss = criterion(pred.real, y_b.real) + criterion(pred.imag, y_b.imag)
            loss.backward()
            adam_optimizer.step()
        if epoch % 10 == 0:
            print(f"Warm-up Epoch {epoch} complete.")

    # 5. Optimizer (EKF Fine-tuning)
    # Conservative parameters for fine-tuning
    optimizer = FDEKF(model.parameters(), p0_var=0.001, r_var=0.1, q_var=1e-7)
    
    residuals = []
    
    print("Starting EKF Fine-tuning (Sample-by-Sample)...")
    model.train()
    
    for i in range(N):
        x_sample = X_tensor[i:i+1] # (1, Mem)
        y_sample = y_tensor[i]    # Scalar
        
        # Closure
        def closure():
            optimizer.zero_grad()
            pred = model(x_sample).squeeze() # Scalar
            return pred, y_sample
        
        # Step
        optimizer.step(closure)
        
        # Compute residual for tracking
        with torch.no_grad():
            pred = model(x_sample).squeeze()
            res = y_sample - pred
            residuals.append(res.item())
            
        if i % 5000 == 0:
            print(f"Step {i}/{N}, Res: {np.abs(res.item()):.5f}")
            
    # Plot
    # Scale back for PSD comparison
    residuals = np.array(residuals) / scale_rx
    rx_target = rx / scale_rx
    
    print(f"Final Residual Mean: {np.mean(np.abs(residuals))}")
    
    plt.figure(figsize=(10, 6))
    plt.psd(rx_target, NFFT=1024, Fs=1.0, label='Reference PIM', color='black', linestyle='--')
    plt.psd(residuals, NFFT=1024, Fs=1.0, label='FDEKF-BFAN Residual', color='red')
    plt.title('Phase 2: Full Network FDEKF Training')
    plt.legend()
    plt.grid(True)
    plt.ylim([-100, 0])
    plt.savefig('phase2_result.png')
    print("Saved phase2_result.png")

if __name__ == "__main__":
    train()
