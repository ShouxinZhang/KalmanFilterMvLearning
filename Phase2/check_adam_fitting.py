import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy.signal import correlate
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.bfan import SymmetrizedBFAN

def load_data():
    mat_path = '/home/wudizhe001/Documents/GitHub/KalmanFilterMvLearning/FDEKF/SISO.mat'
    mat = sio.loadmat(mat_path)
    tx = mat['txa'].flatten()
    nfa = mat['nfa'].flatten()
    n = min(len(tx), len(nfa))
    tx, nfa = tx[:n], nfa[:n]
    
    # Sync (MA-1000)
    freq_rot = 0.37630208
    t = np.arange(len(tx))
    tx = tx * np.exp(-1j * 2 * np.pi * freq_rot * t)
    corr = correlate(np.abs(nfa) - np.mean(np.abs(nfa)), np.abs(tx) - np.mean(np.abs(tx)), mode='full')
    lag = np.arange(-len(tx)+1, len(tx))[np.argmax(corr)]
    tx = np.roll(tx, lag)
    if lag > 0: tx[:lag] = 0
    else: tx[lag:] = 0
    
    phase_diff = np.unwrap(np.angle(nfa) - np.angle(tx))
    phase_smooth = np.convolve(phase_diff, np.ones(100)/100, mode='same') # Faster window for diagnostics
    tx = tx * np.exp(1j * phase_smooth)
    
    scale = 1.0 / np.max(np.abs(tx))
    return tx * scale, nfa * scale

def prepare_input_features(tx, memory_depth):
    N = len(tx)
    padded = np.pad(tx, (memory_depth, 0), mode='constant')
    features = []
    for i in range(N):
        features.append(padded[i : i+memory_depth])
    return np.array(features, dtype=np.complex64)

def diag_adam():
    tx, rx = load_data()
    MEMORY = 15
    HIDDEN = 20 # More capacity for diag
    KNOTS = 20
    
    X = prepare_input_features(tx, MEMORY)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(rx).to(device).to(torch.complex64)
    
    model = SymmetrizedBFAN(memory_depth=MEMORY, hidden_dim=HIDDEN, num_knots=KNOTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Batch training for stability
    BATCH_SIZE = 1024
    EPOCHS = 20
    
    indices = np.arange(len(tx))
    
    print("Pre-training BFAN with Adam...")
    for epoch in range(EPOCHS):
        np.random.shuffle(indices)
        total_loss = 0
        for i in range(0, len(tx), BATCH_SIZE):
            batch_idx = indices[i : i+BATCH_SIZE]
            x_b = X_t[batch_idx]
            y_b = y_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(x_b).squeeze()
            loss = criterion(pred.real, y_b.real) + criterion(pred.imag, y_b.imag)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss / (len(tx)/BATCH_SIZE):.6f}")

    # Final Check
    model.eval()
    with torch.no_grad():
        y_pred = model(X_t).squeeze()
        res = y_t - y_pred
        res_np = res.cpu().numpy()
        rx_np = rx
        
    plt.figure(figsize=(10, 6))
    plt.psd(rx_np, NFFT=1024, Fs=1.0, label='Reference PIM')
    plt.psd(res_np, NFFT=1024, Fs=1.0, label='Adam-BFAN Residual')
    plt.title('Diagnostic: Can BFAN fit the data? (Adam)')
    plt.ylim([-100, -20])
    plt.legend()
    plt.savefig('diag_adam.png')
    print("Saved diag_adam.png")

if __name__ == "__main__":
    diag_adam()
