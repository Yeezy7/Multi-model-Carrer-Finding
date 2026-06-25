import torch

if __name__ == "__main__":
    # Check for MPS (Apple Silicon GPU) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS (Apple Silicon GPU) is available.")
    # Fallback to CPU if MPS is not available
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    
    print(f"Using device: {device}")