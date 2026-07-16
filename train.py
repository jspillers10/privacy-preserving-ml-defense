
"""
Privacy-Preserving Fashion-MNIST Classifier Training
 
Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
to defend against membership inference attacks.
"""
 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from safetensors.torch import save_file
from model import FashionMNISTCNN
 
 
# Privacy Configuration
EPSILON = 3.0              # Privacy budget (lower = more private)
DELTA = 1e-5               # Privacy failure probability
MAX_GRAD_NORM = 1.0        # Gradient clipping threshold
NOISE_MULTIPLIER = 1.1     # Gaussian noise scale
 
# Training Configuration
EPOCHS = 20
BATCH_SIZE = 256           # Larger batch = better privacy-utility ratio
LEARNING_RATE = 0.001
 
 
def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
 
    # Data normalization (must match preprocessing used at evaluation time)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
 
    # Load Fashion-MNIST dataset
    print("\nLoading Fashion-MNIST dataset...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
 
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
 
    # Initialize model
    model = FashionMNISTCNN().to(device)
 
    # Make model compatible with Opacus
    model = ModuleValidator.fix(model)
 
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    # Attach Privacy Engine for DP-SGD
    print("\n" + "="*60)
    print("DIFFERENTIAL PRIVACY CONFIGURATION")
    print("="*60)
    print(f"  Target epsilon (ε): {EPSILON}")
    print(f"  Delta (δ): {DELTA}")
    print(f"  Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"  Max gradient norm: {MAX_GRAD_NORM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")
 
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
    )
 
    # Training loop
    print("Starting training with DP-SGD...\n")
 
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
 
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
 
            train_loss += loss.item()
 
            # Progress update every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
 
        avg_train_loss = train_loss / len(train_loader)
 
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
 
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
 
        accuracy = correct / total
 
        # Get current privacy budget
        epsilon = privacy_engine.get_epsilon(delta=DELTA)
 
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Average Loss: {avg_train_loss:.4f}")
        print(f"  Test Accuracy: {accuracy*100:.2f}%")
        print(f"  Privacy Budget: (ε={epsilon:.4f}, δ={DELTA})")
        print(f"{'='*60}\n")
 
    # Save model in safetensors format
    print("Saving model...")
 
    # Extract state dict (remove DP wrapper if present)
    state_dict = model._module.state_dict() if hasattr(model, '_module') else model.state_dict()
    save_file(state_dict, 'defended_model.safetensors')
 
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"  Final Privacy Budget: (ε={epsilon:.4f}, δ={DELTA})")
    print(f"  Model saved: defended_model.safetensors")
    print("="*60)
 
 
if __name__ == "__main__":
    main()
