import torch
import numpy as np
import time
from stable_pooling import StableHierarchicalPooling
from balanced_features import BalancedFeatureTransformer

def test_node_revival():
    print("Testing node revival in StableHierarchicalPooling...")
    in_channels = 16
    n_super_nodes = 4
    pooling = StableHierarchicalPooling(in_channels, n_super_nodes, pruning_threshold=0.1)
    
    # Force one node to be "dead" by setting its active_mask to 0
    pooling.active_mask[0] = 0.0
    
    # Create input that SHOULD favor the dead node
    # The MLP is randomized, but we can check if gradients can flow
    x = torch.randn(10, in_channels, requires_grad=True)
    batch = torch.zeros(10, dtype=torch.long)
    
    # Try to "revive" it by maximizing its assignment probability
    optimizer = torch.optim.Adam(pooling.parameters(), lr=0.1)
    
    for i in range(5):
        optimizer.zero_grad()
        out, s, losses, mu = pooling(x, batch, tau=1.0)
        
        # Target: maximize assignment to node 0
        loss = -s[:, 0].mean()
        loss.backward()
        optimizer.step()
        
        # Check if active_mask is updated
        # In the original implementation, it uses EMA of (avg_s > threshold)
        # If logits are -1e9, s[:, 0] will be 0, so it will never be > threshold
        print(f"Step {i}, Assignment prob to node 0: {s[:, 0].mean().item():.6f}, Mask: {pooling.active_mask.tolist()}")

    # If it's still 0 or near 0, it's "dead"
    if pooling.active_mask[0] < 0.1:
        print("RESULT: Node 0 is still DEAD (as expected in original implementation)")
    else:
        print("RESULT: Node 0 was REVIVED!")

def test_feature_transformer_scalability():
    print("\nTesting FeatureTransformer scalability...")
    n_super_nodes = 20 # The threshold mentioned in the analysis
    latent_dim = 8
    batch_size = 100
    
    latent_states = np.random.randn(batch_size, n_super_nodes * latent_dim)
    targets = np.random.randn(batch_size, 1)
    
    transformer = BalancedFeatureTransformer(n_super_nodes, latent_dim)
    
    start_time = time.time()
    # Mocking fit to just test transform performance
    transformer.fit(latent_states, targets)
    fit_time = time.time() - start_time
    
    print(f"Fit time for {n_super_nodes} nodes: {fit_time:.4f}s")
    
    start_time = time.time()
    features = transformer.transform(latent_states)
    transform_time = time.time() - start_time
    
    print(f"Transform time: {transform_time:.4f}s")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature matrix size: {features.nbytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    test_node_revival()
    test_feature_transformer_scalability()