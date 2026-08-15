import os
import torch
import argparse
from model import AGA

def load_weights(model, checkpoint_path):
    """
    Loads weights from a catorch or pure PyTorch checkpoint into the AGA model.
    Strips out 'catorch' pipeline prefixes like 'graph.P.' if present.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle dict with 'state_dict' key or direct state_dict
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # catorch pipeline usually prefixes weights with 'graph.<NodeName>.'
        # In inference.ipynb, the Eyra.CausalHORNN node is named 'P', so keys start with 'graph.P.'
        if k.startswith('graph.P.'):
            new_key = k[len('graph.P.'):]
        elif k.startswith('P.'):
            new_key = k[len('P.'):]
        elif k.startswith('model.'):  # Standard DDP or wrapper prefix
            new_key = k[len('model.'):]
        else:
            new_key = k
            
        new_state_dict[new_key] = v
        
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"[Warning] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys: {unexpected_keys}")
        
    print("Weights loaded successfully.")
    return model

def main():
    parser = argparse.ArgumentParser(description="AGA Model Inference (No catorch dependency)")
    parser.add_argument('--checkpoint', type=str, default='ek100_checkpoint', help='Path to the model checkpoint')
    parser.add_argument('--in_dim', type=int, default=1024, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=3806, help='Output dimension (number of classes)')
    parser.add_argument('--order', type=int, default=30, help='Sequence length or order limit')
    parser.add_argument('--recurrent_query', type=str, default='ma', help='Recurrent query type')
    parser.add_argument('--ma_ratio', type=float, default=0.8, help='Moving average ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    args = parser.parse_args()

    # 1. Initialize the model
    # Matches: AGA(in_dim=1024, hidden_dim=2048, out_dim=3806, order=30, recurrent_query='ma', ma_ratio=0.8, recurrent_h=False, gate_x=True)
    model = AGA(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        order=args.order,
        recurrent_query=args.recurrent_query,
        ma_ratio=args.ma_ratio,
        recurrent_h=False,
        gate_x=True
    )
    
    # 2. Load weights if a checkpoint is provided and exists
    if os.path.exists(args.checkpoint):
        model = load_weights(model, args.checkpoint)
    else:
        print(f"[Warning] Checkpoint file {args.checkpoint} not found. Running with randomly initialized weights.")

    model = model.to(args.device)
    model.eval()

    # 3. Simulate an inference pass with dummy data
    print(f"Creating dummy tensor for inference. Batch size: 2, Sequence Length: {args.order}, Input Dim: {args.in_dim}")
    dummy_input = torch.randn(2, args.order, args.in_dim).to(args.device)

    print("Running forward pass...")
    with torch.no_grad():
        # The model returns predictions for the sequence.
        # Shape depends on return_embedding. By default return_embedding=False, so returns shape (B, SeqLen, OutDim)
        output = model(dummy_input)

    print(f"Forward pass successful. Output shape: {output.shape}")
    
if __name__ == '__main__':
    main()
