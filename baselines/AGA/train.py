import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import lmdb
import pickle

from model import AGA
from validate import validate_ek100

class EK100Dataset(Dataset):
    def __init__(self, root_folder, train=True, sequence_length=30, time_step=1.0, random_drift_sampling=False, jitter_mix=1):
        super().__init__()
        self.root_folder = root_folder
        self.train = train
        self.sequence_length = sequence_length
        self.time_step = time_step
        self.random_drift_sampling = random_drift_sampling
        self.jitter_mix = jitter_mix
        
        csv_file = 'training.csv' if train else 'validation.csv'
        self.annotations = pd.read_csv(os.path.join(root_folder, csv_file), header=None, engine='c')
        
        label_file = 'training_frame_labelings' if train else 'validation_frame_labelings'
        with open(os.path.join(root_folder, label_file), 'rb') as f:
            self.frame_labels = pickle.load(f)
            
        self.env = lmdb.open(os.path.join(root_folder, 'rgb'), readonly=True, lock=False)
        self.anticipate_steps = np.arange(time_step * sequence_length, 0, -time_step)
        
        self.samples = []
        for _, record in self.annotations.iterrows():
            _, name, video, start, end, verb, noun, action, source_fps = record
            
            relative_steps = np.ceil(self.anticipate_steps * source_fps).astype(int)
            frames = start - relative_steps
            f_max, f_min = frames.max(), frames.min()
            
            if f_max > 0:
                if f_max >= start:
                    frames[frames >= start] = frames[frames < start].max()
                if f_min < 1:
                    frames[frames < 1] = frames[frames > 0].min()
                
                # We store raw integer frames and the video string to handle random_drift_sampling dynamically
                self.samples.append((frames, name, verb, noun, action, video, source_fps))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, name, verb, noun, action, video, source_fps = self.samples[idx]
        seq = []
        all_labels = []
        start_frame_name = f'{video}_frame_{int(frames[0]):010d}.jpg'

        with self.env.begin(write=False) as txn:
            for i, x in enumerate(frames):
                frame_name = f'{video}_frame_{int(x):010d}.jpg'

                # Apply random_drift_sampling logic from catorch mapping
                if self.train and self.random_drift_sampling and frame_name != start_frame_name:
                    mix = np.zeros(1024, dtype=np.float32)
                    for drift in np.random.randint(0, int(source_fps * self.time_step), self.jitter_mix):
                        k = f'{video}_frame_{int(max(1, x - drift)):010d}.jpg'
                        data = txn.get(k.encode('utf-8'))
                        if data is not None:
                            mix += np.frombuffer(data, 'float32')
                    seq.append(mix / self.jitter_mix)
                else:
                    data = txn.get(frame_name.encode('utf-8'))
                    if data is None:
                        seq.append(np.zeros(1024, dtype=np.float32))
                    else:
                        seq.append(np.frombuffer(data, 'float32'))
                        
                # Sequence step labels
                if frame_name in self.frame_labels:
                    all_labels.append(self.frame_labels[frame_name])
                else:
                    all_labels.append((-1, -1, -1))

        # Append final action to sequence labels
        all_labels.append((verb, noun, action))
        
        seq_tensor = torch.FloatTensor(np.stack(seq))
        lbl_tensor = torch.LongTensor(np.array(all_labels)) # (SeqLen+1, 3) where 3 is (verb, noun, action)

        return {
            'x': seq_tensor,
            'id': name,
            'AY': lbl_tensor[:, 2], # Action sequence
            'VY': lbl_tensor[:, 0], # Verb sequence  
            'NY': lbl_tensor[:, 1], # Noun sequence
            'LastAntY': action,
            'LastAntVY': verb,
            'LastAntNY': noun
        }

def collate_fn(batch):
    return {
        'x': torch.stack([b['x'] for b in batch]),
        'id': [b['id'] for b in batch],
        'AY': torch.stack([b['AY'] for b in batch]), 
        'LastAntVY': torch.tensor([b['LastAntVY'] for b in batch]),
        'LastAntNY': torch.tensor([b['LastAntNY'] for b in batch]),
        'LastAntY': torch.tensor([b['LastAntY'] for b in batch])
    }

def main():
    parser = argparse.ArgumentParser(description="AGA Model Train script (No catorch dependency)")
    parser.add_argument('--in_dim', type=int, default=1024, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=3806, help='Output dimension (number of classes)')
    parser.add_argument('--order', type=int, default=30, help='Sequence length or order limit')
    parser.add_argument('--recurrent_query', type=str, default='ma', help='Recurrent query type')
    parser.add_argument('--ma_ratio', type=float, default=0.8, help='Moving average ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    parser.add_argument('--data_source', type=str, default='datasets/ek100_swinvit', help='Path to the directory containing dataset files')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training/validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr_base', type=float, default=2e-4, help='Base learning rate')
    parser.add_argument('--wd_base', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Where to save the best model')
    args = parser.parse_args()

    # 1. Initialize the model
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
    model = model.to(args.device)

    # 2. Setup DataLoaders
    print(f"Loading EK100 dataset from {args.data_source}...")
    train_dataset = EK100Dataset(args.data_source, train=True, sequence_length=args.order, random_drift_sampling=True)
    val_dataset = EK100Dataset(args.data_source, train=False, sequence_length=args.order, random_drift_sampling=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. Setup Loss, Optimizer, Scaler, Scheduler
    aw_path = 'action_weight.npy'
    if os.path.exists(aw_path):
        aw = torch.from_numpy(np.load(aw_path)).float()
        aw = aw / aw.mean()
        aw = aw.to(args.device)
    else:
        print("[Warning] action_weight.npy not found! Uniform weights will be used.")
        aw = None

    criterion = nn.CrossEntropyLoss(weight=aw, ignore_index=-1, reduction='none')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_base, weight_decay=args.wd_base)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(growth_interval=50) if args.device == 'cuda' else None

    # 4. Training Loop
    best_mt5r = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch in pbar:
            x = batch['x'].to(args.device)
            # AY has shape (B, SeqLen + 1). We need to align it with sequence predictions.
            # In Cat.Functional.Fit ASeqLoss, `expand_y=False` means CE applies on matched sequence dimensions.
            ay = batch['AY'].to(args.device)
            last_ant_y = batch['LastAntY'].to(args.device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(x) # Shape: (B, SeqLen, OutDim)
                    
                    B, S, C = preds.shape
                    preds_flat = preds.view(-1, C)
                    ay_shifted = ay[:, 1:S+1].contiguous().view(-1)
                    
                    # catorch uses reduction='none' then .mean() directly, dragging ignored (-1) zeros into the denominator
                    raw_aseq_loss = criterion(preds_flat, ay_shifted)
                    aseq_loss = raw_aseq_loss.mean()
                    
                    raw_aloss = criterion(preds[:, -1, :], last_ant_y)
                    aloss = raw_aloss.mean()
                    
                    loss = aseq_loss + aloss
            else:
                preds = model(x)
                B, S, C = preds.shape
                preds_flat = preds.view(-1, C)
                ay_shifted = ay[:, 1:S+1].contiguous().view(-1)
                
                raw_aseq_loss = criterion(preds_flat, ay_shifted)
                aseq_loss = raw_aseq_loss.mean()
                
                raw_aloss = criterion(preds[:, -1, :], last_ant_y)
                aloss = raw_aloss.mean()
                loss = aseq_loss + aloss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        print(f"Epoch {epoch} Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation phase
        print(f"Evaluating epoch {epoch}...")
        scores = validate_ek100(model, val_loader, data_source=args.data_source, device=args.device)
        
        # MT5R Extraction: Overall Mean Top-5 Recall Action
        try:
            mt5r = scores.loc[('Overall Mean Top-5 Recall', 'Action')].values[0]
            print(f"Validation MT5R: {mt5r:.2f}%")
            
            if mt5r > best_mt5r:
                best_mt5r = mt5r
                print(f"New best MT5R! Saving model to {args.save_path}...")
                torch.save(model.state_dict(), args.save_path)
        except Exception as e:
            print(f"Could not extract MT5R specifically. Saving latest checkpoint... ({e})")
            torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()