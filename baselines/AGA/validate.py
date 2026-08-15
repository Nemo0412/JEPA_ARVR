import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
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
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[Warning] Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys: {unexpected_keys}")
        
    print("Weights loaded successfully.")
    return model

# --- EK100 Validation Metrics (Extracted from catorch) ---

def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max()+1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi

def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xx = x
    x = x.reshape((-1, x.shape[-1]))
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    res = e_x / e_x.sum(axis=1).reshape(-1, 1)
    return res.reshape(xx.shape)

def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    if len(scores) == 0:
        return [0.0 for k in ks]
    rankings = scores.argsort()[:, ::-1]
    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]

def topk_accuracy_multiple_timesteps(preds, labels, ks=(1, 5)):
    accs = np.array(list(
        zip(*[topk_accuracy(preds[:, t, :], labels, ks) for t in range(preds.shape[1])])))
    return accs

def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    if len(classes) == 0:
        return 0.0
    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)

def topk_recall_multiple_timesteps(preds, labels, k=5, classes=None):
    accs = np.array([topk_recall(preds[:, t, :], labels, k, classes)
                     for t in range(preds.shape[1])])
    return accs.reshape(1, -1)

def get_validation_ids(data_source):
    unseen_participants_ids = pd.read_csv(join(data_source, 'validation_unseen_participants_ids.csv'), names=['id']).squeeze()
    tail_verbs_ids = pd.read_csv(join(data_source, 'validation_tail_verbs_ids.csv'), names=['id']).squeeze()
    tail_nouns_ids = pd.read_csv(join(data_source, 'validation_tail_nouns_ids.csv'), names=['id']).squeeze()
    tail_actions_ids = pd.read_csv(join(data_source, 'validation_tail_actions_ids.csv'), names=['id']).squeeze()
    return unseen_participants_ids, tail_verbs_ids, tail_nouns_ids, tail_actions_ids

def validate_ek100(model, loader, data_source, device='cuda'):
    """
    Standalone validation function for EK100 dataset.
    Args:
        model: The PyTorch model (AGA) to evaluate.
        loader: The PyTorch DataLoader that yields batches of data.
                The batches should be dictionaries with keys like 'id', 'LastAntVY', 'LastAntNY', 'LastAntY', etc.,
                and the model input should be passed either implicitly through the dictionary or appropriately.
        data_source: Path to the directory containing annotation CSV/NPY files.
        device: Device to run validation on.
    """
    model.eval()
    model.to(device)
    predictions = []
    labels = []
    ids = []
    
    print("Running inference on validation set...")
    with torch.inference_mode():
        for i in tqdm(loader):
            # Move items to device
            for key in i:
                if isinstance(i[key], list):
                    continue
                if isinstance(i[key], torch.Tensor):
                    i[key] = i[key].to(device)

            out = model(i.get('x', i.get('features', list(i.values())[-1] if 'id' not in list(i.keys())[-1] else list(i.values())[0])))

            if isinstance(i['id'], (list, tuple)):
                ids.extend(i['id'])
            elif isinstance(i['id'], torch.Tensor):
                ids.extend(i['id'].tolist())
            else:
                ids.append(i['id'])

            preds = out[:, -1, :].cpu().numpy()
            predictions.append(preds)
            labels.append(torch.stack([i['LastAntVY'], i['LastAntNY'], i['LastAntY']], -1).cpu().numpy())
    
    action_scores = np.concatenate(predictions)
    labels = np.concatenate(labels)
    ids = np.array(ids)

    print("Computing metrics...")
    actions = pd.read_csv(join(data_source, 'actions.csv'), index_col='id')

    vi = get_marginal_indexes(actions, 'verb')
    ni = get_marginal_indexes(actions, 'noun')

    action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
    verb_scores = marginalize(action_probs, vi)
    noun_scores = marginalize(action_probs, ni)
    
    overall_verb_recalls = topk_recall_multiple_timesteps(verb_scores[:, None, :], labels[:, 0], k=5)
    overall_noun_recalls = topk_recall_multiple_timesteps(noun_scores[:, None, :], labels[:, 1], k=5)
    overall_action_recalls = topk_recall_multiple_timesteps(action_scores[:, None, :], labels[:, 2], k=5)

    try:
        unseen, tail_verbs, tail_nouns, tail_actions = get_validation_ids(data_source)

        unseen_bool_idx = pd.Series(ids).isin(unseen).values
        tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
        tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
        tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

        tail_verb_recalls = topk_recall_multiple_timesteps(verb_scores[tail_verbs_bool_idx, None, :], labels[:, 0][tail_verbs_bool_idx], k=5)
        tail_noun_recalls = topk_recall_multiple_timesteps(noun_scores[tail_nouns_bool_idx, None, :], labels[:, 1][tail_nouns_bool_idx], k=5)
        tail_action_recalls = topk_recall_multiple_timesteps(action_scores[tail_actions_bool_idx, None, :], labels[:, 2][tail_actions_bool_idx], k=5)

        unseen_verb_recalls = topk_recall_multiple_timesteps(verb_scores[unseen_bool_idx, None, :], labels[:, 0][unseen_bool_idx], k=5)
        unseen_noun_recalls = topk_recall_multiple_timesteps(noun_scores[unseen_bool_idx, None, :], labels[:, 1][unseen_bool_idx], k=5)
        unseen_action_recalls = topk_recall_multiple_timesteps(action_scores[unseen_bool_idx, None, :], labels[:, 2][unseen_bool_idx], k=5)

        all_accuracies = np.concatenate(
            [overall_verb_recalls, overall_noun_recalls, overall_action_recalls, unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls, tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
        ) # 9 x 8

        indices = [
            ('Overall Mean Top-5 Recall', 'Verb'),
            ('Overall Mean Top-5 Recall', 'Noun'),
            ('Overall Mean Top-5 Recall', 'Action'),
            ('Unseen Mean Top-5 Recall', 'Verb'),
            ('Unseen Mean Top-5 Recall', 'Noun'),
            ('Unseen Mean Top-5 Recall', 'Action'),
            ('Tail Mean Top-5 Recall', 'Verb'),
            ('Tail Mean Top-5 Recall', 'Noun'),
            ('Tail Mean Top-5 Recall', 'Action'),
        ]
    except FileNotFoundError:
        print("[Warning] Some validation split files were not found. Returning overall accuracies only.")
        all_accuracies = np.concatenate(
            [overall_verb_recalls, overall_noun_recalls, overall_action_recalls]
        )
        indices = [
            ('Overall Mean Top-5 Recall', 'Verb'),
            ('Overall Mean Top-5 Recall', 'Noun'),
            ('Overall Mean Top-5 Recall', 'Action'),
        ]

    scores = pd.DataFrame(all_accuracies*100, index=pd.MultiIndex.from_tuples(indices))
    print("\nValidation Results:")
    print(scores)
    return scores

# --- EK100 Dataset Implementation ---
import lmdb

class EK100Dataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, sequence_length=30, time_step=1.0):
        self.root_folder = root_folder
        self.sequence_length = sequence_length
        self.time_step = time_step
        
        self.annotations = pd.read_csv(os.path.join(root_folder, 'validation.csv'), header=None, engine='c')
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
                
                frame_names = [f'{video}_frame_{int(x):010d}.jpg' for x in frames]
                self.samples.append((frame_names, name, verb, noun, action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_names, name, verb, noun, action = self.samples[idx]
        seq = []
        with self.env.begin(write=False) as txn:
            for f in frame_names:
                data = txn.get(f.encode('utf-8'))
                if data is None:
                    # fallback zero vector if feature missing
                    seq.append(np.zeros(1024, dtype=np.float32))
                else:
                    seq.append(np.frombuffer(data, 'float32'))
        
        return {
            'x': torch.FloatTensor(np.stack(seq)),
            'id': name,
            'LastAntVY': verb,
            'LastAntNY': noun,
            'LastAntY': action
        }

def main():
    parser = argparse.ArgumentParser(description="AGA Model Validate script (No catorch dependency)")
    parser.add_argument('--checkpoint', type=str, default='ek100_swinbb_aga_best.pt', help='Path to the model checkpoint')
    parser.add_argument('--in_dim', type=int, default=1024, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=2048, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=3806, help='Output dimension (number of classes)')
    parser.add_argument('--order', type=int, default=30, help='Sequence length or order limit')
    parser.add_argument('--recurrent_query', type=str, default='ma', help='Recurrent query type')
    parser.add_argument('--ma_ratio', type=float, default=0.8, help='Moving average ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    parser.add_argument('--data_source', type=str, default='datasets/ek100_swinvit', help='Path to the directory containing dataset CSV/NPY files')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for validation')
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
    
    # 2. Load weights if a checkpoint is provided and exists
    if os.path.exists(args.checkpoint):
        model = load_weights(model, args.checkpoint)
    else:
        print(f"[Warning] Checkpoint file {args.checkpoint} not found. Running with randomly initialized weights.")

    model = model.to(args.device)

    # 3. Setup real DataLoader
    print(f"Loading EK100 dataset from {args.data_source}...")
    dataset = EK100Dataset(args.data_source, sequence_length=args.order, time_step=1.0)
    
    def collate_fn(batch):
        return {
            'x': torch.stack([b['x'] for b in batch]),
            'id': [b['id'] for b in batch],
            'LastAntVY': torch.tensor([b['LastAntVY'] for b in batch]),
            'LastAntNY': torch.tensor([b['LastAntNY'] for b in batch]),
            'LastAntY': torch.tensor([b['LastAntY'] for b in batch])
        }

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # 4. Run validation
    scores = validate_ek100(model, loader, data_source=args.data_source, device=args.device)
    
if __name__ == '__main__':
    main()
