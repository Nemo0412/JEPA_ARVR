import torch.nn as nn
import torch

from scalant.config import Config
from scalant.models.encoder import build_encoder
from scalant.models.decoder import QueryDecoder
from scalant.utils.ouput_target_structure import Prediction
from scalant.models.classification_head import ClassificationHead
from scalant.models.clam import CLAM


def uniform_sample_from_last(tensor, n: int):
    B, T, D = tensor.shape
    step = T // n

    indices = torch.tensor([T - 1 - i * step for i in range(n)][::-1], dtype=torch.long)
    indices.clamp_(min=0)
    return tensor[:, indices]


class QueryPredictor(nn.Module):
    def __init__(self, cfg: Config, num_classes: dict[str, int], dataset):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)

        self.decoder = QueryDecoder(cfg)
        self.long_mem_len = int(cfg.DATA.LONG_MEMORY_LENGTH // cfg.DATA.PAST_STEP_IN_SEC)

        self.drop = cfg.MODEL.DROPOUT
        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)

        # Linear layers for dimension conversion
        self.input_dim_converter = nn.Identity()
        if cfg.MODEL.INPUT_DIM != cfg.MODEL.D_MODEL:
            self.input_dim_converter = nn.Linear(cfg.MODEL.INPUT_DIM, cfg.MODEL.D_MODEL)

        # Classification head
        self.head = ClassificationHead(cfg, num_classes, dataset)

        self.clustering = None
        if cfg.CLUSTERING.ENABLE:
            self.clustering = CLAM(cfg)
            if cfg.CLUSTERING.USE_EMBEDDING:
                self.clustering_emb = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.D_MODEL))
                self.semantic_emb = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.D_MODEL))
                nn.init.normal_(self.clustering_emb, std=.02)
                nn.init.normal_(self.semantic_emb, std=.02)

    def forward(self, past, target=None) -> Prediction:
        # Encoder
        past = self.input_dim_converter(past)

        if self.drop > 0:
            past = self.dropout(past)

        past_semantic = self.encoder(past)

        # To conduct a clean ablation, we always use the last layer output for loss function
        past_semantic_cluster = past_semantic[self.cfg.MODEL.INTERMEDIATE_LAYER_IDX] if isinstance(past_semantic, (tuple, list)) else past_semantic
        past_semantic = past_semantic[-1] if isinstance(past_semantic, (tuple, list)) else past_semantic

        # Split memory into long-term mem and working mem
        long_mem, work_mem = past_semantic[:, :self.long_mem_len], past_semantic[:, self.long_mem_len:]

        if self.cfg.MODEL.SAMPLE_UNIFORM:
            work_len = work_mem.size(1)
            work_mem = uniform_sample_from_last(past_semantic, work_len)

        # Prepare clustering
        cluster_centers = None
        past_reconstructed = None
        if self.clustering is not None:
            if self.cfg.CLUSTERING.ON_FRAME_TOKENS:
                past_ori = past
                cluster_centers, num_points_per_cluster, cluster_labels = self.clustering(past_ori)
                if cluster_centers.shape[-1] != past_semantic.shape[-1]:
                    cluster_centers = self.input_dim_converter(cluster_centers)
            else:
                cluster_centers, num_points_per_cluster, cluster_labels = self.clustering(past_semantic_cluster)

        memory_pos = None
        if self.clustering is None:
            # If there is no clustering module, we use the usual memory positional embedding
            mem_for_decoder = work_mem
            if hasattr(self.encoder, "positional_encoding"):
                past_len = past.size(1)
                long_mem_len = long_mem.size(1)
                memory_pos = self.encoder.positional_encoding().expand(
                    past.shape[0], -1, -1)[:, long_mem_len:past_len]
        else:
            semantic_mem = work_mem if not self.cfg.CLUSTERING.CAT_WORK_LAST else work_mem[:, -1:]
            mem_for_decoder = torch.cat([cluster_centers, semantic_mem], dim=1)
            if hasattr(self, "clustering_emb"):
                cluster_emb = self.clustering_emb.expand(-1, cluster_centers.size(1), -1)
                semantic_emb = self.semantic_emb.expand(-1, semantic_mem.size(1), -1)
                memory_pos = torch.cat([cluster_emb, semantic_emb], dim=1)

        future_pred = self.decoder(mem_for_decoder, memory_pos=memory_pos)

        # Classification
        assert self.cfg.MODEL.CLS_WORK + self.cfg.MODEL.CLS_ALL + self.cfg.MODEL.CLS_LAST == 1
        head_mem = work_mem if self.cfg.MODEL.CLS_WORK else past_semantic if self.cfg.MODEL.CLS_ALL else work_mem[:, -1:]
        out = self.head(head_mem, future_pred)

        out.clusters = cluster_centers
        out.past_reconstructed = past_reconstructed
        return out
