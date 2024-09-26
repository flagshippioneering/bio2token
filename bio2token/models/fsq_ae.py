import torch.nn as nn
import torch
from typing import Optional, Dict, Any, List
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from bio2token.models.fsq import FSQ

from bio2token.utils.utils import Config


class Kabsch:
    def _centroid_adjust(self, X, mask=None):
        if mask is None:
            X_ctd = torch.mean(X, dim=1).unsqueeze(1)
            X_adj = X - X_ctd
        else:
            X_ctd = (torch.sum(X * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=1, keepdim=True)).unsqueeze(1)
            X_adj = (X - X_ctd) * mask.unsqueeze(-1)
        return X_ctd, X_adj

    def compute_kabsch(self, P, Q, mask=None):
        """
        P and Q are [B,N,3] tensors.
        """

        B, N, _ = P.shape

        P_ctd, P_adj = self._centroid_adjust(P, mask)
        Q_ctd, Q_adj = self._centroid_adjust(Q, mask)

        h = torch.bmm(P_adj.permute(0, 2, 1), Q_adj)

        u, singular_values, vt = torch.linalg.svd(h)

        if (singular_values < 0).any():
            raise ValueError("Singular values are negative")

        v = vt.permute(0, 2, 1)
        d = torch.det(v @ u.permute(0, 2, 1))
        e = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).expand(B, 3, 3).to(P.device)
        e = e.clone()
        e[:, 2, 2] = d

        rot = torch.matmul(torch.matmul(v, e), u.permute(0, 2, 1))
        tran = Q_ctd - torch.matmul(rot, P_ctd.transpose(1, 2)).transpose(1, 2)

        return rot, tran

    def apply_kabsch(self, coords, rot, tran):
        coords = torch.matmul(rot, coords.permute(0, 2, 1)).permute(0, 2, 1)
        return coords

    def kabsch(self, P, Q, mask=None):
        rot, tran = self.compute_kabsch(P, Q, mask)
        P_new = self.apply_kabsch(P, rot, tran)
        return P_new, rot, tran


def compute_rmsd(P, Q, mask=None):
    # computing RMSD
    squared_diff = torch.sum((P - Q) ** 2, dim=-1)
    if mask is not None:
        rmsd = torch.sqrt(torch.sum(squared_diff * mask, dim=1) / mask.sum(dim=1))
    else:
        rmsd = torch.sqrt(torch.mean(squared_diff, dim=1))
    return rmsd


def compute_tm_score(P, Q, mask=None, seq_type="protein"):
    d = torch.sum((P - Q) ** 2, dim=-1)
    if mask is not None:
        N = torch.sum(mask, axis=-1)
        if seq_type == "rna":
            d0 = 1.24 * torch.pow(N - 15, 1 / 3) - 1.8
        elif seq_type == "protein":
            d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}. Should be rna or protein")
        d0 = torch.clamp(d0, min=0.5) ** 2
        return torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))) * mask, dim=-1) / N
    else:
        N = P.shape[1]
        if seq_type == "rna":
            d0 = 1.24 * torch.pow(N - 15, 1 / 3) - 1.8
        elif seq_type == "protein":
            d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}. Should be rna or protein")
        d0 = torch.clamp(d0, min=0.5) ** 2
        return torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))), dim=-1) / N


class MambaConfig(Config):
    d_model: int = 128
    n_layer: int = 8
    vocab_size: int = 4097
    d_intermediate: int = 0
    ssm_cfg: Optional[Dict[str, Any]] = None
    attn_layer_idx: Optional[List] = None
    attn_cfg: Optional[Dict] = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = True
    pretrained_model: Optional[str] = None


class FSQ_AE_Config(Config):
    vocab_size: str = "small"
    node_hidden_dims_s: int = 128
    nan_track: bool = True
    pad_track: bool = True
    use_fsq: bool = True
    manual_masking: bool = True
    loss_type: str = "mse"
    compute_tm_score: bool = True
    freeze_encoder: bool = False
    mamba_encoder: MambaConfig
    mamba_decoder: MambaConfig


class FSQ_AE(nn.Module):
    config_cls = FSQ_AE_Config

    def __init__(self, config=config_cls):
        super(FSQ_AE, self).__init__()
        self.config = config
        ###### Define Encoder #######
        self.encoder = MambaLMHeadModel(
            self.config.mamba_encoder,
            initializer_cfg=None,
            device=None,
            dtype=None,
        )
        self.encoder.backbone.embedding = nn.Linear(
            3 + self.config.nan_track + self.config.pad_track,
            self.config.mamba_encoder.d_model,
        )
        self.encoder.lm_head = nn.Identity()

        # Freeze encoder for any fine-tuning of the decoder
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        ####### Define Decoder ######
        self.decoder = MambaLMHeadModel(
            self.config.mamba_decoder,
            initializer_cfg=None,
            device=None,
            dtype=None,
        )
        self.decoder.backbone.embedding = nn.Identity()
        self.decoder.lm_head = nn.Linear(self.config.mamba_decoder.d_model, 3, bias=False)

        self.kabsch = Kabsch()

        assert self.config.vocab_size in ["small", "large"]
        if self.config.vocab_size == "small":
            self.levels = [4, 4, 4, 4, 4, 4]
        elif self.config.vocab_size == "large":
            self.levels = [8, 8, 8, 5, 5, 5]
        self.fsq = FSQ(levels=self.levels, dim=self.config.node_hidden_dims_s)

    def forward(self, batch: dict, mode="inference"):
        # FSQ forward pass
        input_mask = batch["atom_mask"].view(-1, self.config.max_len)

        x_input = batch["coords_groundtruth"].squeeze(1).view(-1, self.config.max_len, 3).clone()
        x_input[~input_mask] = 0
        feature_track = []
        if self.config.nan_track:
            if "nan_mask" in batch.keys():
                nan_mask = batch["nan_mask"].view(-1, self.config.max_len).clone()
            else:
                nan_mask = x_input.new_zeros(x_input.shape, dtype=torch.bool, device=x_input.device)
            feature_track.append(nan_mask)
        if self.config.pad_track:
            if "pad_mask" in batch.keys():
                pad_mask = batch["pad_mask"].view(-1, self.config.max_len).clone()
            else:
                pad_mask = x_input.new_zeros(x_input.shape, dtype=torch.bool, device=x_input.device)
            feature_track.append(pad_mask)
        if len(feature_track) > 0:
            feature_track = torch.stack(feature_track, dim=-1)
            x_input = torch.cat([x_input, feature_track], dim=-1)
        h_nodes = self.encoder(x_input).logits
        h_nodes_cp = h_nodes.clone()
        if self.config.use_fsq:
            h_nodes_cp, batch["indices"] = self.fsq(h_nodes_cp.view(-1, self.config.max_len, self.config.node_hidden_dims_s))
        if self.config.manual_masking:
            h_nodes_cp[~input_mask] = 0

        coords = self.decoder(h_nodes_cp).logits
        if hasattr(batch, "indices"):
            return coords, batch["indices"]
        else:
            return coords, None

    def step(self, batch: dict, mode="inference"):

        coords, indices = self(batch, mode)
        B, N, _ = coords.shape
        coords_groundtruth = batch["coords_groundtruth"]

        if self.config.loss_type == "mse":
            pass
        else:
            raise ValueError("Invalid loss type: " + self.config.loss_type)

        # All structure loss
        atom_mask = batch["atom_mask"].reshape(B, N)
        coords_new_all, _, _ = self.kabsch.kabsch(coords, coords_groundtruth, atom_mask)
        rmsd_all = compute_rmsd(coords_new_all, coords_groundtruth, atom_mask)
        batch["rmsd_all"] = rmsd_all.mean()
        # Use total rmsd as loss
        batch["loss"] = rmsd_all.mean()

        # For monitoring, compute rmsd and TM on backbone and sidechain, if they exist
        if "bb_mask" in batch.keys():
            bb_mask = batch["bb_mask"].reshape(B, N)
            coords_new_bb, _, _ = self.kabsch.kabsch(coords, coords_groundtruth, bb_mask)
            rmsd_backbone = compute_rmsd(coords_new_bb, coords_groundtruth, bb_mask)
            batch["rmsd_bb"] = rmsd_backbone.mean()

        if "sc_mask" in batch.keys():
            sc_mask = batch["sc_mask"].reshape(B, N)
            coords_new_sc, _, _ = self.kabsch.kabsch(coords, coords_groundtruth, sc_mask)
            rmsd_sc = compute_rmsd(coords_new_sc, coords_groundtruth, sc_mask)
            batch["rmsd_sc"] = rmsd_sc.mean()

        if "c_ref_mask" in batch.keys():
            c_ref_mask = batch["c_ref_mask"].reshape(B, N)
        elif "bb_mask" in batch.keys():
            c_ref_mask = bb_mask
        else:
            c_ref_mask = atom_mask

        tm_all = compute_tm_score(
            coords_new_all,
            coords_groundtruth,
            c_ref_mask,
            seq_type=batch["seq_type"],
        )
        batch["tm_all"] = tm_all.mean()
        if "bb_mask" in batch.keys():
            tm_bb = compute_tm_score(
                coords_new_bb,
                coords_groundtruth,
                c_ref_mask,
                seq_type=batch["seq_type"],
            )
            batch["tm_bb"] = tm_bb.mean()

        if mode == "train" or mode == "validation":
            raise NotImplementedError("Not provided")

        elif mode == "inference":
            # Usefull mask for plotting
            return_dict = {}
            if "bb_mask" in batch.keys():
                return_dict["bb_mask"] = bb_mask  # B x L
                return_dict["rmsd_bb"] = rmsd_backbone  # B
                return_dict["tm_bb"] = tm_bb  # B
            if "sc_mask" in batch.keys():
                return_dict["sc_mask"] = sc_mask  # B x L
                return_dict["rmsd_sc"] = rmsd_sc  # B
            if "c_ref_mask" in batch.keys():
                return_dict["c_ref_mask"] = c_ref_mask  # B x L
            return_dict["atom_mask"] = atom_mask  # B x L
            # Atom types
            # Sequence length
            return_dict["length"] = batch["seq_length"]  # B
            # Predicted and ground-truth coordinates
            return_dict["coords_pred_kabsch_all"] = coords_new_all  # B x L x 3
            return_dict["coords_gt"] = coords_groundtruth  # B x L x 3
            # RMSD scores
            return_dict["rmsd"] = rmsd_all  # B
            # TM scores
            return_dict["tm"] = tm_all

            return_dict["indices"] = indices  # B
        return return_dict


if __name__ == "__main__":
    model = FSQ_AE()
    print("Model loaded successfully!")
