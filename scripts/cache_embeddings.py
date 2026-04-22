"""Encode & cache text embeddings offline for all PrimeKG entities.

Must be run BEFORE any fusion experiments. Outputs saved to data/embeddings/.

Encodes textual descriptions (from generate_descriptions.py) with each of
4 candidate text encoders, projects to 128 dims via one of 4 projection
methods, L2-normalizes, and saves as .pt files.

For pca / linear / nonlinear_ae, the projection is fit once on the
combined raw embedding matrix of all entities for a given encoder+tier,
then applied consistently to both drugs and phenotypes. This keeps drug
and phenotype embeddings in the same projected space for cosine scoring.

Text encoders:
  - PubMedBERT:   microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (768-dim)
  - BiomedBERT:   microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract (1024-dim)
  - BioLinkBERT:  michiyasunaga/BioLinkBERT-base (768-dim)
  - SPECTER2:     allenai/specter2_base + allenai/specter2 adapter (768-dim)

Projection methods:
  - pca:           PCA fitted on full embedding matrix, top-128 components
  - linear:        nn.Linear (no bias, Xavier init), trained as linear AE
  - nonlinear_ae:  Nonlinear autoencoder (hidden_dim -> 256 -> 128)
  - none:          No projection, L2-normalize at native dim

Output structure:
  data/embeddings/{encoder}/{tier}/{projection}/
    drug_embeddings.pt       # {"embeddings": Tensor, "node_indices": list[int]}
    phenotype_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.getLogger("adapters.model_mixin").setLevel(logging.ERROR)
logging.getLogger("adapters.loading").setLevel(logging.INFO)

# ── Encoder configs ──────────────────────────────────────────────────────
ENCODER_CONFIGS = {
    "pubmedbert": {
        "hf_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "output_dim": 768,
        "pooling": "mean",
    },
    "biomedbert": {
        "hf_model": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "output_dim": 1024,
        "pooling": "mean",
    },
    "biolinkbert": {
        "hf_model": "michiyasunaga/BioLinkBERT-base",
        "output_dim": 768,
        "pooling": "mean",
    },
    "specter2": {
        "hf_model": "allenai/specter2_base",
        "adapter_model": "allenai/specter2",
        "output_dim": 768,
        "pooling": "cls",
    },
}


# ── Load descriptions ────────────────────────────────────────────────────
def load_descriptions(desc_path: str | Path) -> tuple[list[int], list[str]]:
    """Load description JSON and return aligned node indices and text strings.

    Args:
        desc_path: Path to JSON file with format {node_index: {"name": str, "text": str}}.

    Returns:
        (node_indices, texts) — aligned lists.
    """
    with open(desc_path) as f:
        data = json.load(f)

    node_indices = []
    texts = []
    for idx_str, entry in sorted(data.items(), key=lambda x: int(x[0])):
        node_indices.append(int(idx_str))
        texts.append(entry["text"])

    return node_indices, texts


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducible projection training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Text encoding ────────────────────────────────────────────────────────
def encode_texts(
    texts: list[str],
    model_name: str,
    pooling: str = "mean",
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cuda",
    adapter_name: str | None = None,
) -> torch.Tensor:
    """Encode texts with a HuggingFace transformer.

    Args:
        texts: List of text strings to encode.
        model_name: HuggingFace model identifier.
        pooling: Pooling strategy ("mean" or "cls").
        batch_size: Encoding batch size.
        max_length: Max token length (256 to avoid truncating GPT-4 descriptions).
        device: Device for encoding.
        adapter_name: Optional adapter identifier to load on top of the base model.

    Returns:
        Tensor of shape (N, hidden_dim).
    """
    from transformers import AutoModel, AutoTokenizer

    adapter_setup = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if adapter_name is None:
        model = AutoModel.from_pretrained(model_name).to(device)
    else:
        try:
            from adapters import AdapterSetup, AutoAdapterModel
        except ImportError as exc:
            raise ImportError(
                "SPECTER2 adapter support requires the `adapters` package. "
                "Install it with `pip install -U adapters`."
            ) from exc

        model = AutoAdapterModel.from_pretrained(model_name)
        loaded_adapter_name = model.load_adapter(
            adapter_name,
            source="hf",
            load_as="specter2",
            set_active=True,
        )
        model.set_active_adapters(loaded_adapter_name)
        model.active_adapters = loaded_adapter_name
        adapter_setup = AdapterSetup(loaded_adapter_name)
        model = model.to(device)
        if loaded_adapter_name not in str(model.active_adapters):
            raise RuntimeError(
                f"Adapter activation failed: expected '{loaded_adapter_name}', "
                f"got '{model.active_adapters}'"
            )
    model.eval()

    # Check truncation rate
    n_truncated = 0
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > max_length:
            n_truncated += 1
    if n_truncated > 0:
        logger.warning(
            f"Truncated {n_truncated}/{len(texts)} texts "
            f"({n_truncated / len(texts) * 100:.1f}%) at max_length={max_length}"
        )

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False if adapter_name is not None else None,
            ).to(device)

            if adapter_setup is not None:
                with adapter_setup:
                    outputs = model(**encoded)
            else:
                outputs = model(**encoded)
            if pooling == "cls":
                pooled = outputs.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                token_embeds = outputs.last_hidden_state  # (B, L, D)
                masked = token_embeds * attention_mask
                summed = masked.sum(dim=1)  # (B, D)
                counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
                pooled = summed / counts  # (B, D)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            all_embeddings.append(pooled.cpu())

            if (i // batch_size) % 50 == 0:
                logger.info(f"  Encoded {i + len(batch_texts)}/{len(texts)} texts")

    return torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)


# ── Projection methods ───────────────────────────────────────────────────
def project_pca(
    embeddings: torch.Tensor,
    target_dim: int = 128,
    save_dir: Path | None = None,
) -> torch.Tensor:
    """PCA projection to target_dim, then L2-normalize.

    Args:
        embeddings: (N, D) input embeddings.
        target_dim: Target dimensionality.
        save_dir: If provided, save PCA components for reproducibility.

    Returns:
        (N, target_dim) L2-normalized projected embeddings.
    """
    from sklearn.decomposition import PCA

    X = embeddings.numpy()
    pca = PCA(n_components=target_dim, random_state=42)
    X_proj = pca.fit_transform(X)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "components": torch.from_numpy(pca.components_),  # (target_dim, D)
                "mean": torch.from_numpy(pca.mean_),  # (D,)
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
            save_dir / "pca_components.pt",
        )
        total_var = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA: {target_dim} components capture {total_var:.3f} of variance")

    projected = torch.from_numpy(X_proj)
    return F.normalize(projected, dim=-1)


def project_linear(
    embeddings: torch.Tensor,
    target_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    save_dir: Path | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Learned linear projection via linear autoencoder.

    nn.Linear(D, target_dim, bias=False) with Xavier uniform init.
    Trained as encoder+decoder minimizing MSE reconstruction loss.

    Args:
        embeddings: (N, D) input embeddings.
        target_dim: Target dimensionality.
        epochs: Training epochs.
        lr: Learning rate.
        save_dir: If provided, save projection weights.
        seed: Random seed for init and data order.

    Returns:
        (N, target_dim) L2-normalized projected embeddings.
    """
    set_seed(seed)
    input_dim = embeddings.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = nn.Linear(input_dim, target_dim, bias=False).to(device)
    decoder = nn.Linear(target_dim, input_dim, bias=False).to(device)
    nn.init.xavier_uniform_(encoder.weight)
    nn.init.xavier_uniform_(decoder.weight)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )

    dataset = TensorDataset(embeddings.to(device))
    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=loader_gen)

    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            latent = encoder(batch)
            recon = decoder(latent)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / len(embeddings)
            logger.info(f"  Linear AE epoch {epoch + 1}/{epochs}, loss: {avg_loss:.6f}")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), save_dir / "linear_proj.pt")
        with open(save_dir / "linear_proj_meta.json", "w") as f:
            json.dump(
                {"seed": seed, "target_dim": target_dim, "fit_on": "combined_entities"},
                f,
                indent=2,
            )

    encoder.eval()
    with torch.no_grad():
        projected = encoder(embeddings.to(device)).cpu()

    return F.normalize(projected, dim=-1)


class NonlinearAE(nn.Module):
    """Nonlinear autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon


def project_nonlinear_ae(
    embeddings: torch.Tensor,
    target_dim: int = 128,
    hidden_dim: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    save_dir: Path | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Nonlinear autoencoder projection.

    Encoder: D -> 256 (ReLU) -> 128. Decoder: 128 -> 256 (ReLU) -> D.
    Trained with MSE reconstruction loss.

    Args:
        embeddings: (N, D) input embeddings.
        target_dim: Latent dimensionality.
        hidden_dim: Intermediate layer size.
        epochs: Training epochs.
        lr: Learning rate.
        save_dir: If provided, save encoder/decoder state dicts.
        seed: Random seed for init and data order.

    Returns:
        (N, target_dim) L2-normalized projected embeddings.
    """
    set_seed(seed)
    input_dim = embeddings.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae = NonlinearAE(input_dim, hidden_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    dataset = TensorDataset(embeddings.to(device))
    loader_gen = torch.Generator(device="cpu")
    loader_gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=loader_gen)

    ae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            latent, recon = ae(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / len(embeddings)
            logger.info(
                f"  Nonlinear AE epoch {epoch + 1}/{epochs}, loss: {avg_loss:.6f}"
            )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ae.encoder.state_dict(), save_dir / "ae_encoder.pt")
        torch.save(ae.decoder.state_dict(), save_dir / "ae_decoder.pt")
        with open(save_dir / "ae_meta.json", "w") as f:
            json.dump(
                {
                    "seed": seed,
                    "hidden_dim": hidden_dim,
                    "target_dim": target_dim,
                    "fit_on": "combined_entities",
                },
                f,
                indent=2,
            )

    ae.eval()
    with torch.no_grad():
        projected, _ = ae(embeddings.to(device))
        projected = projected.cpu()

    return F.normalize(projected, dim=-1)


def project_none(embeddings: torch.Tensor) -> torch.Tensor:
    """No projection — just L2-normalize at native dim.

    Args:
        embeddings: (N, D) input embeddings.

    Returns:
        (N, D) L2-normalized embeddings.
    """
    return F.normalize(embeddings, dim=-1)


def split_projected_embeddings(
    projected: torch.Tensor,
    counts: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Split a concatenated projected matrix back by entity type."""
    splits: dict[str, torch.Tensor] = {}
    start = 0
    for entity_type in ["drugs", "phenotypes"]:
        count = counts[entity_type]
        splits[entity_type] = projected[start : start + count]
        start += count
    return splits


def project_and_normalize(
    embeddings: torch.Tensor,
    target_dim: int = 128,
    method: str = "pca",
    save_dir: Path | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Dispatch to projection method, then L2-normalize.

    Args:
        embeddings: (N, D) raw encoder output.
        target_dim: Target dimensionality (ignored for 'none').
        method: One of 'pca', 'linear', 'nonlinear_ae', 'none'.
        save_dir: Directory to save projection artifacts.
        seed: Random seed for projection methods with learnable parameters.

    Returns:
        (N, dim) L2-normalized embeddings. dim = target_dim or D for 'none'.
    """
    if method == "pca":
        return project_pca(embeddings, target_dim, save_dir)
    elif method == "linear":
        return project_linear(embeddings, target_dim, save_dir=save_dir, seed=seed)
    elif method == "nonlinear_ae":
        return project_nonlinear_ae(embeddings, target_dim, save_dir=save_dir, seed=seed)
    elif method == "none":
        return project_none(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")


# ── Main ─────────────────────────────────────────────────────────────────
def main(
    desc_dir: str,
    output_dir: str,
    encoder_name: str,
    desc_tier: str = "tier2",
    projection: str = "pca",
    target_dim: int = 128,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "cuda",
    seed: int = 42,
) -> None:
    """Encode descriptions and cache embeddings for one encoder × tier × projection.

    Args:
        desc_dir: Directory containing description JSON files.
        output_dir: Base output directory for embeddings.
        encoder_name: One of 'pubmedbert', 'biomedbert', 'biolinkbert', 'specter2'.
        desc_tier: Description tier ('tier1', 'tier2', 'gpt4o').
        projection: Projection method ('pca', 'linear', 'nonlinear_ae', 'none').
        target_dim: Target embedding dimension.
        batch_size: Encoding batch size.
        max_length: Max token length for tokenizer.
        device: Device for encoding.
        seed: Random seed for reproducible projection fitting.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(seed)

    desc_dir = Path(desc_dir)
    output_dir = Path(output_dir)

    if encoder_name not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Choose from: {list(ENCODER_CONFIGS.keys())}"
        )

    config = ENCODER_CONFIGS[encoder_name]
    hf_model = config["hf_model"]
    adapter_model = config.get("adapter_model")
    pooling = config.get("pooling", "mean")

    # Output directory: {output_dir}/{encoder}/{tier}/{projection}/
    out_path = output_dir / encoder_name / desc_tier / projection
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Encoder: {encoder_name} ({hf_model})")
    if adapter_model is not None:
        logger.info(f"Adapter: {adapter_model}")
    logger.info(f"Pooling: {pooling}")
    logger.info(f"Tier: {desc_tier}, Projection: {projection}")
    logger.info(f"Seed: {seed}")

    raw_embeddings_by_entity: dict[str, torch.Tensor] = {}
    node_indices_by_entity: dict[str, list[int]] = {}
    counts: dict[str, int] = {}

    for entity_type in ["drugs", "phenotypes"]:
        desc_file = desc_dir / f"{entity_type}_{desc_tier}.json"
        if not desc_file.exists():
            raise FileNotFoundError(f"Description file not found: {desc_file}")

        logger.info(f"Loading {entity_type} descriptions from {desc_file}")
        node_indices, texts = load_descriptions(desc_file)
        logger.info(f"Loaded {len(texts)} {entity_type} descriptions")

        # Encode
        logger.info(f"Encoding {entity_type} with {encoder_name}...")
        raw_embeddings = encode_texts(
            texts,
            hf_model,
            pooling=pooling,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            adapter_name=adapter_model,
        )
        logger.info(f"Raw embeddings shape: {raw_embeddings.shape}")

        raw_embeddings_by_entity[entity_type] = raw_embeddings
        node_indices_by_entity[entity_type] = node_indices
        counts[entity_type] = len(node_indices)

    # Project in a shared space so drugs and phenotypes remain comparable.
    logger.info(f"Projecting with method={projection} on combined entity matrix...")
    combined_raw = torch.cat(
        [raw_embeddings_by_entity["drugs"], raw_embeddings_by_entity["phenotypes"]],
        dim=0,
    )
    logger.info(f"Combined raw embeddings shape: {combined_raw.shape}")

    proj_save_dir = out_path if projection != "none" else None
    combined_projected = project_and_normalize(
        combined_raw,
        target_dim=target_dim,
        method=projection,
        save_dir=proj_save_dir,
        seed=seed,
    )
    logger.info(f"Combined projected embeddings shape: {combined_projected.shape}")

    if proj_save_dir is not None:
        with open(proj_save_dir / "projection_fit_info.json", "w") as f:
            json.dump(
                {
                    "fit_on": "combined_entities",
                    "seed": seed,
                    "n_drugs": counts["drugs"],
                    "n_phenotypes": counts["phenotypes"],
                    "n_total": counts["drugs"] + counts["phenotypes"],
                    "raw_dim": int(combined_raw.shape[1]),
                    "projected_dim": int(combined_projected.shape[1]),
                },
                f,
                indent=2,
            )

    projected_by_entity = split_projected_embeddings(combined_projected, counts)

    for entity_type in ["drugs", "phenotypes"]:
        projected = projected_by_entity[entity_type]
        logger.info(f"{entity_type} projected embeddings shape: {projected.shape}")

        save_path = out_path / f"{entity_type.rstrip('s')}_embeddings.pt"
        torch.save(
            {
                "embeddings": projected,
                "node_indices": node_indices_by_entity[entity_type],
            },
            save_path,
        )
        logger.info(f"Saved: {save_path}")

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache text embeddings")
    parser.add_argument(
        "--desc-dir", type=str, default="data/descriptions",
        help="Directory containing description JSON files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/embeddings",
        help="Base output directory for cached embeddings",
    )
    parser.add_argument(
        "--encoder", type=str, required=True,
        choices=list(ENCODER_CONFIGS.keys()),
        help="Text encoder to use",
    )
    parser.add_argument(
        "--tier", type=str, default="tier2",
        choices=["tier1", "tier2", "gpt4o"],
        help="Description tier",
    )
    parser.add_argument(
        "--projection", type=str, default="pca",
        choices=["pca", "linear", "nonlinear_ae", "none"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--target-dim", type=int, default=128,
        help="Target embedding dimension (ignored for 'none')",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max token length for tokenizer",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for encoding",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible projection fitting",
    )
    args = parser.parse_args()
    main(
        desc_dir=args.desc_dir,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
        desc_tier=args.tier,
        projection=args.projection,
        target_dim=args.target_dim,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )
