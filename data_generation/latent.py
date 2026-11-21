import numpy as np


def seeded_profile(company_id: str, fiscal_year: int, seed: int = 42):
    """Return deterministic ESG latent, quality latent, beta and volatility."""
    offset = int(company_id[1:])
    rng = np.random.default_rng(seed + offset * 100 + fiscal_year)
    return {
        "esg_latent": float(rng.normal(0.0, 1.0)),
        "quality_latent": float(rng.normal(0.0, 1.0)),
        "beta": float(rng.uniform(0.7, 1.3)),
        "idiosyncratic_volatility": float(rng.uniform(0.18, 0.35)),
        "base_drift": float(rng.uniform(0.05, 0.12)),
    }
