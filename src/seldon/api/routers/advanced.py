"""Advanced analytics endpoints: anomaly detection, lore overview, sensitivity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


# ---------------------------------------------------------------------------
# Anomaly detection helpers
# ---------------------------------------------------------------------------

_ANOMALY_METRICS = [
    "population_size",
    "births",
    "deaths",
    "breakthroughs",
    "trait_entropy",
    "total_contribution",
    "mean_contribution",
    "mean_suffering",
    "pairs_formed",
    "dissolutions",
]

_CATEGORY_MAP = {
    "breakthroughs": "breakthrough_spike",
    "deaths": "mortality_surge",
    "births": "birth_spike",
    "population_size": "population_shift",
    "mean_suffering": "suffering_surge",
    "mean_contribution": "contribution_shift",
    "total_contribution": "contribution_shift",
    "trait_entropy": "diversity_shift",
    "pairs_formed": "pairing_surge",
    "dissolutions": "dissolution_surge",
}


def _detect_anomalies(metrics_history) -> dict[str, Any]:
    """Compute z-score anomalies across all generations."""
    n = len(metrics_history)
    if n < 3:
        return {"anomalies": [], "generation_scores": [], "thresholds": {"warning": 1.5, "anomaly": 2.0, "critical": 3.0}}

    # Build per-generation max z-score tracker
    gen_max_z: list[float] = [0.0] * n
    anomalies: list[dict[str, Any]] = []

    for metric_name in _ANOMALY_METRICS:
        values = [float(getattr(m, metric_name, 0)) for m in metrics_history]
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-10:
            continue

        for i, val in enumerate(values):
            z = abs(val - mean) / std
            gen_max_z[i] = max(gen_max_z[i], z)

            if z >= 2.0:
                if z >= 3.0:
                    severity = "critical"
                elif z >= 2.5:
                    severity = "high"
                else:
                    severity = "medium"

                direction = "above" if val > mean else "below"
                category = _CATEGORY_MAP.get(metric_name, "anomaly")

                anomalies.append({
                    "generation": i,
                    "severity": severity,
                    "z_score": round(z, 2),
                    "metric": metric_name,
                    "value": round(val, 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "category": category,
                    "description": (
                        f"Generation {i}: {metric_name}={round(val, 2)} "
                        f"({round(z, 1)} std {direction} mean)"
                    ),
                })

    # Sort by severity descending (critical > high > medium), then z_score
    severity_rank = {"critical": 3, "high": 2, "medium": 1}
    anomalies.sort(key=lambda a: (severity_rank.get(a["severity"], 0), a["z_score"]), reverse=True)

    return {
        "anomalies": anomalies,
        "generation_scores": [round(s, 4) for s in gen_max_z],
        "thresholds": {"warning": 1.5, "anomaly": 2.0, "critical": 3.0},
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{session_id}/anomalies")
def get_anomalies(session_id: str, request: Request) -> dict[str, Any]:
    """Detect statistical anomalies across generations using z-scores."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return _detect_anomalies(session.collector.metrics_history)


@router.get("/{session_id}/lore/overview")
def get_lore_overview(session_id: str, request: Request) -> dict[str, Any]:
    """Lore metrics time series + current societal lore + memory type distribution."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    history = session.collector.metrics_history

    # Time series from metrics history
    time_series = {
        "total_memories": [m.total_memories for m in history],
        "societal_memories": [m.societal_memories for m in history],
        "myths_count": [m.myths_count for m in history],
        "generations": list(range(len(history))),
    }

    # Current societal lore from the lore engine
    current_societal_lore: list[dict[str, Any]] = []
    lore_engine = getattr(session.engine, "lore_engine", None)
    if lore_engine is not None:
        for mem in getattr(lore_engine, "societal_memories", []):
            if hasattr(mem, "to_dict"):
                current_societal_lore.append(mem.to_dict())
            elif isinstance(mem, dict):
                current_societal_lore.append(mem)

    # Memory type distribution from alive agents
    type_dist: dict[str, int] = {"personal": 0, "family": 0, "societal": 0, "myth": 0}
    for agent in session.engine.population:
        for mem in agent.personal_memories:
            mt = mem.get("memory_type", "personal")
            type_dist[mt] = type_dist.get(mt, 0) + 1
        for mem in agent.inherited_lore:
            mt = mem.get("memory_type", "family")
            type_dist[mt] = type_dist.get(mt, 0) + 1

    # Mean fidelity per generation (approximate from metrics)
    mean_fidelity: list[float] = []
    for m in history:
        # Estimate: myths are low fidelity, societal medium, personal high
        total = m.total_memories if m.total_memories > 0 else 1
        myth_frac = m.myths_count / total
        # Rough estimate: myths ~ 0.2 fidelity, rest ~ 0.8
        estimated_fidelity = 0.8 * (1 - myth_frac) + 0.2 * myth_frac
        mean_fidelity.append(round(estimated_fidelity, 4))
    time_series["mean_fidelity"] = mean_fidelity

    return {
        "time_series": time_series,
        "current_societal_lore": current_societal_lore,
        "memory_type_distribution": type_dist,
    }


@router.get("/{session_id}/lore/meme-prevalence")
def get_meme_prevalence(session_id: str, request: Request) -> dict[str, Any]:
    """Meme prevalence data from culture extension metrics."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    history = session.collector.metrics_history

    # Check if culture extension was ever active
    has_culture = any(
        "culture" in m.extension_metrics for m in history
    )

    if not has_culture:
        return {"enabled": False}

    # Build prevalence over time
    meme_names: set[str] = set()
    prevalence_series: dict[str, list[float]] = {}

    for m in history:
        culture_data = m.extension_metrics.get("culture", {})
        prev = culture_data.get("meme_prevalence", {})
        meme_names.update(prev.keys())

    for name in meme_names:
        prevalence_series[name] = []

    for m in history:
        culture_data = m.extension_metrics.get("culture", {})
        prev = culture_data.get("meme_prevalence", {})
        for name in meme_names:
            prevalence_series[name].append(round(prev.get(name, 0.0), 4))

    # Current dominant meme
    if history:
        last_culture = history[-1].extension_metrics.get("culture", {})
        current_dominant = last_culture.get("dominant_meme")
    else:
        current_dominant = None

    # Build meme info list
    memes: list[dict[str, Any]] = []
    culture_ext = session.engine.extensions.get("culture") if session.engine.extensions else None
    if culture_ext is not None:
        for meme_id, meme in getattr(culture_ext, "memes", {}).items():
            memes.append({
                "id": meme_id,
                "name": meme.name,
                "prevalence": round(meme.prevalence, 4),
                "effects": dict(meme.effects),
            })

    return {
        "enabled": True,
        "memes": memes,
        "prevalence_over_time": prevalence_series,
        "generations": list(range(len(history))),
        "current_dominant": current_dominant,
    }


@router.post("/{session_id}/sensitivity")
def compute_sensitivity(
    session_id: str,
    request: Request,
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare multiple sessions to identify which parameters most affect a target metric.

    Body: { "session_ids": [...], "target_metric": "mean_contribution" }
    """
    mgr = request.app.state.session_manager
    session_ids = body.get("session_ids", [])
    target_metric = body.get("target_metric", "mean_contribution")

    if len(session_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 sessions required for sensitivity analysis")

    # Gather data from each session
    sessions_data: list[dict[str, Any]] = []
    for sid in session_ids:
        try:
            session = mgr.get_session(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")

        history = session.collector.metrics_history
        if not history:
            continue

        # Compute summary stats for target metric
        values_map = {
            "population_size": history[-1].population_size,
            "mean_contribution": float(np.mean([m.mean_contribution for m in history if m.population_size > 0])),
            "mean_suffering": float(np.mean([m.mean_suffering for m in history if m.population_size > 0])),
            "breakthroughs": sum(m.breakthroughs for m in history),
            "total_births": sum(m.births for m in history),
            "total_deaths": sum(m.deaths for m in history),
            "trait_entropy": float(np.mean([m.trait_entropy for m in history])),
        }

        outcome = values_map.get(target_metric, 0.0)

        # Extract key config params
        cfg = session.config
        config_vals = {
            "initial_population": cfg.initial_population,
            "trait_drift_rate": cfg.trait_drift_rate,
            "random_seed": cfg.random_seed,
            "generations_to_run": cfg.generations_to_run,
            "lore_enabled": 1.0 if cfg.lore_enabled else 0.0,
            "cognitive_council_enabled": 1.0 if cfg.cognitive_council_enabled else 0.0,
        }
        # Add numeric config values from relationship/fertility/contribution configs
        for key, val in cfg.relationship_config.items():
            if isinstance(val, (int, float)):
                config_vals[f"relationship.{key}"] = float(val)
        for key, val in cfg.fertility_config.items():
            if isinstance(val, (int, float)):
                config_vals[f"fertility.{key}"] = float(val)

        sessions_data.append({
            "session_id": sid,
            "outcome": float(outcome),
            "config": config_vals,
        })

    if len(sessions_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sessions with metrics data")

    # Find parameters that vary between sessions
    all_param_names = set()
    for sd in sessions_data:
        all_param_names.update(sd["config"].keys())

    sensitivities: list[dict[str, Any]] = []
    tornado_data: list[dict[str, Any]] = []

    for param in sorted(all_param_names):
        param_values = [sd["config"].get(param) for sd in sessions_data]
        outcomes = [sd["outcome"] for sd in sessions_data]

        # Skip if param doesn't vary
        unique_vals = set(v for v in param_values if v is not None)
        if len(unique_vals) < 2:
            continue

        # Compute correlation if all numeric
        try:
            pv = np.array([float(v) for v in param_values if v is not None])
            ov = np.array([o for v, o in zip(param_values, outcomes) if v is not None])
            if len(pv) < 2 or pv.std() < 1e-10:
                continue
            correlation = float(np.corrcoef(pv, ov)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        except (TypeError, ValueError):
            continue

        # Tornado: low/high param value and corresponding outcomes
        paired = sorted(zip(param_values, outcomes), key=lambda x: x[0] if x[0] is not None else 0)
        low_val, low_outcome = paired[0]
        high_val, high_outcome = paired[-1]

        impact = abs(high_outcome - low_outcome)

        sensitivities.append({
            "parameter": param,
            "correlation": round(correlation, 4),
            "impact": round(impact, 4),
            "min_value": low_val,
            "max_value": high_val,
            "min_outcome": round(low_outcome, 4),
            "max_outcome": round(high_outcome, 4),
        })

        tornado_data.append({
            "parameter": param,
            "low_outcome": round(low_outcome, 4),
            "high_outcome": round(high_outcome, 4),
            "low_value": low_val,
            "high_value": high_val,
            "swing": round(high_outcome - low_outcome, 4),
        })

    # Sort sensitivities by impact descending
    sensitivities.sort(key=lambda s: s["impact"], reverse=True)
    tornado_data.sort(key=lambda t: abs(t["swing"]), reverse=True)

    return {
        "target_metric": target_metric,
        "session_count": len(sessions_data),
        "sensitivities": sensitivities,
        "tornado_data": tornado_data,
    }
