"""Genetics-related API endpoints: allele frequencies, epigenetic prevalence, correlations."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from seldon.api.serializers import _int
from seldon.core.genetics import GENE_LOCI, _allele_expression

router = APIRouter()


@router.get("/{session_id}/allele-frequencies")
def get_allele_frequencies(session_id: str, request: Request) -> dict[str, Any]:
    """Return per-locus allele counts from the living population."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    living = [a for a in session.engine.population if a.is_alive and a.genome]
    if not living:
        return {"enabled": bool(living or any(a.genome for a in session.all_agents.values())), "loci": {}}

    loci_data: dict[str, dict[str, Any]] = {}
    for locus_name in GENE_LOCI:
        dominant_count = 0
        recessive_count = 0
        for agent in living:
            alleles = agent.genome.get(locus_name)
            if alleles:
                a1, a2 = alleles if isinstance(alleles, (tuple, list)) else (alleles, alleles)
                dominant_count += (1 if a1 == "A" else 0) + (1 if a2 == "A" else 0)
                recessive_count += (1 if a1 == "a" else 0) + (1 if a2 == "a" else 0)
        total = dominant_count + recessive_count
        loci_data[locus_name] = {
            "trait": GENE_LOCI[locus_name],
            "dominant_count": dominant_count,
            "recessive_count": recessive_count,
            "dominant_frequency": round(dominant_count / total, 4) if total > 0 else 0.0,
            "recessive_frequency": round(recessive_count / total, 4) if total > 0 else 0.0,
        }

    return {"enabled": True, "loci": loci_data}


@router.get("/{session_id}/epigenetic-prevalence")
def get_epigenetic_prevalence(session_id: str, request: Request) -> dict[str, Any]:
    """Return marker activation rates across the living population."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    living = [a for a in session.engine.population if a.is_alive and a.epigenetic_state]
    if not living:
        return {"enabled": bool(any(a.epigenetic_state for a in session.all_agents.values())), "markers": {}}

    marker_counts: dict[str, int] = {}
    for agent in living:
        for marker, active in agent.epigenetic_state.items():
            if marker not in marker_counts:
                marker_counts[marker] = 0
            if active:
                marker_counts[marker] += 1

    total = len(living)
    markers = {
        name: {
            "active_count": count,
            "total": total,
            "prevalence": round(count / total, 4),
        }
        for name, count in marker_counts.items()
    }

    return {"enabled": True, "markers": markers}


@router.get("/{session_id}/trait-gene-correlation")
def get_trait_gene_correlation(session_id: str, request: Request) -> dict[str, Any]:
    """Compute Pearson correlations between gene expression and trait values."""
    mgr = request.app.state.session_manager
    try:
        session = mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    ts = session.config.trait_system
    living = [a for a in session.engine.population if a.is_alive and a.genome]
    if len(living) < 3:
        return {"enabled": len(living) > 0, "correlations": {}}

    correlations: dict[str, float] = {}
    for locus_name, trait_name in GENE_LOCI.items():
        try:
            trait_idx = ts.trait_index(trait_name)
        except (KeyError, ValueError):
            continue

        expressions = []
        trait_values = []
        for agent in living:
            alleles = agent.genome.get(locus_name)
            if alleles:
                a1, a2 = alleles if isinstance(alleles, (tuple, list)) else (alleles, alleles)
                expressions.append(_allele_expression(a1, a2))
                trait_values.append(float(agent.traits[trait_idx]))

        if len(expressions) >= 3:
            expr_arr = np.array(expressions)
            trait_arr = np.array(trait_values)
            if np.std(expr_arr) > 0 and np.std(trait_arr) > 0:
                corr = float(np.corrcoef(expr_arr, trait_arr)[0, 1])
                correlations[locus_name] = round(corr, 4)
            else:
                correlations[locus_name] = 0.0

    return {"enabled": True, "correlations": correlations}
