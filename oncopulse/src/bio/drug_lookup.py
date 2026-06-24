import requests
import certifi
from typing import List, Dict, Any
from src.config import DGIDB_API_URL
from src.io.loaders import load_cached_drug_interactions
from src.schemas import DrugInteraction
from src.utils import get_logger

logger = get_logger("drug_lookup")

def query_dgidb_api(gene_symbol: str) -> List[DrugInteraction]:
    """Query live DGIdb API for a given gene, with strict timeout and certificate validation."""
    interactions = []
    try:
        logger.info(f"Querying DGIdb API for {gene_symbol}...")
        params = {"genes": gene_symbol}
        response = requests.get(DGIDB_API_URL, params=params, verify=certifi.where(), timeout=5.0)
        
        if response.status_code == 200:
            data = response.json()
            matched_terms = data.get("matchedTerms", [])
            for term in matched_terms:
                for interaction in term.get("interactions", []):
                    drug_name = interaction.get("drugName")
                    int_type = interaction.get("interactionTypes", ["associated"])[0] if interaction.get("interactionTypes") else "associated"
                    sources = interaction.get("sources", [])
                    score = float(len(sources)) # simple score based on source count
                    
                    if drug_name:
                        interactions.append(DrugInteraction(
                            gene_symbol=gene_symbol,
                            drug_name=drug_name,
                            interaction_type=int_type,
                            sources=sources,
                            score=score
                        ))
            logger.info(f"Live API found {len(interactions)} drug interactions for {gene_symbol}.")
        else:
            logger.warning(f"DGIdb API returned status code {response.status_code} for {gene_symbol}.")
            
    except Exception as e:
        logger.warning(f"Failed to query live DGIdb API for {gene_symbol} ({e}). Falling back to cached snapshot.")
        
    return interactions

def get_drug_interactions(genes: List[str]) -> List[DrugInteraction]:
    """Get drug interactions for a list of genes. Fall back to local cache snapshot if API fails or returns no results."""
    all_interactions = []
    cache = load_cached_drug_interactions()
    
    for gene in genes:
        # Try live query first
        gene_interactions = query_dgidb_api(gene)
        
        # If live query fails or returns nothing, fall back to cached snap
        if not gene_interactions:
            cached_data = cache.get(gene, [])
            for item in cached_data:
                gene_interactions.append(DrugInteraction(
                    gene_symbol=item["gene_symbol"],
                    drug_name=item["drug_name"],
                    interaction_type=item.get("interaction_type", "associated"),
                    sources=item.get("sources", ["DGIdb (Cached)"]),
                    score=item.get("score", 1.0)
                ))
            if cached_data:
                logger.info(f"Loaded {len(cached_data)} drug interactions from cache for {gene}.")
                
        all_interactions.extend(gene_interactions)
        
    return all_interactions

def rank_drug_interactions(
    interactions: List[DrugInteraction], 
    top_drivers: List[Any]
) -> List[DrugInteraction]:
    """Rank drug recommendations by gene contribution and interaction type."""
    # Build a lookup for gene contribution scores and directions
    driver_info = {item.gene_symbol: item for item in top_drivers}
    
    ranked_list = []
    for item in interactions:
        gene = item.gene_symbol
        info = driver_info.get(gene)
        
        if not info:
            continue
            
        # Calculate a ranking priority score
        # Priority points:
        # 1. High absolute contribution (driver strength)
        priority = abs(info.contribution) * 10
        
        # 2. Gene is an unfavorable driver (overexpressed and bad) and the drug inhibits it
        # Or gene is favorable (underexpressed and good) and the drug activates it
        is_unfavorable_high = (info.direction == "unfavorable" and info.contribution > 0)
        is_inhibitory = any(term in str(item.interaction_type).lower() for term in ["inhibitor", "antagonist", "blocker", "suppressor"])
        
        if is_unfavorable_high and is_inhibitory:
            priority += 5.0  # boost for matching inhibition profile
            
        # 3. Add base interaction score (e.g. number of evidence sources)
        priority += item.score * 0.5
        
        # Store temporary rank priority score
        item.score = priority
        ranked_list.append(item)
        
    # Sort by priority score descending
    ranked_list.sort(key=lambda x: x.score, reverse=True)
    return ranked_list
