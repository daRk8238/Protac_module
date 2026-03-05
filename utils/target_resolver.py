"""
Universal Target Resolver
--------------------------
Resolves ANY of the following to a ChEMBL target + UniProt ID:
  - Gene name        : "EGFR", "BRD4", "KRAS"
  - Protein name     : "Epidermal growth factor receptor"
  - UniProt accession: "P00533", "Q09864"
  - ChEMBL ID        : "CHEMBL203"

Resolution order:
  1. Query UniProt REST API (broadest, covers all proteins)
  2. Map UniProt accession → ChEMBL target
  3. Fall back to ChEMBL direct search if needed

Usage:
    resolver = TargetResolver()
    result = resolver.resolve("EGFR")
    result = resolver.resolve("sonic hedgehog")
    result = resolver.resolve("Q09864")
"""

import logging
import requests
from typing import Optional
from chembl_webresource_client.new_client import new_client

logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FETCH_URL  = "https://rest.uniprot.org/uniprotkb"


class ResolvedTarget:
    """Holds all resolved identifiers for a protein target."""
    def __init__(self):
        self.query: str = ""
        self.uniprot_id: str = None
        self.gene_name: str = None
        self.protein_name: str = None
        self.organism: str = None
        self.chembl_id: str = None
        self.chembl_name: str = None
        self.sequence: str = None
        self.found: bool = False
        self.source: str = None       # 'uniprot', 'chembl_direct'
        self.alternatives: list = []  # other matches if ambiguous

    def __repr__(self):
        return (f"<ResolvedTarget {self.gene_name} | UniProt={self.uniprot_id} "
                f"| ChEMBL={self.chembl_id} | found={self.found}>")

    def to_dict(self):
        return {
            "query": self.query,
            "uniprot_id": self.uniprot_id,
            "gene_name": self.gene_name,
            "protein_name": self.protein_name,
            "organism": self.organism,
            "chembl_id": self.chembl_id,
            "chembl_name": self.chembl_name,
            "found": self.found,
            "source": self.source,
            "alternatives": self.alternatives,
        }


class TargetResolver:
    def __init__(self, organism_filter: str = "Homo sapiens"):
        """
        Args:
            organism_filter: Prefer results from this organism.
                             Set to None to include all organisms.
        """
        self.organism_filter = organism_filter
        self.chembl_target = new_client.target

    # ── Main Entry Point ───────────────────────────────────────────────────────

    def resolve(self, query: str, max_alternatives: int = 5) -> ResolvedTarget:
        """
        Resolve any protein identifier to UniProt + ChEMBL IDs.

        Args:
            query: Any protein identifier (name, gene, UniProt ID, ChEMBL ID)
            max_alternatives: How many alternative matches to return

        Returns:
            ResolvedTarget with all available identifiers
        """
        result = ResolvedTarget()
        result.query = query
        query = query.strip()

        logger.info(f"[Resolver] Resolving: '{query}'")

        # 1. If it looks like a UniProt accession (e.g. P00533, Q09864)
        if self._looks_like_uniprot(query):
            logger.info(f"[Resolver] Looks like UniProt accession")
            success = self._resolve_from_uniprot_accession(query, result)
            if success:
                self._map_to_chembl(result)
                return result

        # 2. If it looks like a ChEMBL ID
        if query.upper().startswith("CHEMBL"):
            logger.info(f"[Resolver] Looks like ChEMBL ID")
            success = self._resolve_from_chembl_id(query.upper(), result)
            if success:
                return result

        # 3. Search UniProt by gene name / protein name (broadest)
        logger.info(f"[Resolver] Searching UniProt by name/gene...")
        candidates = self._search_uniprot(query, max_results=10)

        if candidates:
            # Pick best candidate (human first, then reviewed entries)
            best = self._pick_best_candidate(candidates)
            self._populate_from_uniprot_entry(best, result)
            result.source = "uniprot"

            # Store alternatives
            result.alternatives = [
                {
                    "uniprot_id": c.get("primaryAccession"),
                    "gene_name": self._extract_gene(c),
                    "protein_name": self._extract_protein_name(c),
                    "organism": c.get("organism", {}).get("scientificName"),
                }
                for c in candidates[:max_alternatives]
                if c.get("primaryAccession") != result.uniprot_id
            ]

            # Map to ChEMBL
            self._map_to_chembl(result)
            return result

        # 4. Fall back to direct ChEMBL search
        logger.info(f"[Resolver] Falling back to ChEMBL direct search...")
        success = self._search_chembl_direct(query, result)
        if success:
            result.source = "chembl_direct"
            return result

        logger.warning(f"[Resolver] Could not resolve: '{query}'")
        result.found = False
        return result

    def search_suggestions(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Return a list of protein suggestions for autocomplete.
        Used by the API's search endpoint.
        """
        candidates = self._search_uniprot(query, max_results=max_results)
        suggestions = []
        for c in candidates:
            suggestions.append({
                "uniprot_id": c.get("primaryAccession"),
                "gene_name": self._extract_gene(c),
                "protein_name": self._extract_protein_name(c),
                "organism": c.get("organism", {}).get("scientificName"),
                "reviewed": c.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
            })
        return suggestions

    # ── UniProt API ────────────────────────────────────────────────────────────

    def _looks_like_uniprot(self, query: str) -> bool:
        """Check if query looks like a UniProt accession (e.g. P00533, Q09864)."""
        import re
        return bool(re.match(
            r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$',
            query.strip()
        ))

    def _resolve_from_uniprot_accession(self, accession: str, result: ResolvedTarget) -> bool:
        """Fetch a UniProt entry directly by accession."""
        try:
            url = f"{UNIPROT_FETCH_URL}/{accession}.json"
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                logger.warning(f"[Resolver] UniProt accession not found: {accession}")
                return False
            r.raise_for_status()
            entry = r.json()
            self._populate_from_uniprot_entry(entry, result)
            result.source = "uniprot"
            return True
        except Exception as e:
            logger.error(f"[Resolver] UniProt fetch failed for {accession}: {e}")
            return False

    def _search_uniprot(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Search UniProt by gene name or protein name.
        Prefers reviewed (Swiss-Prot) human entries.
        """
        try:
            # Build query: search gene name OR protein name
            search_query = f'(gene:{query} OR protein_name:{query})'
            if self.organism_filter:
                search_query += f' AND (organism_name:"{self.organism_filter}")'

            params = {
                "query": search_query,
                "format": "json",
                "size": max_results,
                "fields": "accession,gene_names,protein_name,organism_name,reviewed,sequence",
            }

            r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
            r.raise_for_status()
            results = r.json().get("results", [])

            # If no results with organism filter, try without
            if not results and self.organism_filter:
                logger.info(f"[Resolver] No human results, retrying without organism filter...")
                params["query"] = f'(gene:{query} OR protein_name:{query})'
                r = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
                r.raise_for_status()
                results = r.json().get("results", [])

            logger.info(f"[Resolver] UniProt search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[Resolver] UniProt search failed: {e}")
            return []

    def _pick_best_candidate(self, candidates: list[dict]) -> dict:
        """
        Pick the best UniProt entry from a list of candidates.
        Priority: reviewed (Swiss-Prot) > human > first result.
        """
        # Prefer reviewed human entries
        for c in candidates:
            is_reviewed = c.get("entryType", "").startswith("UniProtKB reviewed")
            organism = c.get("organism", {}).get("scientificName", "")
            if is_reviewed and "Homo sapiens" in organism:
                return c

        # Reviewed any organism
        for c in candidates:
            if c.get("entryType", "").startswith("UniProtKB reviewed"):
                return c

        return candidates[0]

    def _populate_from_uniprot_entry(self, entry: dict, result: ResolvedTarget):
        """Fill a ResolvedTarget from a UniProt JSON entry."""
        result.uniprot_id = entry.get("primaryAccession")
        result.gene_name = self._extract_gene(entry)
        result.protein_name = self._extract_protein_name(entry)
        result.organism = entry.get("organism", {}).get("scientificName")
        result.found = True

        # Extract sequence if available
        seq_data = entry.get("sequence", {})
        result.sequence = seq_data.get("value") if seq_data else None

    def _extract_gene(self, entry: dict) -> Optional[str]:
        genes = entry.get("genes", [])
        if genes:
            gene_name = genes[0].get("geneName", {})
            return gene_name.get("value") if isinstance(gene_name, dict) else None
        return None

    def _extract_protein_name(self, entry: dict) -> Optional[str]:
        try:
            return (entry["proteinDescription"]["recommendedName"]["fullName"]["value"])
        except (KeyError, TypeError):
            try:
                submitted = entry["proteinDescription"]["submissionNames"]
                return submitted[0]["fullName"]["value"] if submitted else None
            except (KeyError, TypeError, IndexError):
                return None

    # ── ChEMBL Mapping ─────────────────────────────────────────────────────────

    def _map_to_chembl(self, result: ResolvedTarget):
        """Map a UniProt accession to a ChEMBL target ID."""
        if not result.uniprot_id:
            return

        try:
            targets = list(self.chembl_target.filter(
                target_components__accession=result.uniprot_id,
                target_type="SINGLE PROTEIN"
            ))

            if targets:
                best = targets[0]
                result.chembl_id = best.get("target_chembl_id")
                result.chembl_name = best.get("pref_name")
                logger.info(f"[Resolver] Mapped to ChEMBL: {result.chembl_id} ({result.chembl_name})")
            else:
                logger.info(f"[Resolver] No ChEMBL entry for UniProt {result.uniprot_id} — structure/ZINC data still available")

        except Exception as e:
            logger.warning(f"[Resolver] ChEMBL mapping failed: {e}")

    def _resolve_from_chembl_id(self, chembl_id: str, result: ResolvedTarget) -> bool:
        """Resolve directly from a ChEMBL target ID."""
        try:
            targets = list(self.chembl_target.filter(target_chembl_id=chembl_id))
            if not targets:
                return False
            t = targets[0]
            result.chembl_id = chembl_id
            result.chembl_name = t.get("pref_name")
            result.organism = t.get("organism")
            result.found = True

            components = t.get("target_components", [])
            if components:
                result.uniprot_id = components[0].get("accession")
                if result.uniprot_id:
                    self._resolve_from_uniprot_accession(result.uniprot_id, result)

            return True
        except Exception as e:
            logger.error(f"[Resolver] ChEMBL ID lookup failed: {e}")
            return False

    def _search_chembl_direct(self, query: str, result: ResolvedTarget) -> bool:
        """Last resort: search ChEMBL by pref_name."""
        try:
            targets = list(self.chembl_target.filter(
                pref_name__icontains=query,
                target_type="SINGLE PROTEIN"
            ))
            if not targets:
                return False

            best = targets[0]
            result.chembl_id = best.get("target_chembl_id")
            result.chembl_name = best.get("pref_name")
            result.organism = best.get("organism")
            result.found = True

            components = best.get("target_components", [])
            if components:
                result.uniprot_id = components[0].get("accession")

            logger.info(f"[Resolver] ChEMBL direct match: {result.chembl_id}")
            return True
        except Exception as e:
            logger.error(f"[Resolver] ChEMBL direct search failed: {e}")
            return False
