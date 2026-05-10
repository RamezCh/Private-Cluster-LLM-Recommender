"""Fuzzy model name matching using thefuzz library."""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from thefuzz import fuzz
from loguru import logger


@dataclass
class ModelMapping:
    """Result of fuzzy matching between model names across sources."""
    canonical_name: str
    sources: Dict[str, str] = field(default_factory=dict)
    match_score: int = 100
    is_confident: bool = True


class FuzzyModelMatcher:
    """Aligns model names across different data sources using fuzzy matching."""

    def __init__(self, score_threshold: int = 85):
        self.threshold = score_threshold
        self.mappings: Dict[str, ModelMapping] = {}
        self.unmatched: List[Dict] = []

    def _normalize(self, name: str) -> str:
        """Standardize model names for comparison."""
        name = name.lower().strip()
        name = re.sub(r'["\'\-_]', ' ', name)
        name = re.sub(r'instruct|chat|preview|beta|release', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _similarity(self, name1: str, name2: str) -> int:
        """Calculate fuzzy similarity score."""
        n1, n2 = self._normalize(name1), self._normalize(name2)
        
        if n1 == n2:
            return 100
        
        scores = [
            fuzz.token_set_ratio(n1, n2) * 0.5,
            fuzz.sequence_ratio(n1, n2) * 0.3,
            fuzz.partial_ratio(n1, n2) * 0.2,
        ]
        
        return min(100, int(sum(scores)))

    def _select_canonical(self, names: List[str]) -> str:
        """Select the best canonical name from a group."""
        if len(names) == 1:
            return names[0]
        
        patterns = [r'instruct', r'chat', r'\d+\.\d+', r'[A-Z][a-z]']
        best, best_score = names[0], 0
        
        for name in names:
            score = 50 + sum(30 if re.search(p, name, re.I) else 0 for i, p in enumerate(patterns))
            if len(name) > len(best):
                score += 5
            if score > best_score:
                best_score = score
                best = name
        
        return best

    def build_mappings(
        self,
        source_a: List[str],
        source_b: List[str],
        source_c: Optional[List[str]] = None
    ) -> Dict[str, ModelMapping]:
        """Build a mapping table between model names from different sources."""
        logger.info(f"Building mappings: {len(source_a)} + {len(source_b)} + {len(source_c or [])}")
        
        all_names = list(set(source_a + source_b + (source_c or [])))
        groups: Dict[str, List[str]] = {}
        
        for name in all_names:
            norm = self._normalize(name)
            matched = False
            
            for base in groups:
                if self._similarity(norm, base) >= self.threshold:
                    groups[base].append(name)
                    matched = True
                    break
            
            if not matched:
                groups[norm] = [name]
        
        self.mappings = {}
        self.unmatched = []
        
        for base, names in groups.items():
            canonical = self._select_canonical(names)
            
            sources = {}
            for name in names:
                if name in source_a:
                    sources["artificial_analysis"] = name
                if name in source_b:
                    sources["open_evals"] = name
                if source_c and name in source_c:
                    sources["lmsys_arena"] = name
            
            score = self._similarity(names[0], names[-1]) if len(names) > 1 else 100
            
            self.mappings[canonical] = ModelMapping(
                canonical_name=canonical,
                sources=sources,
                match_score=score,
                is_confident=score >= self.threshold
            )
            
            if not self.mappings[canonical].is_confident and len(names) > 1:
                self.unmatched.append({"canonical": canonical, "variants": names, "score": score})
        
        logger.info(f"Created {len(self.mappings)} mappings, {len(self.unmatched)} low-confidence")
        return self.mappings

    def get_canonical(self, model_name: str) -> Optional[str]:
        """Get the canonical name for a given model name."""
        for canonical, mapping in self.mappings.items():
            if model_name in mapping.sources.values():
                return canonical
        
        norm = self._normalize(model_name)
        for canonical in self.mappings:
            if self._similarity(norm, self._normalize(canonical)) >= self.threshold:
                return canonical
        
        return None

    def get_source_variant(self, canonical_name: str, source: str) -> Optional[str]:
        """Get the specific model name variant for a given source."""
        return self.mappings.get(canonical_name, ModelMapping("")).sources.get(source)

    def get_unmatched_report(self) -> Dict:
        """Generate a report of potentially misaligned model names."""
        return {
            "total": len(self.unmatched),
            "items": self.unmatched,
            "suggestion": "Review manually or lower threshold"
        }