from typing import Optional

from config.weights import USE_CASE_KEYWORDS, DEFAULT_USE_CASE


def detect_use_case(text: str) -> tuple[str, list[str]]:
    text_lower = text.lower()
    
    matched_use_cases = []
    
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        if use_case == "general":
            continue
        
        for keyword in keywords:
            if keyword in text_lower:
                if use_case not in matched_use_cases:
                    matched_use_cases.append(use_case)
                break
    
    if not matched_use_cases:
        return DEFAULT_USE_CASE, []
    
    if len(matched_use_cases) == 1:
        return matched_use_cases[0], matched_use_cases
    
    return matched_use_cases[0], matched_use_cases


def get_primary_use_case(text: str) -> str:
    use_case, _ = detect_use_case(text)
    return use_case