USE_CASE_KEYWORDS = {
    "coding": [
        "code", "coding", "programming", "developer", "software",
        "debug", "debugging", "python", "javascript", "java",
        "script", "algorithm", "function", "class", "code generation",
        "code review", "refactor", "git", "repository"
    ],
    "math": [
        "math", "mathematics", "maths", "calculus", "algebra",
        "equation", "numerical", "calculate", "physics", "statistics",
        "probability", "linear algebra", "matrix", "vector",
        "derivative", "integral", "optimization", "theorem"
    ],
    "reasoning": [
        "reason", "reasoning", "logic", "logical", "think", "thinking",
        "analyze", "analysis", "solve", "problem solving", "puzzle",
        "deduction", "inference", "critical thinking", "decision",
        "strategy", "planning", "goal"
    ],
    "general": []
}

USE_CASE_BENCHMARK_WEIGHTS = {
    "coding": {
        "coding": 0.50,
        "math": 0.20,
        "reasoning": 0.20,
        "intelligence_index": 0.10
    },
    "math": {
        "coding": 0.20,
        "math": 0.50,
        "reasoning": 0.20,
        "intelligence_index": 0.10
    },
    "reasoning": {
        "coding": 0.20,
        "math": 0.20,
        "reasoning": 0.50,
        "intelligence_index": 0.10
    },
    "general": {
        "coding": 0.25,
        "math": 0.25,
        "reasoning": 0.25,
        "intelligence_index": 0.25
    }
}

SEMANTIC_WEIGHT = 0.30
BENCHMARK_WEIGHT = 0.50
HARDWARE_WEIGHT = 0.20

TOP_K_RECOMMENDATIONS = 5

DEFAULT_USE_CASE = "general"