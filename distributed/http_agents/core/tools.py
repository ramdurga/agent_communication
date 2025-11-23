"""
Shared tools for all agents.
Each agent type can use a subset of these tools.
"""
from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information on a topic."""
    knowledge = {
        "renewable energy": "Renewable energy comes from natural sources like solar, wind, hydro, and geothermal.",
        "solar": "Solar energy harnesses sunlight using photovoltaic cells or solar thermal systems.",
        "wind": "Wind energy uses turbines to convert kinetic energy from wind into electricity.",
        "ai": "Artificial Intelligence enables machines to learn, reason, and make decisions.",
        "machine learning": "Machine learning is a subset of AI that learns patterns from data.",
        "climate": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "healthcare": "Healthcare AI applications include diagnosis, drug discovery, and patient monitoring.",
        "quantum": "Quantum computing uses quantum mechanics principles for computation.",
    }
    results = []
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            results.append(f"[{key.upper()}]: {value}")
    return "\n".join(results) if results else f"No specific knowledge found for: {query}"


@tool
def extract_keywords(text: str) -> str:
    """Extract key topics and keywords from text."""
    common_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'to', 'of', 'in', 'for', 'on', 'with'
    }
    words = text.lower().split()
    keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in common_words and len(w) > 3]
    unique_keywords = list(dict.fromkeys(keywords))[:10]
    return f"Keywords: {', '.join(unique_keywords)}"


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and tone of the provided text."""
    positive_words = ['good', 'great', 'excellent', 'positive', 'benefit', 'advantage',
                     'improve', 'success', 'effective', 'efficient', 'innovative']
    negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'risk',
                     'challenge', 'difficult', 'concern', 'failure', 'limitation']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = min(95, 60 + (pos_count - neg_count) * 5)
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = min(95, 60 + (neg_count - pos_count) * 5)
    else:
        sentiment = "NEUTRAL"
        confidence = 70

    return f"Sentiment: {sentiment} (confidence: {confidence}%)"


@tool
def summarize_points(text: str) -> str:
    """Extract and summarize key points from text."""
    sentences = text.replace('\n', ' ').split('.')
    key_points = [s.strip() for s in sentences if len(s.strip()) > 30][:5]
    if key_points:
        return "Key Points:\n" + "\n".join(f"• {p}" for p in key_points)
    return "No key points extracted."


@tool
def format_output(content: str, format_type: str = "paragraph") -> str:
    """Format content into a specific structure (paragraph, bullets, numbered)."""
    if format_type == "bullets":
        lines = content.split('\n')
        return "\n".join(f"• {line.strip()}" for line in lines if line.strip())
    elif format_type == "numbered":
        lines = content.split('\n')
        return "\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(lines) if line.strip())
    return content


# Tool registry - maps agent types to their available tools
AGENT_TOOLS = {
    "researcher": [search_knowledge_base, extract_keywords],
    "analyst": [analyze_sentiment, summarize_points, extract_keywords],
    "writer": [format_output, summarize_points]
}
