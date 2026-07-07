def mock_search(subtopic: str) -> str:
    """Return deterministic, canned 'search results' for a subtopic.

    Framework-neutral on purpose: this is a plain function so both the Strands
    and LangGraph apps can wrap it in their own tool abstraction. Deterministic
    output keeps test runs free and reproducible. Swap for a real search later
    behind this same signature.
    """
    return (
        f"Search results for '{subtopic}':\n"
        f"1. Overview of {subtopic}: a concise, factual summary.\n"
        f"2. Key considerations regarding {subtopic}.\n"
        f"3. A commonly cited example involving {subtopic}."
    )
