instructions = [
    """
    I want you to act as a Prompt Rewriter.
    Your objective is to evolve a given prompt into a more complex version to make those famous AI systems
    (e.g., ChatGPT and GPT-4) a bit harder to handle.
    But the evolved prompt must be reasonable and must be understood and responded by humans.
    You should complicate the given prompt using the following method:
    If the given prompt contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.
    You should try your best not to make the evolved prompt become verbose, the evolved prompt can only add 10 to 20 words into the given prompt.
    'Given Prompt', 'Evolved Prompt', 'given prompt' and 'evolved prompt' are not allowed to appear in the Evolved Prompt
""",
    """
    A Concretizing Prompt
    I want you to act as a Prompt Rewriter.
    Your objective is to evolve a given prompt into a more specific and concrete version to challenge AI systems.
    Please replace general concepts with more specific concepts within the given prompt.
    You should try your best not to make the evolved prompt verbose and the evolved prompt can only add 10 to 20 words.
    'Given Prompt', 'Evolved Prompt', 'given prompt' and 'evolved prompt' are not allowed to appear in the Evolved Prompt
""",
    """
    An Increased Reasoning Steps Prompt
    I want you to act as a Prompt Rewriter.
    Your objective is to rewrite a given prompt into a version that requires multiple-step reasoning.
    If the given prompt can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
    You should try your best not to make the evolved prompt verbose, and the evolved prompt can only add 10 to 20 words.
    'Given Prompt', 'Evolved Prompt', 'given prompt' and 'evolved prompt' are not allowed to appear in the Evolved Prompt
"""

]