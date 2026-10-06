"""Execution-policy instructions supplied at finalization and stream recovery."""

MAX_HOPS_FINALIZE_PROMPT = (
    "The tool budget for this run is exhausted. Write the answer now from the "
    "evidence already available in this conversation. No further tool use is "
    "permitted. Explain any unresolved questions or missing evidence rather "
    "than filling those gaps with guesses."
)

CONTINUE_PARTIAL_ASSISTANT_PROMPT = (
    "Resume the incomplete assistant response by supplying only its unwritten "
    "remainder. Treat the existing response as a fixed prefix: do not restate, "
    "revise, or summarize it. Join the continuation naturally to that prefix "
    "without commentary about resuming the response."
)
