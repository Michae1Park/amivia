# Project 12: Tool-calling agent (optional, LLM)

Layer: agent tooling — see `docs/architecture.md` §8. The point where the
system stops being "RAG over a fixed catalog" and becomes an agent that acts
— no classic-recsys analog, a ranking model never decides mid-run to call an
external service.

**Problem:** let the conversational agent fetch live data mid-conversation
(e.g. current weather or flight price for a candidate destination) instead
of only ever grounding in the static catalog.

**Data:** a mock external API (weather or price lookup) standing in for a
real one.

**Algorithm/tech:** not an algorithm-comparison layer — tool use / function
calling, optionally exposed via MCP (Model Context Protocol) so the tool
isn't bespoke to one provider's schema.

**Status:** not yet implemented (README/design only). Wire one mock tool
into the Project 11 conversational flow to see the retrieve-then-act loop
firsthand.
