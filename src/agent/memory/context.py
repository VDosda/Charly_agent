from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agent.memory.models import MemoryBundle, RetrievedEpisode, RetrievedMemoryItem


@dataclass(frozen=True)
class ContextBudgets:
    """
    Char budgets (simple and deterministic).
    Token budgets vary by model; char budgets are good enough for v1.
    """
    lt_max_chars: int = 1800
    mt_max_chars: int = 1800
    st_max_chars: int = 2400  # if you also render ST snippet here
    total_max_chars: int = 4800


@dataclass(frozen=True)
class ContextRenderOptions:
    include_scores: bool = False
    include_evidence: bool = False
    # if True: render headings even if empty (usually False)
    render_empty_blocks: bool = False


def build_memory_context_blocks(
    bundle: MemoryBundle,
    *,
    budgets: ContextBudgets = ContextBudgets(),
    opts: ContextRenderOptions = ContextRenderOptions(),
) -> List[Dict[str, str]]:
    """
    Returns a list of {role, content} blocks (usually role=system)
    to inject into the LLM messages.

    Format is stable and model-friendly:
      - LONG-TERM MEMORY
      - MEDIUM-TERM EPISODES
    """
    lt_text = _render_lt(bundle.lt_items, opts=opts)
    mt_text = _render_mt(bundle.mt_episodes, opts=opts)

    lt_text = _truncate(lt_text, budgets.lt_max_chars)
    mt_text = _truncate(mt_text, budgets.mt_max_chars)

    # Enforce total budget by trimming MT first (usually less critical than LT)
    total = len(lt_text) + len(mt_text)
    if total > budgets.total_max_chars:
        overflow = total - budgets.total_max_chars
        mt_text = _truncate(mt_text, max(0, len(mt_text) - overflow))

    blocks: List[Dict[str, str]] = []

    if lt_text or opts.render_empty_blocks:
        blocks.append(
            {
                "role": "system",
                "content": "LONG-TERM MEMORY (facts/preferences/constraints):\n" + (lt_text or "(none)"),
            }
        )

    if mt_text or opts.render_empty_blocks:
        blocks.append(
            {
                "role": "system",
                "content": "MEDIUM-TERM EPISODES (recent summarized context):\n" + (mt_text or "(none)"),
            }
        )

    return blocks


def _render_lt(items: Sequence[RetrievedMemoryItem], *, opts: ContextRenderOptions) -> str:
    if not items:
        return ""

    lines: List[str] = []
    for r in items:
        it = r.item
        key = f" ({it.mem_key})" if it.mem_key else ""
        base = f"- [{it.kind}{key}] {it.value}"

        if opts.include_scores:
            base += f" [score={r.score:.3f} imp={it.importance:.2f} conf={it.confidence:.2f}]"

        if opts.include_evidence:
            ev = r.evidence
            if ev.vector_distance is not None:
                base += f" [dist={ev.vector_distance:.3f} rec={ev.recency_score:.3f}]"

        lines.append(base)

    return "\n".join(lines)


def _render_mt(eps: Sequence[RetrievedEpisode], *, opts: ContextRenderOptions) -> str:
    if not eps:
        return ""

    lines: List[str] = []
    for r in eps:
        ep = r.episode
        base = f"- {ep.summary}"

        if opts.include_scores:
            base += f" [score={r.score:.3f} imp={ep.importance:.2f} conf={ep.confidence:.2f}]"

        if opts.include_evidence:
            ev = r.evidence
            if ev.vector_distance is not None:
                base += f" [dist={ev.vector_distance:.3f} rec={ev.recency_score:.3f}]"

        lines.append(base)

    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"