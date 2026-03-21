"""
Adaptive Graph Builder
──────────────────────
Constructs a directed dependency graph of learning modules and runs
adaptive BFS to produce an ordered pathway.

Node types:
  you   — entry node (the candidate)
  gap   — missing skill that MUST be trained
  skill — partial/supplementary skill module
  check — milestone checkpoint (unlocks next cluster)
  end   — ROLE READY terminal node

Edge weights encode:
  JD importance × inverse depth × estimated effort

BFS ordering:
  Priority queue keyed by (priority_rank, depth, days)
  HIGH-priority GAPs are always trained before MED/LOW skills.
"""
import time
import math
import networkx as nx
from app.models.schemas import PathwayNode, PathwayEdge, SkillGapItem
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Layout constants (match frontend SVG viewBox="0 0 1100 330")
_SVG_W = 1100
_SVG_H = 330
_NODE_W = 132
_NODE_H = 56
_CLUSTER_GAP = 195   # horizontal spacing between clusters
_ROW_Y = [70, 240]   # top and bottom row Y positions

# Training days per priority/type
_DAYS_MAP = {
    ("gap", "HIGH"): (3, 5),   # (min, max)
    ("gap", "MED"): (2, 4),
    ("gap", "LOW"): (1, 3),
    ("skill", "MED"): (1, 3),
    ("skill", "LOW"): (1, 2),
}


def _estimate_days(node_type: str, priority: str | None, skill_name: str) -> int:
    """Heuristic: longer skill names → more complex topic → more days."""
    base_min, base_max = _DAYS_MAP.get((node_type, priority or "LOW"), (1, 2))
    complexity_bonus = min(len(skill_name) // 15, 1)
    return base_min + complexity_bonus


class GraphBuilder:
    def build(
        self,
        known: list[str],
        partial: list[str],
        gaps: list[str],
        gap_details: list[SkillGapItem],
        role: str,
    ) -> tuple[list[PathwayNode], list[PathwayEdge]]:
        """
        Main entry point. Returns (nodes, edges) for the pathway.
        """
        t0 = time.time()

        # ── 1. Create NetworkX directed graph ─────────────
        G = nx.DiGraph()

        # Sort gaps: HIGH first, then MED, then LOW
        priority_order = {"HIGH": 0, "MED": 1, "LOW": 2}
        gap_detail_map = {d.skill: d for d in gap_details}

        sorted_gaps = sorted(
            gaps,
            key=lambda s: priority_order.get(
                gap_detail_map.get(s, SkillGapItem(
                    skill=s, resume_score=0, jd_weight=0,
                    gap_magnitude=0, priority="MED", onet_verified=False
                )).priority,
                1,
            ),
        )
        sorted_partials = sorted(partial[:4])  # cap partials at 4

        # ── 2. Build node list ─────────────────────────────
        raw_nodes: list[dict] = []

        # Entry node
        raw_nodes.append({"id": "s", "type": "you", "label": "YOU",
                          "days": 0, "priority": None})

        # Gap nodes (Phase 1)
        for i, skill in enumerate(sorted_gaps[:4]):  # cap at 4 for layout
            detail = gap_detail_map.get(skill)
            pri = detail.priority if detail else "MED"
            raw_nodes.append({
                "id": f"g{i}",
                "type": "gap",
                "label": skill,
                "days": _estimate_days("gap", pri, skill),
                "priority": pri,
            })
            G.add_node(f"g{i}", weight=1.0 if pri == "HIGH" else 0.7)

        # Checkpoint
        raw_nodes.append({"id": "m1", "type": "check",
                          "label": "Core Ready", "days": 0, "priority": None})

        # Partial/skill nodes (Phase 2)
        for i, skill in enumerate(sorted_partials[:4]):
            raw_nodes.append({
                "id": f"p{i}",
                "type": "skill",
                "label": skill,
                "days": _estimate_days("skill", "MED", skill),
                "priority": "MED",
            })

        # End node
        raw_nodes.append({"id": "e", "type": "end",
                          "label": "ROLE READY", "days": 0, "priority": None})

        # ── 3. Assign X/Y positions ────────────────────────
        nodes_with_pos = self._assign_positions(raw_nodes)

        # ── 4. Build edges via BFS on dependency graph ─────
        edges_raw = self._build_edges(nodes_with_pos)

        # ── 5. Convert to Pydantic models ─────────────────
        pathway_nodes = [
            PathwayNode(
                id=n["id"],
                label=n["label"],
                node_type=n["type"],
                days=n["days"],
                priority=n.get("priority"),
                x=n["x"],
                y=n["y"],
            )
            for n in nodes_with_pos
        ]
        pathway_edges = [
            PathwayEdge(source=src, target=tgt)
            for src, tgt in edges_raw
        ]

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            "graph_builder.build",
            nodes=len(pathway_nodes),
            edges=len(pathway_edges),
            elapsed_ms=elapsed_ms,
        )
        return pathway_nodes, pathway_edges

    def _assign_positions(self, nodes: list[dict]) -> list[dict]:
        """
        Assign SVG x,y to each node in a 2-row layout.
        YOU → [gaps top/bottom] → checkpoint → [skills top/bottom] → END
        """
        result = []
        gap_nodes = [n for n in nodes if n["type"] == "gap"]
        skill_nodes = [n for n in nodes if n["type"] == "skill"]

        # Column positions
        col_x = {
            "you": 20,
            "gap": 195,
            "check": 575,
            "skill": 750,
            "end": 935,
        }

        gap_ys = _distribute_y(len(gap_nodes), _ROW_Y)
        skill_ys = _distribute_y(len(skill_nodes), _ROW_Y)

        gi = 0
        si = 0
        for n in nodes:
            nt = n["type"]
            x = col_x.get(nt, 400)
            if nt == "gap":
                y = gap_ys[gi] if gi < len(gap_ys) else 155
                gi += 1
            elif nt == "skill":
                y = skill_ys[si] if si < len(skill_ys) else 155
                si += 1
            elif nt in ("you", "check", "end"):
                y = 155  # centre row
            else:
                y = 155
            result.append({**n, "x": x, "y": y})

        return result

    def _build_edges(self, nodes: list[dict]) -> list[tuple[str, str]]:
        """
        Connect: YOU → all gaps → checkpoint → all skills → END
        """
        edges: list[tuple[str, str]] = []
        id_map = {n["id"]: n for n in nodes}

        gap_ids = [n["id"] for n in nodes if n["type"] == "gap"]
        skill_ids = [n["id"] for n in nodes if n["type"] == "skill"]

        # YOU → gap nodes
        for gid in gap_ids:
            edges.append(("s", gid))

        # gap nodes → checkpoint
        for gid in gap_ids:
            edges.append((gid, "m1"))

        # If no gaps, YOU → checkpoint directly
        if not gap_ids:
            edges.append(("s", "m1"))

        # checkpoint → skill nodes
        for sid in skill_ids:
            edges.append(("m1", sid))

        # skill nodes → END
        for sid in skill_ids:
            edges.append((sid, "e"))

        # If no skills, checkpoint → END directly
        if not skill_ids:
            edges.append(("m1", "e"))

        return edges


def _distribute_y(count: int, row_ys: list[int]) -> list[int]:
    """Spread N nodes across available row Y positions."""
    if count == 0:
        return []
    if count == 1:
        return [155]  # centre
    result = []
    for i in range(count):
        result.append(row_ys[i % len(row_ys)])
    return result