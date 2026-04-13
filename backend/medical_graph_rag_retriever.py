# Medical Graph RAG：Neo4j 驱动 + 实体名列表（data/entity_names_export）匹配 query。
# 可选 aliases.json：别名 -> 规范 name（与 Neo4j 节点 name 一致），仅 Python 侧解析，不改 Cypher。
# 环境变量：NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD；可选 ENTITY_NAMES_DIR

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else default


def _default_entity_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "entity_names_export"


@lru_cache(maxsize=1)
def _load_name_lists() -> tuple[list[str], list[str], list[str]]:
    """从 ENTITY_NAMES_DIR 加载疾病/药物/基因名（小写、去空行）。"""
    base = Path(_env("ENTITY_NAMES_DIR", "") or _default_entity_dir())
    diseases, drugs, genes = [], [], []
    if not base.is_dir():
        return diseases, drugs, genes

    def _read(fname: str) -> list[str]:
        p = base / fname
        if not p.is_file():
            return []
        return [
            line.strip().lower()
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]

    diseases = _read("diseases.txt")
    drugs = _read("drugs.txt")
    genes = _read("genes.txt")
    return diseases, drugs, genes


@lru_cache(maxsize=1)
def _load_aliases_json() -> Dict[str, Dict[str, str]]:
    """
    加载 aliases.json：别名（可中文）-> 图内规范 name（小写，须与 Neo4j 节点 name 一致）。
    文件与 diseases.txt 同目录；不存在则视为无别名。
    """
    base = Path(_env("ENTITY_NAMES_DIR", "") or _default_entity_dir())
    p = base / "aliases.json"
    empty = {"diseases": {}, "drugs": {}, "genes": {}}
    if not p.is_file():
        return empty
    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return empty
    out: Dict[str, Dict[str, str]] = {"diseases": {}, "drugs": {}, "genes": {}}
    for key in out:
        block = raw.get(key)
        if not isinstance(block, dict):
            continue
        for a, c in block.items():
            ak = str(a).strip().lower()
            ck = str(c).strip().lower()
            if ak and ck:
                out[key][ak] = ck
    return out


def _extract_category_entities(
    query_lower: str,
    name_list: List[str],
    alias_map: Dict[str, str],
    max_terms: int = 12,
) -> List[str]:
    """
    子串匹配 name_list + 别名键；命中后映射为规范 name（供 Cypher {name: $x}）。
    """
    term_to_canonical: Dict[str, str] = {}
    for n in name_list:
        n = n.strip().lower()
        if n:
            term_to_canonical[n] = n
    for a, c in alias_map.items():
        if a and c:
            term_to_canonical[a] = c
    if not term_to_canonical:
        return []
    search_keys = sorted(term_to_canonical.keys(), key=len, reverse=True)
    matched_keys = _match_terms_in_query(query_lower, search_keys, max_terms=max_terms)
    out: List[str] = []
    seen: set[str] = set()
    for k in matched_keys:
        canon = term_to_canonical[k]
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def _match_terms_in_query(query_lower: str, candidates: List[str], max_terms: int = 12) -> List[str]:
    """按长度降序匹配，优先长词（如 coronary artery disease）。"""
    if not candidates or not query_lower:
        return []
    sorted_c = sorted(set(candidates), key=len, reverse=True)
    found: list[str] = []
    for term in sorted_c:
        if len(term) < 2:
            continue
        if term in query_lower:
            found.append(term)
            if len(found) >= max_terms:
                break
    return found


class MedicalGraphRAGRetriever:
    """从 Neo4j 医疗知识图谱检索文献与子图上下文，供 RAG 拼 prompt。"""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        uri = uri or _env("NEO4J_URI", "neo4j://127.0.0.1:7687")
        user = user or _env("NEO4J_USER", "neo4j")
        password = password or _env("NEO4J_PASSWORD")
        if not password:
            raise ValueError("Neo4j password 未设置：请设置环境变量 NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def extract_entities(self, question: str) -> Dict[str, List[str]]:
        """基于 entity_names + aliases.json 在 query 中做子串匹配，输出规范 name（与 Neo4j 一致）。"""
        q = question.lower()
        aliases = _load_aliases_json()
        diseases_src, drugs_src, genes_src = _load_name_lists()
        if not diseases_src and not drugs_src and not genes_src:
            return self._fallback_keyword_entities(q, aliases)

        diseases = _extract_category_entities(q, diseases_src, aliases["diseases"])
        drugs = _extract_category_entities(q, drugs_src, aliases["drugs"])
        genes = _extract_category_entities(q, genes_src, aliases["genes"])
        return {"diseases": diseases, "drugs": drugs, "genes": genes}

    def _fallback_keyword_entities(
        self, q: str, aliases: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, List[str]]:
        aliases = aliases or _load_aliases_json()
        diseases = [
            "diabetes",
            "cancer",
            "hypertension",
            "stroke",
            "alzheimer",
            "infection",
            "tumor",
            "obesity",
            "heart failure",
            "parkinson",
        ]
        drugs = [
            "insulin",
            "metformin",
            "aspirin",
            "chemotherapy",
            "immunotherapy",
            "statin",
            "antibiotic",
            "antiviral",
        ]
        return {
            "diseases": _extract_category_entities(q, diseases, aliases["diseases"]),
            "drugs": _extract_category_entities(q, drugs, aliases["drugs"]),
            "genes": _extract_category_entities(q, [], aliases["genes"]),
        }

    def retrieve_articles(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """按共现文献检索 Article（图内实体名为小写）。"""
        entities = self.extract_entities(question)
        with self.driver.session() as session:
            if entities["diseases"] and entities["drugs"]:
                disease = entities["diseases"][0]
                drug = entities["drugs"][0]
                query = """
                MATCH (a:Article)-[:MENTIONS_DISEASE]->(d:Disease {name: $disease})
                MATCH (a)-[:MENTIONS_DRUG]->(dr:Drug {name: $drug})
                RETURN a.pmid AS pmid, a.title AS title, a.abstract AS abstract
                LIMIT $top_k
                """
                result = session.run(query, disease=disease, drug=drug, top_k=top_k)
            elif entities["diseases"]:
                disease = entities["diseases"][0]
                query = """
                MATCH (a:Article)-[:MENTIONS_DISEASE]->(d:Disease {name: $disease})
                RETURN a.pmid AS pmid, a.title AS title, a.abstract AS abstract
                ORDER BY a.year DESC
                LIMIT $top_k
                """
                result = session.run(query, disease=disease, top_k=top_k)
            elif entities["drugs"]:
                drug = entities["drugs"][0]
                query = """
                MATCH (a:Article)-[:MENTIONS_DRUG]->(dr:Drug {name: $drug})
                RETURN a.pmid AS pmid, a.title AS title, a.abstract AS abstract
                ORDER BY a.year DESC
                LIMIT $top_k
                """
                result = session.run(query, drug=drug, top_k=top_k)
            else:
                return []
            return [dict(record) for record in result]

    def retrieve_graph_facts(
        self, question: str, drug_limit: int = 8, gene_limit: int = 8
    ) -> Dict[str, Any]:
        """检索 TREATS / ASSOCIATED_WITH / TARGETS 等结构化边。"""
        entities = self.extract_entities(question)
        out: Dict[str, Any] = {"diseases": [], "drugs": [], "genes": []}
        with self.driver.session() as session:
            for disease in entities["diseases"]:
                d = disease
                treats = session.run(
                    """
                    MATCH (dr:Drug)-[r:TREATS]->(x:Disease {name: $disease})
                    RETURN dr.name AS drug, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    disease=d,
                    lim=drug_limit,
                )
                genes = session.run(
                    """
                    MATCH (g:Gene)-[r:ASSOCIATED_WITH]->(x:Disease {name: $disease})
                    RETURN g.name AS gene, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    disease=d,
                    lim=gene_limit,
                )
                out["diseases"].append(
                    {
                        "name": d,
                        "treats": [dict(r) for r in treats],
                        "associated_genes": [dict(r) for r in genes],
                    }
                )
            for drug in entities["drugs"]:
                dr = drug
                targets = session.run(
                    """
                    MATCH (x:Drug {name: $drug})-[r:TARGETS]->(g:Gene)
                    RETURN g.name AS gene, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    drug=dr,
                    lim=gene_limit,
                )
                treats_what = session.run(
                    """
                    MATCH (x:Drug {name: $drug})-[r:TREATS]->(d:Disease)
                    RETURN d.name AS disease, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    drug=dr,
                    lim=drug_limit,
                )
                out["drugs"].append(
                    {
                        "name": dr,
                        "targets": [dict(r) for r in targets],
                        "treats_diseases": [dict(r) for r in treats_what],
                    }
                )
            for gene in entities.get("genes", []):
                gname = gene
                assoc = session.run(
                    """
                    MATCH (g:Gene {name: $gene})-[r:ASSOCIATED_WITH]->(d:Disease)
                    RETURN d.name AS disease, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    gene=gname,
                    lim=gene_limit,
                )
                targets_from = session.run(
                    """
                    MATCH (dr:Drug)-[r:TARGETS]->(g:Gene {name: $gene})
                    RETURN dr.name AS drug, r.evidence_count AS evidence
                    ORDER BY r.evidence_count DESC
                    LIMIT $lim
                    """,
                    gene=gname,
                    lim=drug_limit,
                )
                out["genes"].append(
                    {
                        "name": gname,
                        "associated_diseases": [dict(r) for r in assoc],
                        "targeted_by_drugs": [dict(r) for r in targets_from],
                    }
                )
        return out

    def format_graph_facts_text(self, facts: Dict[str, Any]) -> str:
        lines: List[str] = []
        for block in facts.get("diseases", []):
            name = block["name"]
            lines.append(f"[疾病: {name}]")
            for row in block.get("treats", [])[:8]:
                lines.append(f"  - 药物证据(TREATS): {row['drug']} (共现文献约 {row['evidence']})")
            for row in block.get("associated_genes", [])[:8]:
                lines.append(f"  - 相关基因(ASSOCIATED_WITH): {row['gene']} (证据 {row['evidence']})")
        for block in facts.get("drugs", []):
            name = block["name"]
            lines.append(f"[药物: {name}]")
            for row in block.get("targets", [])[:8]:
                lines.append(f"  - 靶点基因(TARGETS): {row['gene']} (证据 {row['evidence']})")
            for row in block.get("treats_diseases", [])[:8]:
                lines.append(f"  - 治疗疾病(TREATS): {row['disease']} (证据 {row['evidence']})")
        for block in facts.get("genes", []):
            name = block["name"]
            lines.append(f"[基因: {name}]")
            for row in block.get("associated_diseases", [])[:8]:
                lines.append(f"  - 关联疾病(ASSOCIATED_WITH): {row['disease']} (证据 {row['evidence']})")
            for row in block.get("targeted_by_drugs", [])[:8]:
                lines.append(f"  - 靶向药物(TARGETS): {row['drug']} (证据 {row['evidence']})")
        return "\n".join(lines) if lines else ""

    def format_articles_text(self, articles: List[Dict[str, Any]], max_abstract_chars: int = 1200) -> str:
        chunks: List[str] = []
        for i, a in enumerate(articles, 1):
            ab = a.get("abstract") or ""
            if len(ab) > max_abstract_chars:
                ab = ab[:max_abstract_chars] + "..."
            chunks.append(
                f"[文献 {i}] PMID:{a.get('pmid')}\n标题: {a.get('title')}\n摘要: {ab}"
            )
        return "\n\n".join(chunks)

    def _compose_llm_context(self, arts: List[Dict[str, Any]], facts: Dict[str, Any]) -> str:
        parts: List[str] = []
        gtxt = self.format_graph_facts_text(facts)
        if gtxt:
            parts.append("=== 知识图谱结构化证据 ===\n" + gtxt)
        if arts:
            parts.append("=== 相关文献（来自图谱共现检索）===\n" + self.format_articles_text(arts))
        return "\n\n".join(parts)

    def facts_to_subgraph_viz(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """将 retrieve_graph_facts 结果转为前端 vis-network 可用的 nodes/edges（仅 Python，不改 Neo4j）。"""
        nodes: Dict[str, Dict[str, str]] = {}
        edges: List[Dict[str, str]] = []

        def nid(kind: str, name: str) -> str:
            return f"{kind}:{name}"

        def add_node(kind: str, name: str) -> str:
            i = nid(kind, name)
            if i not in nodes:
                nodes[i] = {"id": i, "label": (name or "")[:120], "group": kind}
            return i

        for block in facts.get("diseases", []):
            dn = add_node("Disease", block["name"])
            for row in block.get("treats", []):
                drn = add_node("Drug", row["drug"])
                edges.append({"from": drn, "to": dn, "label": "TREATS"})
            for row in block.get("associated_genes", []):
                gn = add_node("Gene", row["gene"])
                edges.append({"from": gn, "to": dn, "label": "ASSOCIATED_WITH"})
        for block in facts.get("drugs", []):
            drn = add_node("Drug", block["name"])
            for row in block.get("targets", []):
                gn = add_node("Gene", row["gene"])
                edges.append({"from": drn, "to": gn, "label": "TARGETS"})
            for row in block.get("treats_diseases", []):
                dn = add_node("Disease", row["disease"])
                edges.append({"from": drn, "to": dn, "label": "TREATS"})
        for block in facts.get("genes", []):
            gn = add_node("Gene", block["name"])
            for row in block.get("associated_diseases", []):
                dn = add_node("Disease", row["disease"])
                edges.append({"from": gn, "to": dn, "label": "ASSOCIATED_WITH"})
            for row in block.get("targeted_by_drugs", []):
                drn = add_node("Drug", row["drug"])
                edges.append({"from": drn, "to": gn, "label": "TARGETS"})

        seen_e: set[tuple[str, str, str]] = set()
        deduped: List[Dict[str, str]] = []
        for e in edges:
            key = (e["from"], e["to"], e["label"])
            if key in seen_e:
                continue
            seen_e.add(key)
            deduped.append(e)
            if len(deduped) >= 80:
                break

        return {"nodes": list(nodes.values()), "edges": deduped}

    def build_trace_for_ui(
        self, question: str, top_k: int = 5
    ) -> tuple[str, Dict[str, Any], Dict[str, List[str]], Dict[str, Any]]:
        """单次检索：LLM 上下文文本 + 可视化子图 + 实体列表 + facts 结构化结果。"""
        entities = self.extract_entities(question)
        arts = self.retrieve_articles(question, top_k=top_k)
        facts = self.retrieve_graph_facts(question)
        text = self._compose_llm_context(arts, facts)
        subgraph = self.facts_to_subgraph_viz(facts)
        return text, subgraph, entities, facts

    def build_context_for_llm(self, question: str, top_k: int = 5) -> str:
        """结构化子图 + 文献摘要，拼成一段文本。"""
        arts = self.retrieve_articles(question, top_k=top_k)
        facts = self.retrieve_graph_facts(question)
        return self._compose_llm_context(arts, facts)


def merge_with_vector_chunks(
    graph_text: str,
    vector_texts: List[str],
    dedupe_substrings: bool = True,
) -> str:
    kept: List[str] = []
    for t in vector_texts:
        if not t or not t.strip():
            continue
        if dedupe_substrings and graph_text and t[:200] in graph_text:
            continue
        kept.append(t.strip())
    body = "\n\n---\n\n".join(kept) if kept else ""
    if graph_text and body:
        return graph_text + "\n\n---\n\n[向量检索补充]\n" + body
    return graph_text or body
