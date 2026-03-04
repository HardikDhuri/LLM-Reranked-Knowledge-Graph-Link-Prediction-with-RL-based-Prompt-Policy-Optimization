"""Integration tests with all external dependencies mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.candidates import filter_candidates_tail, generate_tail_candidates
from src.eval.metrics import RankingResult, compute_all_metrics
from src.rl.features import QueryFeatureExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm_client(score: float = 0.8) -> MagicMock:
    client = MagicMock()
    client.chat_completion_json.return_value = {
        "parsed_json": {"score": score},
        "content": f'{{"score": {score}}}',
        "usage": {"total_tokens": 50},
        "latency_s": 0.1,
    }
    client.stats.return_value = {
        "total_calls": 1,
        "total_failures": 0,
        "total_tokens": 50,
        "total_latency_s": 0.1,
        "avg_latency_s": 0.1,
    }
    return client


def _make_mock_resolver() -> MagicMock:
    resolver = MagicMock()
    resolver.mid_to_text.side_effect = lambda mid: {
        "mid": mid,
        "qid": "Q1",
        "label": mid,
        "description": "",
    }
    return resolver


def _make_mock_template(tid: str = "test") -> MagicMock:
    template = MagicMock()
    template.id = tid
    template.render.return_value = [
        {"role": "system", "content": "You are a KG expert."},
        {"role": "user", "content": "Is this triple plausible?"},
    ]
    return template


# ---------------------------------------------------------------------------
# Integration: TripleScorer → score_triple returns float
# ---------------------------------------------------------------------------


def test_triple_scorer_full_pipeline():
    from src.models.scorer import TripleScorer

    llm = _make_mock_llm_client(score=0.75)
    resolver = _make_mock_resolver()
    template = _make_mock_template("t1")

    scorer = TripleScorer(llm, resolver, template)
    score = scorer.score_triple("/m/h1", "/r/rel", "/m/t1")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.75)


def test_triple_scorer_returns_zero_on_llm_failure():
    from src.models.scorer import TripleScorer

    llm = MagicMock()
    llm.chat_completion_json.side_effect = RuntimeError("LLM error")
    resolver = _make_mock_resolver()
    template = _make_mock_template()

    scorer = TripleScorer(llm, resolver, template)
    score = scorer.score_triple("/m/h1", "/r/rel", "/m/t1")

    assert score == 0.0


def test_triple_scorer_uses_cache_on_second_call():
    from src.models.scorer import TripleScorer

    llm = _make_mock_llm_client(score=0.5)
    resolver = _make_mock_resolver()
    template = _make_mock_template()

    scorer = TripleScorer(llm, resolver, template)
    scorer.score_triple("/m/h1", "/r/rel", "/m/t1")
    scorer.score_triple("/m/h1", "/r/rel", "/m/t1")

    assert llm.chat_completion_json.call_count == 1


# ---------------------------------------------------------------------------
# Integration: LLMReranker → rerank_tail_candidates returns RankingResult
# ---------------------------------------------------------------------------


def test_llm_reranker_full_pipeline():
    from src.models.reranker import LLMReranker
    from src.models.scorer import TripleScorer

    # Wire up scorer with mocked LLM that scores true_tail highest
    llm = MagicMock()
    call_count = [0]

    def side_effect(messages):
        call_count[0] += 1
        # First call is for true_tail "/m/t0", give it a high score
        score = 0.9 if call_count[0] == 1 else 0.1
        return {
            "parsed_json": {"score": score},
            "content": f'{{"score": {score}}}',
            "usage": {"total_tokens": 10},
            "latency_s": 0.01,
        }

    llm.chat_completion_json.side_effect = side_effect
    resolver = _make_mock_resolver()
    template = _make_mock_template()

    scorer = TripleScorer(llm, resolver, template)
    reranker = LLMReranker(scorer)

    result = reranker.rerank_tail_candidates(
        head="/m/h1",
        relation="/r/rel",
        true_tail="/m/t0",
        candidates=["/m/t0", "/m/t1", "/m/t2"],
    )

    assert isinstance(result, RankingResult)
    assert result.true_rank == 1
    assert result.num_candidates == 3


def test_llm_reranker_true_tail_not_in_candidates():
    from src.models.reranker import LLMReranker
    from src.models.scorer import TripleScorer

    llm = _make_mock_llm_client(score=0.5)
    resolver = _make_mock_resolver()
    template = _make_mock_template()
    scorer = TripleScorer(llm, resolver, template)
    reranker = LLMReranker(scorer)

    candidates = ["/m/t1", "/m/t2"]
    result = reranker.rerank_tail_candidates(
        head="/m/h1",
        relation="/r/rel",
        true_tail="/m/t0_missing",
        candidates=candidates,
    )

    assert result.true_rank == len(candidates) + 1


def test_llm_reranker_increments_total_queries():
    from src.models.reranker import LLMReranker
    from src.models.scorer import TripleScorer

    llm = _make_mock_llm_client()
    scorer = TripleScorer(llm, _make_mock_resolver(), _make_mock_template())
    reranker = LLMReranker(scorer)

    assert reranker.total_queries == 0
    reranker.rerank_tail_candidates("/m/h", "/r/r", "/m/t0", ["/m/t0", "/m/t1"])
    assert reranker.total_queries == 1


# ---------------------------------------------------------------------------
# Integration: RLPromptSelector → select_and_score returns results + updates agent
# ---------------------------------------------------------------------------


def test_rl_prompt_selector_full_pipeline():
    from src.rl.prompt_selector import RLPromptSelector

    llm = _make_mock_llm_client(score=0.6)
    resolver = _make_mock_resolver()

    prompt_manager = MagicMock()
    template_ids = ["t1", "t2", "t3"]
    prompt_manager.list_ids.return_value = template_ids
    prompt_manager.get.side_effect = lambda tid: _make_mock_template(tid)

    feature_extractor = QueryFeatureExtractor()

    selector = RLPromptSelector(
        llm_client=llm,
        resolver=resolver,
        prompt_manager=prompt_manager,
        feature_extractor=feature_extractor,
        agent_type="linucb",
    )

    candidates = ["/m/t0", "/m/t1", "/m/t2"]
    template_id, scored, true_rank = selector.select_and_score(
        "/m/h1", "/r/rel", "/m/t0", candidates
    )

    assert template_id in template_ids
    assert len(scored) == len(candidates)
    assert isinstance(true_rank, int)
    assert true_rank >= 1
    assert selector.agent.total_steps == 1


def test_rl_prompt_selector_epsilon_greedy():
    from src.rl.prompt_selector import RLPromptSelector

    llm = _make_mock_llm_client()
    resolver = _make_mock_resolver()

    prompt_manager = MagicMock()
    template_ids = ["t1", "t2"]
    prompt_manager.list_ids.return_value = template_ids
    prompt_manager.get.side_effect = lambda tid: _make_mock_template(tid)

    feature_extractor = QueryFeatureExtractor()

    selector = RLPromptSelector(
        llm_client=llm,
        resolver=resolver,
        prompt_manager=prompt_manager,
        feature_extractor=feature_extractor,
        agent_type="epsilon_greedy",
    )

    _, scored, _ = selector.select_and_score(
        "/m/h1", "/r/rel", "/m/t0", ["/m/t0", "/m/t1"]
    )
    assert len(scored) == 2


# ---------------------------------------------------------------------------
# Integration: BudgetAgent → decide + record_decision with correct budget tracking
# ---------------------------------------------------------------------------


def test_budget_agent_full_pipeline():
    from src.rl.budget_agent import BudgetAgent

    feature_extractor = QueryFeatureExtractor()
    agent = BudgetAgent(
        feature_extractor=feature_extractor,
        total_budget=5,
        agent_type="linucb",
        cost_per_llm_query=1,
    )

    assert agent.remaining_budget == 5
    action = agent.decide("/m/h", "/r/r", "/m/t")
    assert action in (BudgetAgent.ARM_EMBEDDING, BudgetAgent.ARM_LLM)

    agent.record_decision(
        "/m/h", "/r/r", "/m/t", action=BudgetAgent.ARM_LLM, reward=0.5
    )
    assert agent.remaining_budget == 4
    assert agent.llm_queries == 1
    assert agent.total_queries == 1


def test_budget_agent_exhausted_budget_returns_embedding():
    from src.rl.budget_agent import BudgetAgent

    feature_extractor = QueryFeatureExtractor()
    agent = BudgetAgent(
        feature_extractor=feature_extractor,
        total_budget=0,
        agent_type="linucb",
    )

    action = agent.decide("/m/h", "/r/r", "/m/t")
    assert action == BudgetAgent.ARM_EMBEDDING


def test_budget_agent_budget_tracking_multiple_llm_calls():
    from src.rl.budget_agent import BudgetAgent

    feature_extractor = QueryFeatureExtractor()
    agent = BudgetAgent(
        feature_extractor=feature_extractor,
        total_budget=3,
        agent_type="epsilon_greedy",
        cost_per_llm_query=1,
    )

    for _ in range(3):
        agent.record_decision("/m/h", "/r/r", "/m/t", BudgetAgent.ARM_LLM, reward=0.5)

    assert agent.remaining_budget == 0
    action = agent.decide("/m/h", "/r/r", "/m/t")
    assert action == BudgetAgent.ARM_EMBEDDING


# ---------------------------------------------------------------------------
# Integration: eval pipeline — generate_candidates → filter → rank → compute_metrics
# ---------------------------------------------------------------------------


def test_eval_pipeline_end_to_end():
    entities = [f"/m/e{i}" for i in range(10)]
    true_tail = "/m/e0"
    h, r = "/m/h1", "/r/rel"
    known_triples: set[tuple[str, str, str]] = {(h, r, "/m/e1"), (h, r, "/m/e2")}

    # Step 1: generate candidates
    candidates = generate_tail_candidates(
        h, r, true_tail, entities, num_candidates=6, seed=42
    )
    assert true_tail in candidates

    # Step 2: filter
    filtered = filter_candidates_tail(h, r, candidates, true_tail, known_triples)
    assert "/m/e1" not in filtered
    assert "/m/e2" not in filtered
    assert true_tail in filtered

    # Step 3: rank with a simple score fn
    def score_fn(hh: str, rr: str, tt: str) -> float:
        return 1.0 if tt == true_tail else 0.0

    scored = [(t, score_fn(h, r, t)) for t in filtered]
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [t for t, _ in scored]
    true_rank = ranked.index(true_tail) + 1

    result = RankingResult(
        query=(h, r, true_tail),
        true_rank=true_rank,
        num_candidates=len(filtered),
        scored_candidates=scored,
    )

    # Step 4: compute metrics
    metrics = compute_all_metrics([result], ks=(1, 3, 10))
    assert metrics["MRR"] == pytest.approx(1.0)
    assert metrics["Hits@1"] == pytest.approx(1.0)
    assert metrics["num_queries"] == 1
