# tests/test_main.py
# Tests for all endpoints
# Run: pytest tests/ -v

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as app_module
from main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# GET / — Root endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_returns_200(self):
        """Root endpoint should return 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_required_keys(self):
        """Response should contain message, docs, ask_endpoint"""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "ask_endpoint" in data

    def test_root_message_content(self):
        """Message should contain the word 'running'"""
        response = client.get("/")
        assert "running" in response.json()["message"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# GET /health — Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self):
        """Status should be 'ok'"""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_has_llm_enabled_key(self):
        """Response should contain llm_enabled key"""
        response = client.get("/health")
        assert "llm_enabled" in response.json()

    def test_health_llm_enabled_is_boolean(self):
        """llm_enabled value should be a boolean"""
        response = client.get("/health")
        assert isinstance(response.json()["llm_enabled"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# GET /llm-status — LLM status check
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMStatusEndpoint:

    def test_llm_status_returns_200(self):
        """LLM status endpoint should return 200"""
        response = client.get("/llm-status")
        assert response.status_code == 200

    def test_llm_status_has_required_keys(self):
        """Response should contain llm_enabled, status, and message"""
        response = client.get("/llm-status")
        data = response.json()
        assert "llm_enabled" in data
        assert "status" in data
        assert "message" in data

    def test_llm_status_value_matches_health(self):
        """llm_enabled should be the same in both /llm-status and /health"""
        status_response = client.get("/llm-status")
        health_response = client.get("/health")
        assert status_response.json()["llm_enabled"] == health_response.json()["llm_enabled"]

    def test_llm_status_active_when_enabled(self):
        """Status should be 'active' when LLM is enabled"""
        app_module.USE_LLM = True
        response = client.get("/llm-status")
        assert response.json()["status"] == "active"

    def test_llm_status_inactive_when_disabled(self):
        """Status should be 'inactive' when LLM is disabled"""
        app_module.USE_LLM = False
        response = client.get("/llm-status")
        assert response.json()["status"] == "inactive"
        # Reset
        app_module.USE_LLM = True


# ─────────────────────────────────────────────────────────────────────────────
# POST /llm-toggle — LLM enable/disable
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMToggleEndpoint:

    def test_toggle_returns_200(self):
        """Toggle endpoint should return 200"""
        response = client.post("/llm-toggle", json={"enabled": True})
        assert response.status_code == 200

    def test_toggle_enable_llm(self):
        """current_status should be True when LLM is enabled"""
        response = client.post("/llm-toggle", json={"enabled": True})
        assert response.json()["current_status"] is True

    def test_toggle_disable_llm(self):
        """current_status should be False when LLM is disabled"""
        response = client.post("/llm-toggle", json={"enabled": False})
        assert response.json()["current_status"] is False

    def test_toggle_shows_previous_status(self):
        """Response should contain the previous_status before toggle"""
        app_module.USE_LLM = True
        response = client.post("/llm-toggle", json={"enabled": False})
        assert response.json()["previous_status"] is True

    def test_toggle_has_message(self):
        """Response should contain a message field"""
        response = client.post("/llm-toggle", json={"enabled": True})
        assert "message" in response.json()

    def test_toggle_actually_changes_status(self):
        """Status change should be reflected in /llm-status after toggle"""
        client.post("/llm-toggle", json={"enabled": False})
        status = client.get("/llm-status").json()
        assert status["llm_enabled"] is False
        # Reset
        client.post("/llm-toggle", json={"enabled": True})

    def test_toggle_missing_body_returns_422(self):
        """Request without body should return 422"""
        response = client.post("/llm-toggle", json={})
        assert response.status_code == 422

    def test_toggle_invalid_type_returns_422(self):
        """Passing a non-boolean string for enabled should return 422"""
        response = client.post("/llm-toggle", json={"enabled": "yes"})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /ask — Main FAQ endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestAskEndpoint:

    def setup_method(self):
        """Disable LLM before each test for consistent results"""
        app_module.USE_LLM = False

    def teardown_method(self):
        """Reset LLM state after each test"""
        app_module.USE_LLM = False

    # ── Response structure tests ──

    def test_ask_returns_200(self):
        """Ask endpoint should return 200"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        assert response.status_code == 200

    def test_ask_response_has_required_keys(self):
        """Response should contain answer, confidence, source, and fallback"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "source" in data
        assert "fallback" in data

    def test_ask_confidence_is_float(self):
        """Confidence should be a float"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        assert isinstance(response.json()["confidence"], float)

    def test_ask_source_is_list(self):
        """Source should be a list"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        assert isinstance(response.json()["source"], list)

    def test_ask_fallback_is_boolean(self):
        """Fallback should be a boolean"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        assert isinstance(response.json()["fallback"], bool)

    def test_ask_no_internal_keys_in_response(self):
        """Internal key _retrieved_faq should not be present in response"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        assert "_retrieved_faq" not in response.json()

    # ── Match tests ──

    def test_ask_exact_match_query(self):
        """Exact FAQ question should return fallback as False"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        data = response.json()
        assert data["fallback"] is False
        assert len(data["source"]) > 0

    def test_ask_similar_query_matches(self):
        """A query with similar meaning should also find a match"""
        response = client.post("/ask", json={"query": "I forgot my password, what do I do?"})
        data = response.json()
        assert data["fallback"] is False

    def test_ask_refund_query(self):
        """Refund query should return source id '2'"""
        response = client.post("/ask", json={"query": "What is the refund policy?"})
        data = response.json()
        assert data["fallback"] is False
        assert "2" in data["source"]

    def test_ask_confidence_between_0_and_1(self):
        """Confidence score should be between 0.0 and 1.0"""
        response = client.post("/ask", json={"query": "How to reset password?"})
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    # ── Fallback tests ──

    def test_ask_out_of_scope_returns_fallback(self):
        """Query outside FAQ scope should return fallback True and empty source"""
        response = client.post("/ask", json={"query": "What is the weather today in Tokyo?"})
        data = response.json()
        assert data["fallback"] is True
        assert data["source"] == []

    def test_ask_fallback_answer_text(self):
        """Fallback response should contain the standard no-information message"""
        response = client.post("/ask", json={"query": "Who is the president of France?"})
        data = response.json()
        if data["fallback"]:
            assert "don't have enough information" in data["answer"].lower()

    def test_ask_fallback_low_confidence(self):
        """Fallback response should have a low confidence score"""
        response = client.post("/ask", json={"query": "What is the weather today in Tokyo?"})
        data = response.json()
        if data["fallback"]:
            assert data["confidence"] < 0.25

    # ── Validation tests ──

    def test_ask_empty_query_returns_422(self):
        """Empty string query should return 422"""
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 422

    def test_ask_missing_query_returns_422(self):
        """Missing query field should return 422"""
        response = client.post("/ask", json={})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /faqs — All FAQs list
# ─────────────────────────────────────────────────────────────────────────────

class TestFAQsEndpoint:

    def test_faqs_returns_200(self):
        """FAQs endpoint should return 200"""
        response = client.get("/faqs")
        assert response.status_code == 200

    def test_faqs_has_total_key(self):
        """Response should contain total count"""
        response = client.get("/faqs")
        assert "total" in response.json()

    def test_faqs_has_faqs_list(self):
        """Response should contain a faqs list"""
        response = client.get("/faqs")
        assert "faqs" in response.json()
        assert isinstance(response.json()["faqs"], list)

    def test_faqs_total_matches_list_length(self):
        """Total count should match the actual length of the faqs list"""
        response = client.get("/faqs")
        data = response.json()
        assert data["total"] == len(data["faqs"])

    def test_faqs_each_item_has_required_keys(self):
        """Each FAQ item should contain id, question, and answer"""
        response = client.get("/faqs")
        for faq in response.json()["faqs"]:
            assert "id" in faq
            assert "question" in faq
            assert "answer" in faq

    def test_faqs_not_empty(self):
        """FAQ list should not be empty"""
        response = client.get("/faqs")
        assert response.json()["total"] > 0