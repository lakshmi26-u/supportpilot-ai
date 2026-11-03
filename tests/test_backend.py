import numpy as np

from src import backend


def test_lookup_existing_order():
    result = backend.lookup_order("ORD1001")
    assert result["status"] == "success"
    assert result["order"]["order_id"] == "ORD1001"


def test_lookup_lowercase_order():
    result = backend.lookup_order("ord1001")
    assert result["status"] == "success"


def test_lookup_unknown_order():
    result = backend.lookup_order("ORD9999")
    assert result["status"] == "not_found"


def test_lookup_invalid_format():
    result = backend.lookup_order("ABC1234")
    assert result["status"] == "invalid_input"


def test_lookup_empty_input():
    result = backend.lookup_order("")
    assert result["status"] == "invalid_input"


def test_create_return_for_eligible_order():
    result = backend.create_return_request("ORD1002")
    assert result["status"] == "pending_human_approval"
    assert result["approved"] is False


def test_create_return_for_undelivered_order():
    result = backend.create_return_request("ORD1001")
    assert result["status"] == "not_eligible"


def test_create_return_for_ineligible_order():
    result = backend.create_return_request("ORD1005")
    assert result["status"] == "not_eligible"


def test_create_return_for_unknown_order():
    result = backend.create_return_request("ORD9999")
    assert result["status"] == "not_found"


import pytest


@pytest.fixture
def fresh_return_request():
    order_id = "ORD1002"
    original = backend.return_requests.get(order_id)

    backend.return_requests[order_id] = {
        "order_id": order_id,
        "status": "pending",
        "reviewer": None
    }

    yield order_id

    if original is None:
        backend.return_requests.pop(order_id, None)
    else:
        backend.return_requests[order_id] = original

    backend.save_return_requests()


def test_approve_return_request(fresh_return_request):
    result = backend.review_return_request(
        fresh_return_request, "approved", "Test Reviewer"
    )
    assert result["status"] == "approved"
    assert result["reviewer"] == "Test Reviewer"


def test_reject_return_request(fresh_return_request):
    result = backend.review_return_request(
        fresh_return_request, "rejected", "Test Reviewer"
    )
    assert result["status"] == "rejected"


def test_invalid_review_decision(fresh_return_request):
    result = backend.review_return_request(
        fresh_return_request, "cancelled", "Test Reviewer"
    )
    assert result["status"] == "invalid_decision"


def test_missing_reviewer(fresh_return_request):
    result = backend.review_return_request(
        fresh_return_request, "approved", ""
    )
    assert result["status"] == "invalid_input"


def test_review_already_reviewed_request(fresh_return_request):
    backend.review_return_request(
        fresh_return_request, "approved", "Test Reviewer"
    )

    result = backend.review_return_request(
        fresh_return_request, "rejected", "Another Reviewer"
    )
    assert result["status"] == "already_reviewed"


def test_review_nonexistent_request():
    result = backend.review_return_request(
        "ORD9999", "approved", "Test Reviewer"
    )
    assert result["status"] == "not_found"


def test_classify_shipping_message(monkeypatch):
    class FakeResponse:
        text = "shipping"

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(backend, "client", FakeClient())

    result = backend.classify_message("Where is my delivery?")

    assert result == "shipping"


def test_classify_invalid_gemini_category(monkeypatch):
    class FakeResponse:
        text = "complaint"

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(backend, "client", FakeClient())

    with pytest.raises(ValueError):
        backend.classify_message("Where is my delivery?")


def test_classify_empty_message():
    with pytest.raises(ValueError):
        backend.classify_message("")


def test_search_faq_returns_matching_result(monkeypatch):
    class FakeModel:
        def encode(self, query, normalize_embeddings=True):
            return np.array([0.1, 0.2])

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            return {
                "ids": [["FAQ003"]],
                "metadatas": [[{
                    "question": "How long does shipping take?",
                    "answer": "Standard shipping takes 3-5 business days.",
                    "category": "shipping",
                    "source": "FAQ"
                }]],
                "distances": [[0.25]],
                "documents": [["Shipping information"]]
            }

    monkeypatch.setattr(backend, "model", FakeModel())
    monkeypatch.setattr(backend, "collection", FakeCollection())

    result = backend.search_faq("Where is my delivery?")

    assert len(result) == 1
    assert result[0]["faq_id"] == "FAQ003"
    assert result[0]["category"] == "shipping"


def test_search_faq_filters_irrelevant_result(monkeypatch):
    class FakeModel:
        def encode(self, query, normalize_embeddings=True):
            return np.array([0.1, 0.2])

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            return {
                "ids": [["FAQ003"]],
                "metadatas": [[{
                    "question": "How long does shipping take?",
                    "answer": "Standard shipping takes 3-5 business days.",
                    "category": "shipping",
                    "source": "FAQ"
                }]],
                "distances": [[0.90]],
                "documents": [["Shipping information"]]
            }

    monkeypatch.setattr(backend, "model", FakeModel())
    monkeypatch.setattr(backend, "collection", FakeCollection())

    result = backend.search_faq("Tell me about an unrelated topic")

    assert result == []


def test_search_faq_empty_query():
    with pytest.raises(ValueError):
        backend.search_faq("")



def test_process_customer_message_with_evidence(monkeypatch):
    monkeypatch.setattr(
        backend, "classify_message",
        lambda message: "shipping"
    )

    fake_faqs = [{
        "faq_id": "FAQ003",
        "question": "How long does shipping take?",
        "answer": "3-5 business days",
        "category": "shipping",
        "source": "FAQ",
        "distance": 0.25
    }]

    monkeypatch.setattr(
        backend, "search_faq",
        lambda message: fake_faqs
    )

    result = backend.process_customer_message(
        "Where is my delivery?"
    )

    assert result["category"] == "shipping"
    assert result["status"] == "evidence_found"
    assert result["faqs"] == fake_faqs


def test_process_customer_message_without_evidence(monkeypatch):
    monkeypatch.setattr(
        backend, "classify_message",
        lambda message: "shipping"
    )

    monkeypatch.setattr(
        backend, "search_faq",
        lambda message: []
    )

    result = backend.process_customer_message(
        "Where is my delivery?"
    )

    assert result["category"] == "shipping"
    assert result["status"] == "escalate"
    assert result["faqs"] == []
    assert "Insufficient FAQ evidence" in result["message"]



def test_route_customer_message_invalid_input(monkeypatch):
    logged_events = []

    monkeypatch.setattr(
        backend,
        "log_customer_event",
        lambda category, status: logged_events.append((category, status))
    )

    result = backend.route_customer_message("")

    assert result["status"] == "invalid_input"
    assert result["message"] == "Please enter your question."
    assert ("unknown", "invalid_input") in logged_events


def test_route_customer_message_escalates_without_faq(monkeypatch):
    logged_events = []

    monkeypatch.setattr(
        backend, "extract_order_id",
        lambda message: None
    )

    monkeypatch.setattr(
        backend, "process_customer_message",
        lambda message: {
            "category": "shipping",
            "status": "escalate",
            "message": "Insufficient FAQ evidence.",
            "faqs": []
        }
    )

    monkeypatch.setattr(
        backend,
        "log_customer_event",
        lambda category, status: logged_events.append((category, status))
    )

    result = backend.route_customer_message(
        "Tell me something not covered by the FAQs"
    )

    assert result["status"] == "escalate"
    assert result["category"] == "shipping"
    assert ("shipping", "escalate") in logged_events


def test_route_customer_message_generates_ai_response(monkeypatch):
    logged_events = []

    monkeypatch.setattr(
        backend, "extract_order_id",
        lambda message: None
    )

    monkeypatch.setattr(
        backend, "process_customer_message",
        lambda message: {
            "category": "shipping",
            "status": "evidence_found",
            "faqs": [{"answer": "Shipping takes 3-5 business days."}]
        }
    )

    monkeypatch.setattr(
        backend, "generate_support_response",
        lambda message, faqs: {
            "status": "response_generated",
            "response": "Your order usually ships within 3-5 business days."
        }
    )

    monkeypatch.setattr(
        backend,
        "log_customer_event",
        lambda category, status: logged_events.append((category, status))
    )

    result = backend.route_customer_message(
        "How long does shipping take?"
    )

    assert result["category"] == "shipping"
    assert result["status"] == "success"
    assert "3-5 business days" in result["response"]
    assert ("shipping", "success") in logged_events



def test_route_customer_message_with_order_id(monkeypatch):
    logged_events = []

    monkeypatch.setattr(
        backend, "extract_order_id",
        lambda message: "ORD1001"
    )

    monkeypatch.setattr(
        backend, "process_order_message",
        lambda message: {
            "status": "success",
            "order_id": "ORD1001",
            "order_status": "Shipped",
            "shipping_status": "In Transit",
            "estimated_delivery": "2026-11-04"
        }
    )

    monkeypatch.setattr(
        backend, "generate_order_response",
        lambda message, result: "Your order is in transit."
    )

    monkeypatch.setattr(
        backend,
        "log_customer_event",
        lambda category, status: logged_events.append((category, status))
    )

    result = backend.route_customer_message(
        "Where is my order ORD1001?"
    )

    assert result["category"] == "order"
    assert result["status"] == "success"
    assert result["response"] == "Your order is in transit."
    assert ("order", "success") in logged_events


def test_route_customer_message_when_order_fails(monkeypatch):
    logged_events = []

    monkeypatch.setattr(
        backend, "extract_order_id",
        lambda message: "ORD9999"
    )

    monkeypatch.setattr(
        backend, "process_order_message",
        lambda message: {
            "status": "order_not_found",
            "message": "Order not found."
        }
    )

    monkeypatch.setattr(
        backend,
        "log_customer_event",
        lambda category, status: logged_events.append((category, status))
    )

    result = backend.route_customer_message(
        "Where is my order ORD9999?"
    )

    assert result["status"] == "order_not_found"
    assert ("order", "order_not_found") in logged_events


def test_generate_support_response_when_gemini_returns_empty(monkeypatch):
    class FakeResponse:
        text = "   "

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(backend, "client", FakeClient())

    faqs = [{
        "question": "How long does shipping take?",
        "answer": "3-5 business days"
    }]

    result = backend.generate_support_response(
        "How long does shipping take?",
        faqs
    )

    assert result["status"] == "escalate"
    assert "human support" in result["response"]


def test_generate_support_response_when_gemini_fails(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise RuntimeError("Simulated API failure")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(backend, "client", FakeClient())

    faqs = [{
        "question": "How long does shipping take?",
        "answer": "3-5 business days"
    }]

    result = backend.generate_support_response(
        "How long does shipping take?",
        faqs
    )

    assert result["status"] == "escalate"
    assert "temporarily unavailable" in result["response"]




def test_duplicate_return_with_different_order_id_case():
    order_id = "ORD1002"

    original_request = backend.return_requests.get(order_id)

    backend.return_requests[order_id] = {
        "order_id": order_id,
        "status": "pending",
        "reviewer": None
    }

    try:
        result = backend.submit_return_for_review("ord1002")

        assert result["status"] == "already_submitted"

    finally:
        if original_request is None:
            backend.return_requests.pop(order_id, None)
        else:
            backend.return_requests[order_id] = original_request

        backend.save_return_requests()


def test_review_return_request_lowercase_order_id(monkeypatch):
    monkeypatch.setitem(
        backend.return_requests,
        "ORD1002",
        {
            "order_id": "ORD1002",
            "status": "pending",
            "reviewer": None
        }
    )

    monkeypatch.setattr(backend, "save_return_requests", lambda: None)

    result = backend.review_return_request(
        "ord1002",
        "approved",
        "Test Reviewer"
    )

    assert result["status"] == "approved"
    assert result["order_id"] == "ORD1002"
    assert backend.return_requests["ORD1002"]["reviewer"] == "Test Reviewer"
