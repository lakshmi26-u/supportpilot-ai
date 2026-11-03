
import streamlit as st
import sys
import os
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Load Gemini API key from Streamlit Cloud secrets
if "GEMINI_API_KEY" not in os.environ:
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

import src.backend as backend

st.set_page_config(
    page_title="SupportPilot AI",
    page_icon="🤖",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #eef2ff;
}
.status-card {
    padding: 16px;
    border-radius: 12px;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🤖 SupportPilot AI")
st.sidebar.caption("AI Customer Support & Operations")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Chat Assistant",
        "Order Lookup",
        "Return Management",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.success("AI Backend Connected")
st.sidebar.caption("Powered by Gemini + RAG")

# ---------- Helper ----------
def show_result(result):
    if isinstance(result, dict):
        st.json(result)
    else:
        st.write(result)


# ---------- Chat Assistant ----------
if page == "Chat Assistant":

    st.title("💬 Customer Support Assistant")
    st.caption("Ask questions about orders, shipping, returns, payments and products.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("How can I help you today?")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    result = backend.route_customer_message(user_input)

                    if result.get("status") == "success":
                        answer = result.get(
                            "response",
                            "Your request was processed successfully."
                        )
                    elif result.get("status") == "escalate":
                        answer = result.get(
                            "message",
                            "Your request needs human support."
                        )
                    else:
                        answer = result.get(
                            "message",
                            "Something went wrong. Please try again."
                        )

                except Exception:
                    answer = (
                        "Sorry, I couldn't process your request right now. "
                        "Please contact human support."
                    )

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------- Order Lookup ----------
elif page == "Order Lookup":

    st.title("📦 Order Lookup")
    st.caption("Search the sample order database.")

    order_id = st.text_input(
        "Enter Order ID",
        placeholder="Example: ORD1001"
    )

    if st.button("Search Order", type="primary"):

        if not order_id.strip():
            st.warning("Please enter an order ID.")

        else:
            with st.spinner("Searching order..."):
                try:
                    result = backend.lookup_order(order_id.strip())
                    show_result(result)
                except Exception:
                    st.error("Unable to retrieve order details.")


# ---------- Return Management ----------
elif page == "Return Management":

    st.title("↩️ Return Management")
    st.caption("Submit and review simulated return requests.")

    st.subheader("Submit a return request")

    return_order_id = st.text_input(
        "Order ID",
        placeholder="Example: ORD1002",
        key="return_order_id"
    )

    if st.button("Submit for Human Review", type="primary"):

        if not return_order_id.strip():
            st.warning("Please enter an order ID.")

        else:
            try:
                result = backend.submit_return_for_review(
                    return_order_id.strip()
                )
                show_result(result)
            except Exception:
                st.error("Unable to submit the return request.")

    st.divider()
    st.subheader("Review pending requests")

    requests = backend.return_requests

    if not requests:
        st.info("No return requests have been submitted in this session.")

    else:
        for order_id, request in requests.items():

            with st.expander(f"Return request: {order_id}"):

                show_result(request)

                decision = st.selectbox(
                    "Decision",
                    ["approved", "rejected"],
                    key=f"decision_{order_id}"
                )

                if st.button(
                    "Submit Review",
                    key=f"review_{order_id}"
                ):
                    try:
                        result = backend.review_return_request(
                            order_id,
                            decision,
                            "Support Agent 01"
                        )
                        show_result(result)
                        st.rerun()
                    except Exception:
                        st.error("Unable to review this request.")


# ---------- About Project ----------
elif page == "About Project":

    st.title("ℹ️ About SupportPilot AI")

    st.markdown("""
    ### AI Customer Support and Operations Agent

    An AI-powered customer support application that combines
    Retrieval-Augmented Generation (RAG), message classification,
    order lookup and human approval workflows.

    **Core capabilities**

    - FAQ retrieval using semantic search
    - Customer message classification
    - Mock order information retrieval
    - Gemini-powered response generation
    - Human escalation when evidence is insufficient
    - Simulated return approval workflow

    **Technology stack**

    - Python
    - Streamlit
    - Google Gemini API
    - ChromaDB
    - Sentence Transformers
    - Pandas

    **Safety design**

    - Uses mock order data
    - Does not issue actual refunds
    - Return decisions are simulated
    - Uncertain questions are escalated to human support
    """)

    st.info(
        "Portfolio demonstration project. "
        "Not connected to a real company's order or payment system."
    )
