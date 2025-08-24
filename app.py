import sys
import time
import uuid
import logging
import streamlit as st
from main import ChatBot

# ---------------- Page config (keep FIRST) ----------------
st.set_page_config(page_title="Random Fortune Telling Bot", layout="centered")

# ---------------- Logging ----------------
logger = logging.getLogger("fortune_app")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # use DEBUG for more detail
logger.info("Streamlit app started")


# ---------------- Cache the bot ----------------
@st.cache_resource(show_spinner="Warming up the oracle (first run may take a bit)…")
def get_bot():
    logger.info("Cache miss: creating ChatBot()")
    t0 = time.perf_counter()
    bot = ChatBot()  # will raise if env vars/files are missing
    logger.info("ChatBot() created in %.2fs", time.perf_counter() - t0)
    return bot


# ---------------- One-time init with status ----------------
if "init_done" not in st.session_state:
    with st.status("Initializing bot…", expanded=True) as status:
        try:
            bot = get_bot()
            st.session_state["bot"] = bot
            # persistent session id for in-memory chat history
            st.session_state["session_id"] = f"ui-{uuid.uuid4().hex[:8]}"
            status.update(label="Bot ready", state="complete")
            logger.info("Bot ready")
        except Exception as e:
            logger.exception("Bot initialization failed")
            status.update(label="Initialization failed", state="error")
            st.error(f"Bot initialization failed: {e}")
            st.stop()
    st.session_state["init_done"] = True
else:
    bot = st.session_state.get("bot")
    if bot is None:
        logger.error("Bot missing from session_state; reinitializing.")
        bot = get_bot()
        st.session_state["bot"] = bot
    logger.info("Bot fetched from cache (fast path)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Random Fortune Telling Bot")
    st.markdown("**Status:** ✅ Ready" if bot else "**Status:** ❌ Not ready")
    st.caption("Memory-enabled • Tools: lucky number, now")
    if "session_id" in st.session_state:
        st.caption(f"Session: `{st.session_state['session_id']}`")
    st.caption(
        "Tip: run with a free port, e.g. `streamlit run streamlit.py --server.port 8507`"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()
    with col2:
        if st.button("♻️ Reset session"):
            # New session_id → fresh memory thread
            st.session_state["session_id"] = f"ui-{uuid.uuid4().hex[:8]}"
            st.session_state.pop("messages", None)
            st.rerun()


# ---------------- Helper ----------------
def generate_response(user_text: str) -> str:
    """Call the LangGraph + memory pipeline exposed by ChatBot."""
    global bot
    # Check if bot is None
    if bot is None:
        try:
            logger.warning("Global bot is None, reinitializing...")
            bot = get_bot()
            st.session_state["bot"] = bot
        except Exception as e:
            logger.exception("Failed to initialize bot")
            return f"Failed to initialize the bot: {e}"
            
    session_id = st.session_state.get("session_id", "streamlit-user")
    logger.info(
        "Inference start | session=%s | prompt_len=%d", session_id, len(user_text)
    )
    t0 = time.perf_counter()
    try:
        reply = bot.answer_with_graph(user_text, session_id=session_id)
        logger.info(
            "Inference success in %.2fs | reply_len=%s",
            time.perf_counter() - t0,
            (len(reply) if isinstance(reply, str) else "n/a"),
        )
        return reply
    except Exception as e:
        logger.exception("Inference failed after %.2fs", time.perf_counter() - t0)
        return f"Oops, something went wrong: {e}"


# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome, let's unveil your future."}
    ]
    logger.info("Initialized chat history in session_state")

# ---------------- Display history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- Input + respond ----------------
user_text = st.chat_input("Ask your question…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the stars…"):
            reply = generate_response(user_text)
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
