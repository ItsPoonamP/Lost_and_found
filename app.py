"""
app.py — Streamlit query interface.

Run this in Terminal 2 (pipeline can run simultaneously):
    streamlit run app.py

Two tabs:
    🔍 Find Object   — upload a photo → CLIP search → show last known location
    📋 Recent        — browse all observations, filter by class
"""

import streamlit as st
from PIL import Image
import os
import config

from core.embedder import Embedder
from core.database import Database
from core.search   import VectorSearch


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lost & Found",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .result-card {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .badge {
        background: #0f3460;
        color: #e2e2e2;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.82rem;
        margin-right: 6px;
    }
    .score-high { color: #00e676; font-weight: bold; }
    .score-mid  { color: #ffb300; font-weight: bold; }
    .score-low  { color: #ef5350; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource
def load_components():
    embedder = Embedder()
    db       = Database()
    search   = VectorSearch()
    return embedder, db, search


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 0.80:  return "score-high"
    if score >= 0.60:  return "score-mid"
    return "score-low"


def render_result(obs: dict, score: float, rank: int):
    pct  = score * 100
    css  = score_class(score)
    name = obs["class_name"]
    ts   = obs["timestamp"]
    tid  = obs["track_id"]
    conf = obs["confidence"]

    st.markdown(f"""
    <div class="result-card">
        <span class="badge">#{rank}</span>
        <span class="badge">{name}</span>
        <span class="badge">Track ID {tid}</span>
        <span class="{css}" style="float:right">{pct:.1f}% match</span>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_info = st.columns([1, 1])

    with col_img:
        fp = obs.get("frame_path", "")
        if fp and os.path.exists(fp):
            st.image(fp, caption=f"Frame captured at {ts}", use_container_width=True)
        else:
            st.warning("Frame image not found on disk.")

    with col_info:
        st.markdown(f"**Last seen:** {ts}")
        st.markdown(f"**Object class:** {name}")
        st.markdown(f"**Track ID:** `{tid}`")
        st.markdown(f"**Detection confidence:** {conf:.2%}")
        st.markdown(f"**Visual similarity:** {pct:.1f}%")
        st.progress(min(score, 1.0))

    st.divider()


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    embedder, db, search = load_components()

    # ── Sidebar stats ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🔍 Lost & Found")
        st.caption("AI-powered object retrieval from CCTV footage")
        st.divider()

        if st.button("↻ Refresh stats"):
            st.cache_resource.clear()
            st.rerun()

        stats = db.stats()
        st.metric("Observations in DB",  stats["total"])
        st.metric("FAISS vectors",        search.total)

        if stats["classes"]:
            st.subheader("Objects tracked")
            for cls in stats["classes"]:
                st.write(f"• **{cls['name']}** — {cls['count']}")
        else:
            st.info("No data yet. Run `python pipeline.py` first.")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_find, tab_recent = st.tabs(["🔍 Find Object", "📋 Recent Sightings"])

    # ════════════════════════════════════════════════════════════════════════════
    with tab_find:
        st.header("Find my lost object")
        st.write("Upload any photo of the object — CLIP will find visually similar frames.")

        left, right = st.columns([1, 1.5], gap="large")

        with left:
            uploaded = st.file_uploader(
                "Upload a photo of your object",
                type=["jpg", "jpeg", "png", "webp"],
            )
            top_k = st.slider("Max results to show", 1, 10, config.TOP_K)

            # Optional text query
            with st.expander("Or describe the object (text search)"):
                text_query = st.text_input(
                    "Describe it",
                    placeholder="e.g.  red backpack with black straps",
                )

            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

            if uploaded:
                query_image = Image.open(uploaded).convert("RGB")
                st.image(query_image, caption="Your query image", use_container_width=True)

        with right:
            if search_btn:
                if search.total == 0:
                    st.error("Database is empty — start the pipeline first.")
                    st.code("python pipeline.py", language="bash")
                    st.stop()

                # Determine query embedding
                if text_query.strip():
                    with st.spinner("Encoding text query…"):
                        query_emb = embedder.encode_text(text_query.strip())
                    st.info(f"Text query: *{text_query}*")
                elif uploaded:
                    with st.spinner("Encoding image…"):
                        query_emb = embedder.encode_image(query_image)
                else:
                    st.warning("Please upload an image or enter a text description.")
                    st.stop()

                with st.spinner("Searching FAISS index…"):
                    results = search.search(query_emb, top_k=top_k)

                if not results:
                    st.error("No matches found in the database.")
                else:
                    obs_ids = [obs_id for obs_id, _ in results]
                    scores  = {obs_id: score for obs_id, score in results}

                    observations = db.get_by_ids(obs_ids)
                    # build lookup by id for correct score assignment
                    obs_map = {o["id"]: o for o in observations}

                    st.success(f"Found **{len(results)}** match(es)")

                    for rank, (obs_id, score) in enumerate(results, start=1):
                        obs = obs_map.get(obs_id)
                        if obs:
                            render_result(obs, score, rank)

    # ════════════════════════════════════════════════════════════════════════════
    with tab_recent:
        st.header("Recent Sightings")

        class_names = ["All"] + [v for v in config.TRACKED_CLASSES.values()]
        col_filter, col_btn = st.columns([3, 1])
        with col_filter:
            class_filter = st.selectbox("Filter by object class", class_names)
        with col_btn:
            st.write("")  # spacer
            load_btn = st.button("Load", type="primary", use_container_width=True)

        if load_btn:
            cn = None if class_filter == "All" else class_filter
            recent = db.get_recent(class_name=cn, limit=30)

            if not recent:
                st.info("No observations yet.")
            else:
                st.write(f"Showing {len(recent)} most recent observations")
                for obs in recent:
                    with st.expander(
                        f"**{obs['class_name']}** | Track #{obs['track_id']} | {obs['timestamp']}"
                    ):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if os.path.exists(obs.get("frame_path", "")):
                                st.image(obs["frame_path"], use_container_width=True)
                        with c2:
                            st.write(f"**Class:** {obs['class_name']}")
                            st.write(f"**Track ID:** `{obs['track_id']}`")
                            st.write(f"**Time:** {obs['timestamp']}")
                            st.write(f"**Confidence:** {obs['confidence']:.2%}")
                            x1, y1, x2, y2 = obs["x1"], obs["y1"], obs["x2"], obs["y2"]
                            st.write(f"**BBox:** ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")


if __name__ == "__main__":
    main()
