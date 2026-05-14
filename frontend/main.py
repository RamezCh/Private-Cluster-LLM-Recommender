"""Streamlit frontend. Clean, fast chat interface."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from backend.services.parser import parse_hardware_input, get_available_gpu_options, ParsedHardware
from backend.services.recommender import get_recommender, ScoredModel


GREETING = """
👋 **LLM Recommender**

Find the best locally-hostable open-weight LLM for your hardware and use case.

1. Select your GPU(s)
2. Describe your use case
3. Get ranked recommendations
"""


def init_state():
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "last_hw" not in st.session_state:
        st.session_state.last_hw = None
    if "last_uc" not in st.session_state:
        st.session_state.last_uc = ""


def render_card(model: ScoredModel, rank: int):
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader(f"#{rank}: {model.model_id.split('/')[-1]}")
            st.caption(f"`{model.model_id}`")
        with c2:
            st.metric("Score", f"{int(model.final_score * 100)}%")

        with st.expander("Details", expanded=rank <= 2):
            a, b, c = st.columns(3)
            with a:
                st.markdown("**Benchmarks**")
                st.write(f"- Coding: {model.coding:.1f}")
                st.write(f"- Math: {model.math_score:.1f}")
                st.write(f"- Reasoning: {model.reasoning:.1f}")
                st.write(f"- Intelligence: {model.intelligence_index:.1f}")
            with b:
                st.markdown("**Model**")
                st.write(f"- Params: {model.params_billions:.1f}B")
                st.write(f"- Size: {model.vram_fp16:.1f} GB")
                st.write(f"- Type: {'MoE' if model.is_moe else 'Dense'}")
                st.write(f"- Strategy: {model.hosting_strategy}")
            with c:
                st.markdown("**Hardware Fit**")
                hw = model.matched_hardware
                st.write(f"- Status: {hw.get('status', 'N/A')}")
                st.write(f"- Quant: {hw.get('quantization', 'N/A')}")
                st.write(f"- Parallelism: {hw.get('parallelism', 'N/A')}")

            if model.hf_repo_id:
                st.markdown(f"[Hugging Face](https://huggingface.co/{model.hf_repo_id})")

        st.divider()


def render_results(recs: list[ScoredModel], hw: ParsedHardware, uc: str):
    st.subheader(f"Top {len(recs)} Recommendations")
    if recs:
        st.success(f"Found {len(recs)} models for {hw.count}x {hw.gpu_name} — **{uc}**")
        for i, m in enumerate(recs, 1):
            render_card(m, i)
    else:
        st.warning("No models found.")


def main():
    init_state()
    st.set_page_config(page_title="LLM Recommender", page_icon="🤖", layout="wide")
    st.title("🤖 LLM Recommender")
    st.markdown("Find the best open-weight LLMs for your hardware")

    with st.chat_message("assistant"):
        st.markdown(GREETING)

    st.divider()

    with st.form("rec_form", clear_on_submit=False):
        st.subheader("Hardware")
        gpu_opts = get_available_gpu_options()
        gpu_names = [n for n, _ in gpu_opts]
        sel_name = st.selectbox("GPU Type", options=gpu_names,
                                index=gpu_names.index("A100 80GB") if "A100 80GB" in gpu_names else 0)
        gpu_count = st.number_input("Number of GPUs", min_value=1, max_value=16, value=1, step=1)

        sel_id = None
        for n, gid in gpu_opts:
            if n == sel_name:
                sel_id = gid
                break

        from config.config import GPU_CATALOG
        if sel_id:
            cfg = GPU_CATALOG.get(sel_id)
            if cfg:
                st.caption(f"**{cfg.vram_gb} GB/GPU** | Total: **{cfg.vram_gb * gpu_count} GB** | "
                           f"Tier: {cfg.tier.replace('_', ' ').title()}")

        st.subheader("Use Case")
        use_case = st.text_area(
            "Describe your use case",
            placeholder="e.g., Code generation and debugging, Mathematical reasoning...",
            height=100,
        )

        c1, c2, c3 = st.columns([1, 1, 4])
        submitted = st.form_submit_button("🔍 Get Recommendations", type="primary", use_container_width=True)
        cleared = st.form_submit_button("🗑️ Clear", use_container_width=True)

    if cleared:
        st.session_state.recommendations = []
        st.session_state.last_hw = None
        st.session_state.last_uc = ""
        st.rerun()

    hw = None
    if sel_id:
        cfg = GPU_CATALOG.get(sel_id)
        if cfg:
            hw = ParsedHardware(sel_id, sel_name, cfg.vram_gb, gpu_count,
                                cfg.vram_gb * gpu_count, cfg.tier)

    if submitted and hw and use_case:
        with st.spinner("Finding best models..."):
            try:
                start = time.time()
                recs = get_recommender().recommend(
                    hardware=hw,
                    use_case_text=use_case,
                    user_query=use_case,
                    top_k=5,
                )
                elapsed = time.time() - start

                st.session_state.recommendations = recs
                st.session_state.last_hw = hw
                st.session_state.last_uc = use_case

                st.info(f"Completed in {elapsed:.2f}s — searching from "
                        f"{get_recommender().model_count:,} models")
                render_results(recs, hw, use_case)

            except Exception as e:
                import traceback
                st.error(f"Error: {str(e)}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    elif submitted and not hw:
        st.warning("Please select a valid GPU.")
    elif submitted and not use_case:
        st.warning("Please describe your use case.")

    if st.session_state.recommendations and st.session_state.last_hw:
        if not (submitted and hw and use_case):
            render_results(st.session_state.recommendations,
                           st.session_state.last_hw, st.session_state.last_uc)


if __name__ == "__main__":
    main()