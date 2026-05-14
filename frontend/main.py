import sys
import streamlit as st
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.hardware_parser import get_available_gpu_options, parse_hardware_input, ParsedHardware
from backend.services.recommender import get_recommender, ScoredModel


GREETING = """
👋 **Welcome to the LLM Recommender!**

I'll help you find the best locally-hostable open-weight LLM for your hardware and use case.

**How it works:**
1. Tell me about your GPU(s) - how many and what type
2. Describe your use case (coding, math, reasoning, etc.)
3. I'll recommend the best models that fit your setup

*Powered by Open LLM Leaderboard data with semantic search and benchmark scoring.*
"""


def render_hardware_input() -> Optional[ParsedHardware]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        gpu_options = get_available_gpu_options()
        gpu_names = [name for name, _ in gpu_options]
        
        selected_gpu_name = st.selectbox(
            "GPU Type",
            options=gpu_names,
            index=gpu_names.index("A100 80GB") if "A100 80GB" in gpu_names else 0,
            help="Select your GPU type"
        )
    
    with col2:
        gpu_count = st.number_input(
            "Number of GPUs",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="How many of this GPU do you have?"
        )
    
    selected_gpu_id = None
    for name, gid in gpu_options:
        if name == selected_gpu_name:
            selected_gpu_id = gid
            break
    
    if selected_gpu_id:
        from backend.services.gpu_catalog import GPU_CATALOG
        gpu_config = GPU_CATALOG.get(selected_gpu_id, {})
        
        st.caption(f"**{gpu_config.get('vram_gb', 0)} GB per GPU** | "
                   f"Total: **{gpu_config.get('vram_gb', 0) * gpu_count} GB VRAM** | "
                   f"Tier: {gpu_config.get('tier', 'unknown').replace('_', ' ').title()}")
        
        return ParsedHardware(
            gpu_id=selected_gpu_id,
            gpu_name=selected_gpu_name,
            vram_gb=gpu_config.get('vram_gb', 0),
            count=gpu_count,
            total_vram_gb=gpu_config.get('vram_gb', 0) * gpu_count,
            tier=gpu_config.get('tier', 'unknown')
        )
    
    return None


def render_model_card(model: ScoredModel, rank: int) -> None:
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"#{rank}: {model.model_id.split('/')[-1]}")
            st.caption(f"Full ID: `{model.model_id}`")
        
        with col2:
            score_pct = int(model.final_score * 100)
            st.metric("Match Score", f"{score_pct}%")
        
        with st.expander("Details", expanded=rank <= 2):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**Benchmarks**")
                st.write(f"- Coding: {model.coding:.1f}")
                st.write(f"- Math: {model.math_score:.1f}")
                st.write(f"- Reasoning: {model.reasoning:.1f}")
                st.write(f"- Intelligence: {model.intelligence_index:.1f}")
            
            with col_b:
                st.markdown("**Model Info**")
                st.write(f"- Parameters: {model.params_billions:.1f}B")
                st.write(f"- Size: {model.vram_fp16:.1f} GB")
                st.write(f"- Type: {'MoE' if model.is_moe else 'Dense'}")
                st.write(f"- Strategy: {model.hosting_strategy}")
            
            with col_c:
                st.markdown("**Hardware Fit**")
                hw = model.matched_hardware
                st.write(f"- Status: {hw.get('status', 'N/A')}")
                st.write(f"- Quantization: {hw.get('quantization', 'N/A')}")
                st.write(f"- Parallelism: {hw.get('parallelism', 'N/A')}")
            
            if model.hf_repo_id:
                hf_url = f"https://huggingface.co/{model.hf_repo_id}"
                st.markdown(f"[View on Hugging Face]({hf_url})")
        
        st.divider()


def render_results(recommendations: list[ScoredModel], hardware: ParsedHardware, use_case: str) -> None:
    st.subheader(f"Top {len(recommendations)} Recommendations")
    
    if recommendations:
        st.success(f"Found {len(recommendations)} models compatible with "
                   f"{hardware.count}x {hardware.gpu_name} for **{use_case}**")
        
        for i, model in enumerate(recommendations, 1):
            render_model_card(model, i)
    else:
        st.warning("No models found matching your criteria.")


def main():
    st.set_page_config(
        page_title="LLM Recommender",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM Recommender")
    st.markdown("Find the best open-weight LLMs for your hardware")
    
    with st.chat_message("assistant"):
        st.markdown(GREETING)
    
    st.divider()
    
    with st.form("recommendation_form", clear_on_submit=False):
        st.subheader("Hardware Configuration")
        hardware = render_hardware_input()
        
        st.subheader("Use Case")
        use_case = st.text_area(
            "Describe your use case",
            placeholder="e.g., Code generation and debugging, Mathematical reasoning, General chat...",
            height=100,
            help="Describe what you'll use the LLM for"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            submitted = st.form_submit_button("🔍 Get Recommendations", type="primary", use_container_width=True)
        with col_btn2:
            clear = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    if clear:
        st.rerun()
    
    if submitted and hardware and use_case:
        with st.spinner("Finding best models..."):
            try:
                start_time = time.time()
                recommender = get_recommender()
                recommendations = recommender.recommend(
                    hardware=hardware,
                    use_case_text=use_case,
                    user_query=use_case,
                    top_k=5
                )
                elapsed = time.time() - start_time
                
                st.info(f"Query completed in {elapsed:.2f}s. "
                        f"Searching from {len(recommender.models):,} models.")
                
                render_results(recommendations, hardware, use_case)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif submitted and not hardware:
        st.warning("Please select a valid GPU configuration.")
    elif submitted and not use_case:
        st.warning("Please describe your use case.")


if __name__ == "__main__":
    main()