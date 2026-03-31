import streamlit as st
import subprocess
import sys

# 1. Page Configuration for a Modern Dashboard Look
st.set_page_config(page_title="Autonomous DBA Agent", page_icon="🤖", layout="wide")

st.title("🚀 Autonomous DBA Optimization Agent")
st.markdown("**Status:** Sovereign AI Secured | **Tier:** Top Tier (3.00/3.20)")

# 2. Sidebar for Hackathon Context
st.sidebar.header("Agent Configuration")
st.sidebar.info(
    "This dashboard acts as a secure telemetry monitor. "
    "The core 72B-parameter optimization engine remains fully containerized and unaltered, "
    "executing autonomous Chain-of-Thought reasoning to optimize database storage and query costs."
)

st.markdown("---")

# 3. The Execution Trigger
if st.button("Initialize & Run Agent", type="primary"):
    st.markdown("#### 📡 Live Agent Telemetry")
    
    # Placeholder to update the terminal output live
    terminal_output = st.empty()
    log_text = ""
    
    with st.spinner("Agent waking up... Connecting to Environment..."):
        # 4. The "Safe" Execution: Runs your exact file as a separate process
        process = subprocess.Popen(
            [sys.executable, "inference.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 5. Live Streaming the Output
        for line in process.stdout:
            log_text += line
            # Displays the output in a clean, dark "code" block
            terminal_output.code(log_text, language="bash")
            
        process.wait()
        
        # 6. Final Verdict
        if process.returncode == 0:
            st.success("✅ Agent execution completed successfully.")
            st.balloons() # Adds a nice visual touch for the demo
        else:
            st.error("❌ Agent execution encountered an error.")