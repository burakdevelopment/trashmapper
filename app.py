import streamlit as st
import cv2
import time
import numpy as np
from backend import TrashEngine, COLORS

st.set_page_config(page_title="TrashMapper", layout="wide")


if 'engine' not in st.session_state:
    st.session_state.engine = TrashEngine()
    st.session_state.processing = False
    st.session_state.last_time = time.time()

engine = st.session_state.engine

st.sidebar.title("♻️ TrashMapper V1")
st.sidebar.info("Raspberry Pi 5 + Cam v3")

mode = st.sidebar.radio("Mod", ["Live View", "Session/Mapping", "Reports"])

fps_placeholder = st.sidebar.empty()
obj_count_placeholder = st.sidebar.metric("Instant Object", 0)

st.title("Smart Waste Detection & Mapping")

col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    status_text = st.empty()
    if mode == "Oturum/Haritalama":
        if not engine.is_running:
            if st.button("🚀 START (Mapping)", type="primary"):
                engine.start_session()
                st.rerun()
        else:
            st.warning("Registration and Mapping Active...")
            st.write(f"Distance Covered (Estimated): {engine.pose['y']:.1f} m")
            if st.button("🛑 Finish and Report."):
                report = engine.stop_session()
                st.session_state.last_report = report
                st.success("Report Created!")
                time.sleep(1)
                st.rerun()

    if 'last_report' in st.session_state and mode == "Raporlar":
        rep = st.session_state.last_report
        st.subheader("Last Session Summary")
        st.write(f"**Total Waste:** {rep['total_objects']}")
        
        st.image(rep['hist_path'], caption="Waste Distribution")
        st.image(rep['heatmap_path'], caption="Temperature Map")
        
        st.write("### 💡 Suggestions")
        for sug in rep['suggestions']:
            st.error(sug)


run_loop = True
while run_loop:
    
    current_time = time.time()
    dt = current_time - st.session_state.last_time
    st.session_state.last_time = current_time
    fps = 1.0 / (dt + 1e-9)
    
    
    frame = engine.get_frame()
    
    
    detections = engine.detect(frame)
    
    
    if engine.is_running:
        engine.update_map(detections, dt)
    
    
    for det in detections:
        x1, y1, x2, y2 = [int(x) for x in det.box]
        color = [int(c) for c in COLORS[det.label_id % len(COLORS)]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label_name} {det.conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    
    fps_placeholder.write(f"FPS: {fps:.1f}")
    obj_count_placeholder.metric("Instant Object", len(detections))
    
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    time.sleep(0.001)
