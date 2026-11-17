import streamlit as st
import pandas as pd
import sqlite3
import os
#import time
import joblib

from sleep_rules import calculate_stress_level, calculate_movement_score, evaluate_sleep_quality

# ================= 1. CẤU HÌNH & KHỞI TẠO =================
st.set_page_config(page_title="Sleep AIoT Monitor", layout="wide")

# Ép chữ trong các ô thông số (Metric) tự động xuống hàng
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important; 
        word-wrap: break-word !important; 
        font-size: 24px !important; 
        line-height: 1.2 !important; 
    }
    /* Làm nổi bật thanh radio (Menu tabs) */
    div[role="radiogroup"] > label {
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "data", "database", "sleep_records.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sleep_rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

@st.cache_resource
def load_ai_models():
    try:
        rf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return rf, scaler
    except Exception as e:
        st.error(f"Lỗi tải AI: {e}")
        return None, None

rf_model, scaler = load_ai_models()

def get_data(limit):
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT * FROM sensor_data ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

# Khởi tạo bộ nhớ tạm để lưu danh sách các tab báo cáo
if 'reports' not in st.session_state:
    st.session_state.reports = {}
if 'report_counter' not in st.session_state:
    st.session_state.report_counter = 1
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Theo dõi Real-time"

# ================= 2. THANH ĐIỀU HƯỚNG (SIDEBAR) =================
with st.sidebar:
    st.header("🌅 Tổng kết giấc ngủ")
    st.write("Bấm vào nút dưới đây sau khi thức dậy để hệ thống tổng hợp dữ liệu cả đêm.")
    
    if st.button("Tạo báo cáo tổng kết", use_container_width=True):
        df_night = get_data(limit=1000) # Lấy dữ liệu cả đêm
        
        if not df_night.empty:
            # 1. Tính toán trung bình toàn thời gian đo
            avg_temp = df_night['temperature'].mean() if 'temperature' in df_night.columns else 25.0
            avg_hum = df_night['humidity'].mean() if 'humidity' in df_night.columns else 60.0
            avg_lux = df_night['light_lux'].mean() if 'light_lux' in df_night.columns else 0.0
            avg_hr = df_night['heart_rate'].mean()
            avg_spo2 = df_night['spo2'].mean()
            
            stress_score = calculate_stress_level(df_night['heart_rate'].values, df_night['spo2'].values)
            mov_score, mov_status = calculate_movement_score(df_night[['accel_x', 'accel_y', 'accel_z']])
            
            # 2. Đưa vào AI
            if rf_model and scaler:
                input_df = pd.DataFrame([{
                    'Heart Rate': avg_hr, 'Physical Activity Level': mov_score,
                    'Sleep Duration': 6.0, 'Stress Level': stress_score
                }])
                input_scaled = scaler.transform(input_df)
                final_score = rf_model.predict(input_scaled)[0] * 10 
                eval_result = evaluate_sleep_quality(score=final_score, temp_c=avg_temp, hum_pct=avg_hum, hr_bpm=avg_hr)
                
                # 3. Lưu toàn bộ dữ liệu vào 1 Tab báo cáo mới
                report_name = f"Báo cáo lần {st.session_state.report_counter}"
                st.session_state.reports[report_name] = {
                    'df': df_night, 'score': final_score, 'rating': eval_result['rating'], 'tips': eval_result['tips'],
                    'temp': avg_temp, 'hum': avg_hum, 'lux': avg_lux, 'hr': avg_hr, 'spo2': avg_spo2, 
                    'mov': mov_score, 'mov_status': mov_status, 'stress': stress_score
                }
                
                st.session_state.report_counter += 1
                st.session_state.active_tab = report_name # Chuyển view sang báo cáo vừa tạo
                st.rerun()
        else:
            st.warning("Chưa có đủ dữ liệu!")

    st.markdown("---")
    st.subheader("📌 Danh sách chế độ xem")
    
    # Tạo danh sách các Tab
    options = ["Theo dõi Real-time"] + list(st.session_state.reports.keys())
    
    # Render các Tab dưới dạng Radio button
    selected_tab = st.radio("Chọn màn hình hiển thị:", options, index=options.index(st.session_state.active_tab), label_visibility="collapsed")
    
    # Nếu người dùng click chuyển Tab -> Cập nhật trạng thái và load lại trang
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()


# ================= 3. MÀN HÌNH CHÍNH (MAIN VIEW) =================

if st.session_state.active_tab != "Theo dõi Real-time":
    # ------------------ GIAO DIỆN BÁO CÁO TĨNH ------------------
    rep = st.session_state.reports[st.session_state.active_tab]
    
    st.title(f"📑 Báo cáo tổng kết: {st.session_state.active_tab}")
    
    st.subheader("🧠 Đánh giá Giấc ngủ & Khuyến nghị y tế")
    st.success(f"**Điểm AI Trung bình:** {rep['score']:.1f}/100 - **Xếp loại:** {rep['rating']}")
    
    for tip in rep['tips']:
        st.info(f"💡 {tip}")
        
    st.markdown("---")
    st.subheader("📊 Thông số Trung bình (Cả đêm)")
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Nhịp tim (BPM)", f"{rep['hr']:.1f}")
    col2.metric("SpO2 (%)", f"{rep['spo2']:.1f}%")
    col3.metric("Mức độ Stress", rep['stress'])
    col4.metric("Vận động", rep['mov_status'], f"{rep['mov']}/10", delta_color="off")
    col5.metric("Nhiệt độ (°C)", f"{rep['temp']:.1f}°C")
    col6.metric("Độ ẩm (%)", f"{rep['hum']:.1f}%")
    col7.metric("Ánh sáng (Lux)", f"{rep['lux']:.1f} lx")
    
    st.markdown("---")
    df_night = rep['df']
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("❤️ Biểu đồ Nhịp tim & SpO2 (Toàn bộ thời gian)")
        st.line_chart(df_night.set_index("timestamp")[["heart_rate", "spo2"]])
    with col_chart2:
        st.subheader("🏃 Gia tốc Chuyển động (Toàn bộ thời gian)")
        st.line_chart(df_night.set_index("timestamp")[["accel_x", "accel_y", "accel_z"]])

else:
    # ------------------ GIAO DIỆN THEO DÕI REAL-TIME ------------------
    st.title("🌙 Theo dõi Giấc ngủ AIoT Dashboard")
    @st.fragment(run_every="2s")
    def show_realtime_content():
        df = get_data(limit=100)
        
        if df.empty or len(df) < 5:
            st.warning("⏳ Đang chờ thu thập dữ liệu từ phần cứng ESP32...")
        else:
            latest = df.iloc[-1]
            
            # Tính toán chỉ số Real-time
            stress_score = calculate_stress_level(df['heart_rate'].values, df['spo2'].values)
            mov_score, mov_status = calculate_movement_score(df.tail(30)[['accel_x', 'accel_y', 'accel_z']])
            
            if rf_model and scaler:
                input_df = pd.DataFrame([{
                    'Heart Rate': latest['heart_rate'], 'Physical Activity Level': mov_score,
                    'Sleep Duration': 6.0, 'Stress Level': stress_score,
                }])
                
                input_scaled = scaler.transform(input_df)
                predicted_score = rf_model.predict(input_scaled)[0] * 10
                
                st.subheader("🧠 Trạng thái giấc ngủ (Real-time)")
                if predicted_score >= 80:
                    st.success(f"Điểm AI: **{predicted_score:.1f}/100** - Trạng thái: **Ngủ sâu / Thư giãn**")
                elif predicted_score >= 60:
                    st.info(f"Điểm AI: **{predicted_score:.1f}/100** - Trạng thái: **Ngủ nông**")
                else:
                    st.error(f"Điểm AI: **{predicted_score:.1f}/100** - Trạng thái: **Thức / Trằn trọc**")
                
                temp = latest.get('temperature', 25.0)
                hum = latest.get('humidity', 60.0)
                alerts = []
                if temp > 28.5: alerts.append(f"Nhiệt độ cao ({temp:.1f}°C)")
                elif temp < 20.0: alerts.append(f"Nhiệt độ thấp ({temp:.1f}°C)")
                if hum > 75.0: alerts.append(f"Quá ẩm ({hum:.1f}%)")
                if alerts:
                    st.warning("⚠️ Cảnh báo Môi trường: " + " | ".join(alerts))

            st.markdown("---")
            st.subheader("📊 Thông số Thời gian thực (1 phút qua)")
            
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            col1.metric("Nhịp tim", f"{latest['heart_rate']:.1f}")
            col2.metric("SpO2", f"{latest['spo2']}%")
            col3.metric("Mức độ Stress", stress_score)
            col4.metric("Vận động", mov_status, f"{mov_score}/10")
            col5.metric("Nhiệt độ", f"{latest.get('temperature', 0):.1f}°C")
            col6.metric("Độ ẩm", f"{latest.get('humidity', 0):.1f}%")
            col7.metric("Ánh sáng", f"{latest.get('light_lux', 0):.0f} lx")
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("❤️ Biểu đồ Nhịp tim & SpO2")
                st.line_chart(df.set_index("timestamp")[["heart_rate", "spo2"]])
            with col_chart2:
                st.subheader("🏃 Gia tốc (Chuyển động cơ thể)")
                st.line_chart(df.set_index("timestamp")[["accel_x", "accel_y", "accel_z"]])

    # === SỬA: Gọi hàm vừa tạo thay cho time.sleep(2) ===
    show_realtime_content()