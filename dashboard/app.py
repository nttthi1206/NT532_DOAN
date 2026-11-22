import json
import os
import sqlite3
import time
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

from sleep_rules import (
    calculate_movement_score,
    calculate_stress_level,
    evaluate_sleep_quality,
)
from llm_advisor import (
    PRESET_SCENARIOS,
    analyze_sleep_with_gemini,
    generate_simulated_night_df,
    load_api_key_from_env,
)

# ================= 1. CẤU HÌNH TRANG & CSS =================
st.set_page_config(
    page_title="Sleep AIoT Monitor",
    page_icon="🌙",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

    /* Áp dụng font chữ tiếng Việt hiện đại, to rõ cho toàn bộ trang */
    html, body, [class*="css"], .stApp {
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Tăng kích thước chữ nội dung văn bản, đoạn văn, danh sách */
    .stMarkdown p, .stMarkdown li, .stText, div[data-testid="stText"] {
        font-size: 17.5px !important;
        line-height: 1.6 !important;
    }

    /* Tăng kích thước chữ trong các hộp thông báo (Success, Info, Warning, Error) */
    div[data-testid="stAlertContent"] p {
        font-size: 17.5px !important;
        line-height: 1.55 !important;
        font-weight: 500 !important;
    }

    /* Tăng kích thước chữ chú thích (Caption) cho dễ đọc, không bị mờ nhỏ */
    div[data-testid="stCaptionContainer"] p, .stCaption {
        font-size: 15px !important;
        line-height: 1.5 !important;
        opacity: 0.92 !important;
    }

    /* Tiêu đề chính và các tiêu đề phụ to, đậm, rõ nét */
    h1 {
        font-size: 32px !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px !important;
    }
    h2, h3 {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    h4 {
        font-size: 21px !important;
        font-weight: 700 !important;
        margin-top: 10px !important;
    }
    h5 {
        font-size: 18.5px !important;
        font-weight: 700 !important;
    }

    /* Chống cắt chữ cho toàn bộ ô Metric (Label, Value, Delta) */
    div[data-testid="stMetric"] {
        overflow: visible !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        font-size: 15.5px !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        font-size: 24px !important;
        line-height: 1.25 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
    }

    /* Làm nổi bật thanh Tabs chính trên màn hình (tự động xuống dòng/giãn chiều cao, không bị cắt chữ) */
    div[data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 10px !important;
        padding-bottom: 6px !important;
    }
    button[data-baseweb="tab"] {
        height: auto !important;
        min-height: 54px !important;
        padding: 12px 22px !important;
        border-radius: 10px 10px 0 0 !important;
        background-color: rgba(128, 128, 128, 0.08) !important;
        white-space: normal !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.14) !important;
    }
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div,
    button[data-baseweb="tab"] {
        font-size: 20.5px !important;
        font-weight: 800 !important;
        line-height: 1.3 !important;
        white-space: normal !important;
    }

    /* Nhãn của các ô chọn (Selectbox, Slider, Radio, Checkbox, Toggle, Button) */
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stCheckbox"] label p,
    div[role="radiogroup"] label p {
        font-size: 16.5px !important;
        font-weight: 600 !important;
        white-space: normal !important;
    }

    /* Nút bấm tự động giãn chiều cao nếu chữ dài, không bao giờ mất chữ */
    div.stButton > button {
        height: auto !important;
        min-height: 44px !important;
        padding: 8px 14px !important;
        white-space: normal !important;
    }
    div.stButton > button p {
        font-size: 16.5px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        white-space: normal !important;
    }
    div[data-baseweb="select"] > div {
        font-size: 16px !important;
    }

    /* Tiêu đề của thanh thu gọn (Expander) */
    details summary p {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "data", "database", "sleep_records.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sleep_rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
PORTABLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "sleep_rf_portable.json")


# ================= 2. NẠP MÔ HÌNH AI (RANDOM FOREST) =================
class PortableRandomForest:
    """Bộ dự đoán Random Forest & StandardScaler đọc từ JSON để tương thích mọi phiên bản Python/scikit-learn."""

    def __init__(self, data):
        self.feature_names = data["feature_names_in"]
        self.mean = data["mean"]
        self.scale = data["scale"]
        self.trees = data["trees"]

    def transform(self, df):
        row = df.iloc[0]
        return [
            (float(row[col]) - m) / s
            for col, m, s in zip(self.feature_names, self.mean, self.scale)
        ]

    def predict(self, scaled_x):
        x = scaled_x[0] if hasattr(scaled_x[0], "__len__") else scaled_x
        preds = []
        for t in self.trees:
            node = 0
            left = t["left_child"]
            right = t["right_child"]
            feat = t["feature"]
            thresh = t["threshold"]
            while left[node] != -1:
                if x[feat[node]] <= thresh[node]:
                    node = left[node]
                else:
                    node = right[node]
            preds.append(t["value"][node])
        return [sum(preds) / len(preds)]


@st.cache_resource
def load_ai_models():
    try:
        rf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return rf, scaler
    except Exception as e:
        if os.path.exists(PORTABLE_MODEL_PATH):
            try:
                with open(PORTABLE_MODEL_PATH, "r", encoding="utf-8") as f:
                    portable = PortableRandomForest(json.load(f))
                return portable, portable
            except Exception as e2:
                st.error(f"Lỗi tải AI: {e2}")
                return None, None
        st.error(f"Lỗi tải AI: {e}")
        return None, None


rf_model, scaler = load_ai_models()


# ================= 3. HÀM XỬ LÝ DỮ LIỆU & BÁO CÁO =================
def get_data(limit=100):
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            f"SELECT * FROM sensor_data ORDER BY id DESC LIMIT {limit}", conn
        )
        conn.close()
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def save_df_to_sqlite(df_sim):
    """Lưu dữ liệu giả lập vào bảng sensor_data để test cả màn hình Real-time."""
    try:
        os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                accel_x REAL,
                accel_y REAL,
                accel_z REAL,
                light_lux REAL,
                heart_rate REAL,
                spo2 INTEGER,
                temperature REAL,
                humidity REAL
            )
            """
        )
        rows = df_sim[
            [
                "timestamp",
                "accel_x",
                "accel_y",
                "accel_z",
                "light_lux",
                "heart_rate",
                "spo2",
                "temperature",
                "humidity",
            ]
        ].values.tolist()
        cursor.executemany(
            """
            INSERT INTO sensor_data
            (timestamp, accel_x, accel_y, accel_z, light_lux, heart_rate, spo2, temperature, humidity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu giả lập vào SQLite: {e}")
        return False


def build_report_from_dataframe(
    df_night, sleep_duration=7.5, api_key="", report_label=None
):
    """Tính toán chỉ số từ DataFrame, dự đoán điểm bằng Random Forest và phân tích bằng Gemini LLM."""
    avg_temp = (
        float(df_night["temperature"].mean())
        if "temperature" in df_night.columns
        else 25.0
    )
    avg_hum = (
        float(df_night["humidity"].mean())
        if "humidity" in df_night.columns
        else 60.0
    )
    avg_lux = (
        float(df_night["light_lux"].mean())
        if "light_lux" in df_night.columns
        else 0.0
    )

    valid_hr_series = df_night.loc[df_night["heart_rate"] > 40, "heart_rate"]
    valid_spo2_series = df_night.loc[df_night["spo2"] > 50, "spo2"]

    avg_hr = float(valid_hr_series.mean()) if not valid_hr_series.empty else 72.0
    avg_spo2 = float(valid_spo2_series.mean()) if not valid_spo2_series.empty else 97.0

    stress_score = calculate_stress_level(
        df_night["heart_rate"].values, df_night["spo2"].values
    )
    mov_score, mov_status = calculate_movement_score(
        df_night[["accel_x", "accel_y", "accel_z"]]
    )

    if rf_model and scaler:
        input_df = pd.DataFrame(
            [
                {
                    "Heart Rate": avg_hr,
                    "Physical Activity Level": mov_score,
                    "Sleep Duration": sleep_duration,
                    "Stress Level": stress_score,
                }
            ]
        )
        input_scaled = scaler.transform(input_df)
        final_score = float(rf_model.predict(input_scaled)[0] * 10)
    else:
        final_score = 70.0

    eval_result = evaluate_sleep_quality(
        score=final_score, temp_c=avg_temp, hum_pct=avg_hum, hr_bpm=avg_hr
    )

    metrics = {
        "score": final_score,
        "rating": eval_result["rating"],
        "hr": avg_hr,
        "spo2": avg_spo2,
        "stress": stress_score,
        "mov_score": mov_score,
        "mov_status": mov_status,
        "temp": avg_temp,
        "hum": avg_hum,
        "lux": avg_lux,
        "sleep_duration": sleep_duration,
    }

    llm_analysis = analyze_sleep_with_gemini(
        metrics=metrics,
        rule_tips=eval_result["tips"],
        api_key=api_key,
    )

    return {
        "df": df_night,
        "score": final_score,
        "rating": eval_result["rating"],
        "tips": eval_result["tips"],
        "temp": avg_temp,
        "hum": avg_hum,
        "lux": avg_lux,
        "hr": avg_hr,
        "spo2": avg_spo2,
        "mov": mov_score,
        "mov_status": mov_status,
        "stress": stress_score,
        "sleep_duration": sleep_duration,
        "llm_analysis": llm_analysis,
        "scenario_note": report_label,
        "analyzed_at": datetime.now().strftime("%H:%M:%S - %d/%m/%Y"),
    }


def render_detailed_report(rep, key_prefix="rep"):
    """Hiển thị giao diện Báo cáo & Phân tích Giấc ngủ trực quan, dễ đọc."""
    llm = rep.get("llm_analysis", {})
    score = rep["score"]
    rating = rep["rating"]

    # Header thông tin nguồn & thời gian
    st.caption(
        f"🕒 Thời điểm phân tích: **{rep.get('analyzed_at', 'Vừa xong')}**  |  "
        f"📌 Nguồn dữ liệu: **{rep.get('scenario_note', 'SQLite Database')}**  |  "
        f"⏳ Thời lượng ngủ: **{rep.get('sleep_duration', 7.5):.1f} giờ**"
    )

    # Khối Điểm số tổng quát + Nút phân tích lại
    col_score, col_retry = st.columns([3.5, 1.5])
    with col_score:
        if score >= 80:
            st.success(
                f"🌟 **Chất lượng Giấc ngủ: {score:.1f} / 100 điểm** — Xếp loại: **{rating}**"
            )
        elif score >= 60:
            st.warning(
                f"🌤️ **Chất lượng Giấc ngủ: {score:.1f} / 100 điểm** — Xếp loại: **{rating}**"
            )
        else:
            st.error(
                f"⚠️ **Chất lượng Giấc ngủ: {score:.1f} / 100 điểm** — Xếp loại: **{rating}**"
            )
    with col_retry:
        if st.button(
            "🔄 Phân tích lại (Gemini)",
            key=f"{key_prefix}_retry_btn",
            use_container_width=True,
        ):
            with st.spinner("Đang gọi Gemini AI phân tích lại..."):
                metrics_retry = {
                    "score": rep["score"],
                    "rating": rep["rating"],
                    "hr": rep["hr"],
                    "spo2": rep["spo2"],
                    "stress": rep["stress"],
                    "mov_score": rep["mov"],
                    "mov_status": rep["mov_status"],
                    "temp": rep["temp"],
                    "hum": rep["hum"],
                    "lux": rep["lux"],
                    "sleep_duration": rep.get("sleep_duration", 7.5),
                }
                rep["llm_analysis"] = analyze_sleep_with_gemini(
                    metrics=metrics_retry,
                    rule_tips=rep["tips"],
                    api_key=st.session_state.gemini_api_key,
                )
                rep["analyzed_at"] = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
                st.rerun()

    # Nhãn nguồn AI
    source_tag = llm.get("source", "rule_fallback")
    if source_tag.startswith("gemini"):
        st.caption(f"✨ Báo cáo tư vấn được sinh tự động bởi **Google {source_tag.upper()}**")
    elif llm.get("error_detail"):
        st.warning(f"ℹ️ {llm['error_detail']}")

    # Bảng chỉ số trung bình chia làm 2 nhóm rõ ràng: Sinh lý & Môi trường (tránh nhồi 7 cột gây cắt chữ)
    col_g1, col_g2 = st.columns([4, 3])
    with col_g1:
        with st.container(border=True):
            st.markdown("##### 🫀 Chỉ số Sinh lý Trung bình")
            c1, c2 = st.columns(2)
            c1.metric("❤️ Nhịp tim", f"{rep['hr']:.1f} BPM")
            c2.metric("🩸 SpO2", f"{rep['spo2']:.1f}%")
            c3, c4 = st.columns(2)
            c3.metric("🧠 Mức Stress", f"{rep['stress']}/10")
            c4.metric("🏃 Vận động", rep["mov_status"], f"Điểm: {rep['mov']}/10", delta_color="off")

    with col_g2:
        with st.container(border=True):
            st.markdown("##### 🏠 Môi trường Phòng ngủ Trung bình")
            e1, e2 = st.columns(2)
            e1.metric("🌡️ Nhiệt độ", f"{rep['temp']:.1f}°C")
            e2.metric("💧 Độ ẩm", f"{rep['hum']:.1f}%")
            st.metric("💡 Ánh sáng", f"{rep['lux']:.1f} Lux")

    # 1. Đánh giá tổng quan
    st.markdown("#### 🧠 1. Đánh giá Tổng quan")
    with st.container(border=True):
        st.markdown(llm.get("overall_assessment", "Không có dữ liệu đánh giá."))

    # 2. Phân tích các yếu tố chính ảnh hưởng
    st.markdown("#### 🔍 2. Nguyên nhân & Các Yếu tố Ảnh hưởng Chính")
    key_factors = llm.get("key_factors", [])
    if key_factors:
        cols_per_row = min(3, len(key_factors))
        for i in range(0, len(key_factors), cols_per_row):
            row_items = key_factors[i : i + cols_per_row]
            factor_cols = st.columns(len(row_items))
            for idx, item in enumerate(row_items):
                impact = item.get("impact", "Cần chú ý")
                if "Tích cực" in impact:
                    badge = "🟢 Tích cực"
                elif "Tiêu cực" in impact:
                    badge = "🔴 Tiêu cực"
                else:
                    badge = "🟡 Cần chú ý"

                with factor_cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"**{item.get('factor', 'Yếu tố')}**")
                        st.caption(f"Mức độ ảnh hưởng: **{badge}**")
                        st.write(item.get("analysis", ""))

    # 3. Khuyến nghị cá nhân hóa
    st.markdown("#### 💡 3. Lời khuyên Cải thiện Giấc ngủ Cá nhân hóa")
    col_env, col_habit = st.columns(2)
    with col_env:
        with st.container(border=True):
            st.markdown("##### 🏠 Điều chỉnh Môi trường Phòng ngủ")
            for rec in llm.get("environment_recommendations", []):
                st.info(f"🌡️ {rec}")
    with col_habit:
        with st.container(border=True):
            st.markdown("##### 🧘 Điều chỉnh Thói quen & Sinh lý")
            for rec in llm.get("habit_recommendations", []):
                st.success(f"🌿 {rec}")

    # 4. Biểu đồ chi tiết thu gọn trong Expander để giao diện không bị dài/rối
    with st.expander("📈 Xem Biểu đồ Diễn biến Cảm biến của Phiên ngủ này", expanded=False):
        df_night = rep["df"]
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**❤️ Nhịp tim (BPM) & SpO2 (%)**")
            st.line_chart(df_night.set_index("timestamp")[["heart_rate", "spo2"]])
        with ch2:
            st.markdown("**🏃 Gia tốc Chuyển động 3 trục (MPU6050)**")
            st.line_chart(
                df_night.set_index("timestamp")[["accel_x", "accel_y", "accel_z"]]
            )


# ================= 4. KHỞI TẠO SESSION STATE =================
if "reports" not in st.session_state:
    st.session_state.reports = {}
if "report_counter" not in st.session_state:
    st.session_state.report_counter = 1
if "selected_report_name" not in st.session_state:
    st.session_state.selected_report_name = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = load_api_key_from_env()
if "auto_gemini_enabled" not in st.session_state:
    st.session_state.auto_gemini_enabled = True
if "auto_gemini_interval_sec" not in st.session_state:
    st.session_state.auto_gemini_interval_sec = 60
if "auto_wake_detect" not in st.session_state:
    st.session_state.auto_wake_detect = True
if "last_auto_analysis_ts" not in st.session_state:
    st.session_state.last_auto_analysis_ts = 0.0
if "last_auto_analyzed_db_id" not in st.session_state:
    st.session_state.last_auto_analyzed_db_id = None
if "realtime_llm_report" not in st.session_state:
    st.session_state.realtime_llm_report = None
if "last_wake_report_db_id" not in st.session_state:
    st.session_state.last_wake_report_db_id = None


# ================= 5. THANH BÊN TRÁI (SIDEBAR GỌN GÀNG) =================
with st.sidebar:
    st.title("🌙 Sleep AIoT")
    st.caption("Hệ thống Giám sát & Tư vấn Giấc ngủ Thông minh")
    st.markdown("---")

    st.subheader("1️⃣ Trạng thái Hệ thống")
    if rf_model and scaler:
        st.success("🧠 Mô hình AI: **Random Forest (Sẵn sàng)**")
    else:
        st.error("⚠️ Mô hình AI: **Chưa nạp được**")

    if st.session_state.gemini_api_key:
        key_type = (
            "OpenRouter Gemini"
            if st.session_state.gemini_api_key.startswith("sk-or-")
            else "Google Gemini"
        )
        st.info(f"🤖 Tư vấn LLM: **{key_type} (Đã kết nối)**")
    else:
        st.warning("🤖 Tư vấn LLM: **Chế độ dự phòng (Chưa có Key)**")

    with st.expander("🔑 Đổi Gemini API Key", expanded=False):
        api_key_input = st.text_input(
            "Nhập Gemini / OpenRouter Key:",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIzaSy... hoặc sk-or-v1-...",
        )
        st.session_state.gemini_api_key = api_key_input.strip()

    st.markdown("---")
    st.subheader("2️⃣ Chế độ Tự động (ESP32)")
    st.session_state.auto_gemini_enabled = st.toggle(
        "Tự động phân tích Gemini định kỳ",
        value=st.session_state.auto_gemini_enabled,
        help="Tự động gọi Gemini phân tích khi có dữ liệu mới từ ESP32 mà không cần bấm nút.",
    )

    interval_options = {
        "30 giây (Test nhanh)": 30,
        "1 phút": 60,
        "5 phút": 300,
        "15 phút": 900,
        "30 phút": 1800,
    }
    current_label = next(
        (
            k
            for k, v in interval_options.items()
            if v == st.session_state.auto_gemini_interval_sec
        ),
        "1 phút",
    )
    selected_interval_label = st.selectbox(
        "Chu kỳ cập nhật lời khuyên:",
        list(interval_options.keys()),
        index=list(interval_options.keys()).index(current_label),
        disabled=not st.session_state.auto_gemini_enabled,
    )
    st.session_state.auto_gemini_interval_sec = interval_options[
        selected_interval_label
    ]

    st.session_state.auto_wake_detect = st.toggle(
        "Tự động chốt báo cáo khi thức dậy",
        value=st.session_state.auto_wake_detect,
        help="Khi phòng sáng (> 50 lux) và có cử động mạnh (>= 5/10), tự động lưu 1 bản báo cáo tổng kết vào Tab 2.",
    )

    st.markdown("---")
    st.caption(
        "💡 **Hướng dẫn nhanh:**\n"
        "- **Tab 1:** Xem thông số cảm biến ESP32 trực tiếp mỗi 2 giây.\n"
        "- **Tab 2:** Xem đánh giá chuyên sâu và lời khuyên từ Gemini AI.\n"
        "- **Tab 3:** Tạo dữ liệu giả lập để demo khi không cắm mạch ESP32."
    )


# ================= 6. MÀN HÌNH CHÍNH: CHIA 3 TAB RÕ RÀNG =================
st.title("🌙 Hệ thống Giám sát & Phân tích Giấc ngủ AIoT")

tab_realtime, tab_analysis, tab_simulator = st.tabs(
    [
        "📡 1. Theo dõi Giấc ngủ (Real-time)",
        "🧠 2. Phân tích & Lời khuyên (Gemini AI)",
        "🧪 3. Giả lập Thiết bị (Test Demo)",
    ]
)


# -------------------------------------------------------------------------
# TAB 1: THEO DÕI GIẤC NGỦ THỜI GIAN THỰC (REAL-TIME)
# -------------------------------------------------------------------------
with tab_realtime:
    @st.fragment(run_every="2s")
    def show_realtime_content():
        df = get_data(limit=100)

        if df.empty or len(df) < 5:
            st.info(
                "⏳ **Đang chờ dữ liệu từ mạch ESP32...**\n\n"
                "Nếu bạn đang không kết nối phần cứng ESP32, hãy chuyển sang tab **'🧪 3. Giả lập Thiết bị (Test Demo)'** ở phía trên để tạo dữ liệu mẫu!"
            )
            return

        latest = df.iloc[-1]
        latest_id = (
            int(latest["id"])
            if "id" in latest and pd.notna(latest["id"])
            else len(df)
        )

        # Lọc các mẫu = 0 khi chưa đặt ngón tay lên cảm biến MAX30102
        valid_hr_df = df[df["heart_rate"] > 40]
        current_hr = (
            float(latest["heart_rate"])
            if latest["heart_rate"] > 40
            else (
                float(valid_hr_df["heart_rate"].iloc[-1])
                if not valid_hr_df.empty
                else 72.0
            )
        )
        valid_spo2_df = df[df["spo2"] > 50]
        current_spo2 = (
            int(latest["spo2"])
            if latest["spo2"] > 50
            else (
                int(valid_spo2_df["spo2"].iloc[-1])
                if not valid_spo2_df.empty
                else 97
            )
        )

        current_temp = float(latest.get("temperature", 25.0))
        current_hum = float(latest.get("humidity", 60.0))
        current_lux = float(latest.get("light_lux", 0.0))

        # Tính toán Stress và Điểm vận động từ cửa sổ thời gian thực
        stress_score = calculate_stress_level(
            df["heart_rate"].values, df["spo2"].values
        )
        mov_score, mov_status = calculate_movement_score(
            df.tail(30)[["accel_x", "accel_y", "accel_z"]]
        )

        # Dự đoán điểm tức thời bằng Random Forest
        predicted_score = 70.0
        if rf_model and scaler:
            input_df = pd.DataFrame(
                [
                    {
                        "Heart Rate": current_hr,
                        "Physical Activity Level": mov_score,
                        "Sleep Duration": 7.5,
                        "Stress Level": stress_score,
                    }
                ]
            )
            input_scaled = scaler.transform(input_df)
            predicted_score = float(rf_model.predict(input_scaled)[0] * 10)

        # --- KHỐI 1: BANNER TRẠNG THÁI HIỆN TẠI ---
        col_status, col_env_alert = st.columns([3, 2])
        with col_status:
            if predicted_score >= 80:
                st.success(
                    f"🟢 **Trạng thái hiện tại: NGỦ SÂU / THƯ GIÃN** — Điểm AI tức thời: **{predicted_score:.1f}/100**"
                )
            elif predicted_score >= 60:
                st.info(
                    f"🔵 **Trạng thái hiện tại: NGỦ NÔNG** — Điểm AI tức thời: **{predicted_score:.1f}/100**"
                )
            else:
                st.error(
                    f"🔴 **Trạng thái hiện tại: THỨC / TRẰN TRỌC** — Điểm AI tức thời: **{predicted_score:.1f}/100**"
                )

        with col_env_alert:
            alerts = []
            if current_temp > 28.5:
                alerts.append(f"Phòng nóng ({current_temp:.1f}°C)")
            elif current_temp < 20.0:
                alerts.append(f"Phòng lạnh ({current_temp:.1f}°C)")
            if current_hum > 75.0:
                alerts.append(f"Độ ẩm cao ({current_hum:.1f}%)")
            if current_lux > 30.0:
                alerts.append(f"Đèn sáng ({current_lux:.0f} Lux)")

            if alerts:
                st.warning("⚠️ **Cảnh báo Môi trường:** " + " | ".join(alerts))
            else:
                st.success("✅ **Môi trường phòng ngủ:** Lý tưởng & Ổn định")

        # --- KHỐI 2: THÔNG SỐ CẢM BIẾN (TÁCH 2 NHÓM RỘNG RÃI, KHÔNG BỊ CẮT CHỮ) ---
        with st.container(border=True):
            st.markdown("##### 🫀 Chỉ số Sinh lý & Cử động (MAX30102 & MPU6050)")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("❤️ Nhịp tim", f"{current_hr:.1f} BPM")
            b2.metric("🩸 SpO2", f"{current_spo2}%")
            b3.metric("🧠 Mức Stress", f"{stress_score}/10")
            b4.metric("🏃 Vận động", mov_status, f"Điểm: {mov_score}/10", delta_color="off")

        with st.container(border=True):
            st.markdown("##### 🏠 Môi trường Phòng ngủ (DHT22 & BH1750)")
            r1, r2, r3 = st.columns(3)
            r1.metric("🌡️ Nhiệt độ", f"{current_temp:.1f}°C")
            r2.metric("💧 Độ ẩm", f"{current_hum:.1f}%")
            r3.metric("💡 Ánh sáng", f"{current_lux:.0f} Lux")

        # --- KHỐI 3: BIỂU ĐỒ THỜI GIAN THỰC ---
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            with st.container(border=True):
                st.markdown("##### 📈 Đường biểu diễn Nhịp tim & SpO2 (100 mẫu gần nhất)")
                st.line_chart(
                    df.set_index("timestamp")[["heart_rate", "spo2"]], height=260
                )
        with col_chart2:
            with st.container(border=True):
                st.markdown("##### 📈 Gia tốc Chuyển động cơ thể (3 trục X, Y, Z)")
                st.line_chart(
                    df.set_index("timestamp")[["accel_x", "accel_y", "accel_z"]],
                    height=260,
                )

        # --- KHỐI 4: XỬ LÝ TỰ ĐỘNG PHÂN TÍCH GEMINI & PHÁT HIỆN THỨC DẬY ---
        now_ts = time.time()

        # Tự động chốt báo cáo khi phát hiện thức dậy
        if (
            st.session_state.auto_wake_detect
            and current_lux > 50.0
            and mov_score >= 5
            and st.session_state.last_wake_report_db_id != latest_id
        ):
            df_full = get_data(limit=1000)
            report_data = build_report_from_dataframe(
                df_night=df_full if not df_full.empty else df,
                sleep_duration=7.5,
                api_key=st.session_state.gemini_api_key,
                report_label="Tự động chốt khi phát hiện thức dậy (ESP32)",
            )
            report_name = f"Báo cáo #{st.session_state.report_counter} (Tự động khi thức dậy)"
            st.session_state.reports[report_name] = report_data
            st.session_state.selected_report_name = report_name
            st.session_state.report_counter += 1
            st.session_state.last_wake_report_db_id = latest_id
            st.session_state.realtime_llm_report = report_data
            st.session_state.last_auto_analysis_ts = now_ts
            st.session_state.last_auto_analyzed_db_id = latest_id
            st.toast(
                f"🌅 Phát hiện thức dậy! Đã lưu '{report_name}' sang Tab 2 (Phân tích & Lời khuyên)."
            )

        # Tự động gọi Gemini định kỳ khi có dữ liệu mới
        elapsed = now_ts - st.session_state.last_auto_analysis_ts
        has_new_data = latest_id != st.session_state.last_auto_analyzed_db_id

        if (
            st.session_state.auto_gemini_enabled
            and has_new_data
            and (
                st.session_state.realtime_llm_report is None
                or elapsed >= st.session_state.auto_gemini_interval_sec
            )
        ):
            with st.spinner("🤖 Đang tự động cập nhật nhận xét từ Gemini AI..."):
                auto_rep = build_report_from_dataframe(
                    df_night=df,
                    sleep_duration=7.5,
                    api_key=st.session_state.gemini_api_key,
                    report_label="Phân tích Tự động Real-time (100 mẫu gần nhất)",
                )
                st.session_state.realtime_llm_report = auto_rep
                st.session_state.last_auto_analysis_ts = time.time()
                st.session_state.last_auto_analyzed_db_id = latest_id
                if "⚡ Bản phân tích Tự động (Real-time)" not in st.session_state.reports:
                    st.session_state.selected_report_name = "⚡ Bản phân tích Tự động (Real-time)"
                st.session_state.reports["⚡ Bản phân tích Tự động (Real-time)"] = auto_rep

        # Tóm tắt nhanh lời khuyên tự động ở cuối màn hình Real-time
        with st.container(border=True):
            c_title, c_action = st.columns([3.5, 1.5])
            with c_title:
                st.markdown("##### 🤖 Tóm tắt Nhận xét Tự động từ Gemini AI")
            with c_action:
                if st.button("⚡ Cập nhật ngay", use_container_width=True):
                    with st.spinner("Đang phân tích dữ liệu hiện tại với Gemini AI..."):
                        manual_rep = build_report_from_dataframe(
                            df_night=df,
                            sleep_duration=7.5,
                            api_key=st.session_state.gemini_api_key,
                            report_label="Cập nhật trực tiếp từ màn hình Real-time",
                        )
                        st.session_state.realtime_llm_report = manual_rep
                        st.session_state.last_auto_analysis_ts = time.time()
                        st.session_state.last_auto_analyzed_db_id = latest_id
                        st.session_state.reports["⚡ Bản phân tích Tự động (Real-time)"] = manual_rep
                        st.session_state.selected_report_name = "⚡ Bản phân tích Tự động (Real-time)"

            rt_rep = st.session_state.realtime_llm_report
            if rt_rep:
                rt_llm = rt_rep.get("llm_analysis", {})
                next_in = max(
                    0,
                    int(
                        st.session_state.auto_gemini_interval_sec
                        - (time.time() - st.session_state.last_auto_analysis_ts)
                    ),
                )
                st.caption(
                    f"🕒 Cập nhật lúc: **{rt_rep.get('analyzed_at', '')}**  |  "
                    f"⏱️ Làm mới tự động sau: **{next_in} giây** (khi có dữ liệu ESP32 mới)  |  "
                    f"👉 *Xem bản phân tích đầy đủ ở Tab **'🧠 2. Phân tích & Lời khuyên'***"
                )
                st.write(rt_llm.get("overall_assessment", ""))
            else:
                st.caption(
                    "Chưa có bản phân tích tự động. Hãy bật 'Tự động phân tích Gemini định kỳ' ở thanh bên trái hoặc bấm '⚡ Cập nhật ngay'."
                )

    show_realtime_content()


# -------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH GIẤC NGỦ & LỜI KHUYÊN CÁ NHÂN HÓA (GEMINI AI)
# -------------------------------------------------------------------------
with tab_analysis:
    # Thanh công cụ tạo báo cáo tổng kết từ toàn bộ dữ liệu trong SQLite
    with st.container(border=True):
        col_desc, col_dur, col_create = st.columns([3, 1, 1.5])
        with col_desc:
            st.markdown("##### 🌅 Tổng kết & Phân tích toàn bộ Phiên ngủ từ Database")
            st.caption(
                "Gom toàn bộ dữ liệu cảm biến đã thu thập trong `sleep_records.db` (tối đa 1000 mẫu gần nhất) để chấm điểm tổng kết và nhận tư vấn chuyên sâu từ Gemini AI."
            )
        with col_dur:
            sql_sleep_dur = st.number_input(
                "Thời lượng ngủ (giờ):",
                min_value=1.0,
                max_value=14.0,
                value=7.5,
                step=0.5,
            )
        with col_create:
            st.write("")  # Căn dòng
            if st.button(
                "📊 Tạo Báo cáo Tổng kết Mới",
                type="primary",
                use_container_width=True,
            ):
                df_night = get_data(limit=1000)
                if not df_night.empty:
                    with st.spinner(
                        "Đang tổng hợp dữ liệu cả đêm và gọi Gemini AI phân tích..."
                    ):
                        report_data = build_report_from_dataframe(
                            df_night=df_night,
                            sleep_duration=sql_sleep_dur,
                            api_key=st.session_state.gemini_api_key,
                            report_label="Tổng kết cả đêm từ SQLite (sleep_records.db)",
                        )
                        report_name = f"Báo cáo #{st.session_state.report_counter} (Tổng kết SQLite)"
                        st.session_state.reports[report_name] = report_data
                        st.session_state.selected_report_name = report_name
                        st.session_state.report_counter += 1
                        st.rerun()
                else:
                    st.warning("Chưa có dữ liệu trong SQLite để tạo báo cáo!")

    # Hiển thị báo cáo đã chọn
    report_keys = list(st.session_state.reports.keys())
    if not report_keys:
        st.info(
            "📭 **Hiện chưa có bản báo cáo chi tiết nào.**\n\n"
            "- Bạn có thể bấm nút **'📊 Tạo Báo cáo Tổng kết Mới'** ở ngay phía trên (nếu đã có dữ liệu trong Database).\n"
            "- Hoặc chuyển sang tab **'🧪 3. Giả lập Thiết bị (Test Demo)'** để chạy thử một kịch bản mẫu!"
        )
    else:
        # Đảm bảo selected_report_name hợp lệ
        if (
            st.session_state.selected_report_name not in report_keys
        ):
            st.session_state.selected_report_name = report_keys[-1]

        col_select, col_del = st.columns([3.5, 1.5])
        with col_select:
            chosen_rep_name = st.selectbox(
                "📂 Chọn bản báo cáo để xem chi tiết:",
                options=report_keys,
                index=report_keys.index(st.session_state.selected_report_name),
            )
            st.session_state.selected_report_name = chosen_rep_name
        with col_del:
            st.write("")
            if st.button("🗑️ Xóa báo cáo này", use_container_width=True):
                del st.session_state.reports[chosen_rep_name]
                remaining = list(st.session_state.reports.keys())
                st.session_state.selected_report_name = (
                    remaining[-1] if remaining else None
                )
                st.rerun()

        st.markdown("---")
        render_detailed_report(
            st.session_state.reports[st.session_state.selected_report_name],
            key_prefix=f"tab2_{st.session_state.selected_report_name}",
        )


# -------------------------------------------------------------------------
# TAB 3: GIẢ LẬP THIẾT BỊ (TEST DEMO KHI KHÔNG CÓ MẠCH ESP32)
# -------------------------------------------------------------------------
with tab_simulator:
    st.markdown("#### 🧪 Công cụ Giả lập Tín hiệu Cảm biến ESP32 (Dành cho Demo & Kiểm thử)")
    st.caption(
        "Giúp bạn tạo nhanh dữ liệu của 1 đêm ngủ theo các tình huống thực tế để kiểm thử mô hình Random Forest và xem cách Gemini AI phân tích, đưa ra lời khuyên."
    )

    with st.container(border=True):
        sim_mode = st.radio(
            "Chọn phương thức giả lập:",
            [
                "📋 Chọn từ 4 Kịch bản Giấc ngủ Mẫu (Khuyên dùng khi thuyết trình)",
                "🎛️ Tự điều chỉnh từng thông số cảm biến (Custom Sliders)",
            ],
            horizontal=True,
        )

        st.markdown("---")

        if "Kịch bản Giấc ngủ Mẫu" in sim_mode:
            selected_scenario_name = st.selectbox(
                "Chọn kịch bản giấc ngủ muốn thử nghiệm:",
                list(PRESET_SCENARIOS.keys()),
            )
            sc = PRESET_SCENARIOS[selected_scenario_name]
            st.info(f"📝 **Mô tả kịch bản:** {sc['description']}")

            sim_hr = sc["heart_rate"]
            sim_spo2 = sc["spo2"]
            sim_temp = sc["temperature"]
            sim_hum = sc["humidity"]
            sim_lux = sc["light_lux"]
            sim_dur = sc["sleep_duration"]
            sim_mov = sc["movement_intensity"]
            sim_hrv = sc["hr_variability"]
            sim_label = selected_scenario_name

            # Hiển thị tóm tắt thông số của kịch bản mẫu (chia 2 hàng x 3 cột để không bị cắt chữ)
            s1, s2, s3 = st.columns(3)
            s1.metric("❤️ Nhịp tim mẫu", f"{sim_hr} BPM")
            s2.metric("🩸 SpO2 mẫu", f"{sim_spo2}%")
            s3.metric("⏳ Thời lượng ngủ", f"{sim_dur} giờ")

            s4, s5, s6 = st.columns(3)
            s4.metric("🌡️ Nhiệt độ phòng", f"{sim_temp}°C")
            s5.metric("💧 Độ ẩm phòng", f"{sim_hum}%")
            s6.metric("💡 Ánh sáng", f"{sim_lux} Lux")

        else:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("##### 🫀 Thiết lập Chỉ số Sinh lý & Vận động")
                sim_hr = st.slider(
                    "Nhịp tim trung bình (BPM)", 45.0, 120.0, 78.0, 1.0
                )
                sim_spo2 = st.slider(
                    "Nồng độ Oxy máu SpO2 (%)", 85.0, 100.0, 96.0, 0.5
                )
                sim_dur = st.slider(
                    "Thời lượng ngủ (Giờ)", 3.0, 10.0, 6.0, 0.5
                )
                mov_choice = st.selectbox(
                    "Mức độ cử động cơ thể (MPU6050):",
                    [
                        "Ngủ im lìm (Ít cử động)",
                        "Chuyển động vừa phải",
                        "Trằn trọc / Thức nhiều",
                    ],
                    index=2,
                )
                sim_mov = (
                    "low"
                    if "Ngủ im lìm" in mov_choice
                    else ("medium" if "vừa phải" in mov_choice else "high")
                )

            with c_right:
                st.markdown("##### 🏠 Thiết lập Môi trường Phòng ngủ & Thần kinh")
                sim_temp = st.slider(
                    "Nhiệt độ phòng (°C)", 16.0, 36.0, 29.5, 0.5
                )
                sim_hum = st.slider("Độ ẩm phòng (%)", 30.0, 95.0, 74.0, 1.0)
                sim_lux = st.slider(
                    "Cường độ ánh sáng (Lux)", 0.0, 300.0, 45.0, 5.0
                )
                stress_choice = st.select_slider(
                    "Độ biến thiên nhịp tim (HRV / Mức độ thư giãn):",
                    options=[
                        "Căng thẳng cao (HRV thấp)",
                        "Bình thường",
                        "Thư giãn sâu (HRV cao)",
                    ],
                    value="Căng thẳng cao (HRV thấp)",
                )
                sim_hrv = (
                    1.5
                    if "Căng thẳng" in stress_choice
                    else (3.5 if "Bình thường" in stress_choice else 5.5)
                )

            sim_label = "Giả lập Tùy chỉnh (Custom Sliders)"

        st.markdown("---")
        col_opt, col_run = st.columns([3, 2])
        with col_opt:
            write_to_db = st.checkbox(
                "📥 Đồng thời lưu dữ liệu giả lập này vào SQLite (để Tab 1 Real-time cũng hiển thị dữ liệu này)",
                value=True,
            )
        with col_run:
            run_sim_btn = st.button(
                "🚀 Chạy Giả lập & Phân tích bằng Gemini AI",
                type="primary",
                use_container_width=True,
            )

    if run_sim_btn:
        with st.spinner(
            "Đang sinh chuỗi dữ liệu cảm biến giả lập, chấm điểm Random Forest và gọi Gemini AI..."
        ):
            df_sim = generate_simulated_night_df(
                heart_rate=sim_hr,
                spo2=sim_spo2,
                temperature=sim_temp,
                humidity=sim_hum,
                light_lux=sim_lux,
                movement_intensity=sim_mov,
                hr_variability=sim_hrv,
                num_points=60,
            )
            if write_to_db:
                save_df_to_sqlite(df_sim)

            report_data = build_report_from_dataframe(
                df_night=df_sim,
                sleep_duration=sim_dur,
                api_key=st.session_state.gemini_api_key,
                report_label=sim_label,
            )
            report_name = f"Báo cáo #{st.session_state.report_counter} ({sim_label[:28]})"
            st.session_state.reports[report_name] = report_data
            st.session_state.selected_report_name = report_name
            st.session_state.realtime_llm_report = report_data
            st.session_state.report_counter += 1

        st.success(
            f"✅ Đã tạo xong **{report_name}**! Kết quả phân tích chi tiết hiển thị ngay bên dưới (và cũng đã được lưu vào **Tab 2**)."
        )

    # Hiển thị ngay kết quả báo cáo vừa chạy giả lập ở bên dưới để tiện theo dõi
    if st.session_state.selected_report_name and st.session_state.selected_report_name in st.session_state.reports:
        st.markdown("---")
        st.markdown(f"### 📋 Kết quả Phân tích gần nhất: `{st.session_state.selected_report_name}`")
        render_detailed_report(
            st.session_state.reports[st.session_state.selected_report_name],
            key_prefix="tab3_latest",
        )
