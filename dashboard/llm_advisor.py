"""llm_advisor.py

Module tích hợp Google Gemini API để:
1. Đưa ra đánh giá tổng quan chất lượng giấc ngủ và phân tích các yếu tố chính ảnh hưởng.
2. Cung cấp các khuyến nghị cá nhân hóa nhằm cải thiện thói quen và môi trường ngủ.
3. Hỗ trợ sinh dữ liệu giả lập (Simulator) theo kịch bản để test demo khi không có ESP32.
"""

from datetime import datetime, timedelta
import json
import os
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Các kịch bản giả lập mẫu để demo nhanh trên Dashboard
PRESET_SCENARIOS = {
    "🌟 Kịch bản 1: Giấc ngủ sâu lý tưởng": {
        "description": "Phòng mát mẻ, tối hoàn toàn, nhịp tim ổn định, SpO2 cao và hầu như không trằn trọc.",
        "heart_rate": 64.0,
        "spo2": 98.0,
        "temperature": 25.0,
        "humidity": 55.0,
        "light_lux": 2.0,
        "sleep_duration": 8.0,
        "movement_intensity": "low",  # Ngủ im lìm
        "hr_variability": 5.5,        # HRV tốt -> Stress thấp
    },
    "🔥 Kịch bản 2: Ngủ trằn trọc do phòng nóng & sáng": {
        "description": "Nhiệt độ phòng cao (30.5°C), độ ẩm cao (78%) và đèn ngủ sáng (85 lux) khiến cơ thể trằn trọc.",
        "heart_rate": 82.0,
        "spo2": 96.5,
        "temperature": 30.5,
        "humidity": 78.0,
        "light_lux": 85.0,
        "sleep_duration": 5.5,
        "movement_intensity": "high",  # Trằn trọc nhiều
        "hr_variability": 3.0,
    },
    "😰 Kịch bản 3: Stress cao & Nhịp tim nhanh (Dùng Caffeine/Thức khuya)": {
        "description": "Môi trường phòng ổn nhưng nhịp tim khi ngủ cao (92 BPM) và biến thiên nhịp tim thấp (Stress cao).",
        "heart_rate": 92.0,
        "spo2": 97.0,
        "temperature": 26.0,
        "humidity": 58.0,
        "light_lux": 8.0,
        "sleep_duration": 5.0,
        "movement_intensity": "medium",
        "hr_variability": 1.2,        # HRV rất thấp -> Stress cao
    },
    "⚠️ Kịch bản 4: Nghi ngờ ngáy / Ngưng thở khi ngủ (SpO2 sụt giảm)": {
        "description": "SpO2 trung bình giảm xuống 91.5%, nhịp tim dao động mạnh kèm các đợt giật mình chuyển động.",
        "heart_rate": 86.0,
        "spo2": 91.5,
        "temperature": 25.5,
        "humidity": 62.0,
        "light_lux": 5.0,
        "sleep_duration": 6.0,
        "movement_intensity": "high",
        "hr_variability": 2.0,
    },
}


def load_api_key_from_env():
    """Đọc GEMINI_API_KEY từ biến môi trường hoặc file .env ở thư mục gốc."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "GEMINI_API_KEY":
                        return v.strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def generate_simulated_night_df(
    heart_rate=70.0,
    spo2=97.0,
    temperature=26.0,
    humidity=60.0,
    light_lux=5.0,
    movement_intensity="low",
    hr_variability=4.5,
    num_points=60,
):
    """Sinh DataFrame chuỗi thời gian cảm biến giả lập tự nhiên cho 1 đêm ngủ."""
    rng = np.random.default_rng()
    now = datetime.now()
    timestamps = [
        (now - timedelta(seconds=(num_points - i) * 2)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(num_points)
    ]

    # Sinh nhịp tim có độ lệch chuẩn tương ứng với hr_variability
    hr_series = np.clip(
        rng.normal(loc=heart_rate, scale=max(0.8, hr_variability), size=num_points),
        45.0,
        150.0,
    ).round(1)

    # Sinh SpO2
    spo2_series = np.clip(
        rng.normal(loc=spo2, scale=0.8, size=num_points),
        82.0,
        100.0,
    ).round(0).astype(int)

    # Sinh nhiệt độ, độ ẩm, ánh sáng có dao động nhỏ tự nhiên
    temp_series = np.clip(
        rng.normal(loc=temperature, scale=0.2, size=num_points), 15.0, 40.0
    ).round(1)
    hum_series = np.clip(
        rng.normal(loc=humidity, scale=0.8, size=num_points), 20.0, 98.0
    ).round(1)
    lux_series = np.clip(
        rng.normal(loc=light_lux, scale=max(0.5, light_lux * 0.05), size=num_points),
        0.0,
        1000.0,
    ).round(1)

    # Sinh gia tốc MPU6050 (accel_x, accel_y, accel_z) dựa theo mức độ vận động
    base_x, base_y, base_z = -0.5, 0.2, 9.7
    if movement_intensity == "low":
        noise_scale = 0.08
        spike_prob = 0.0
    elif movement_intensity == "medium":
        noise_scale = 0.25
        spike_prob = 0.08
    else:  # high (Trằn trọc / Thức)
        noise_scale = 0.6
        spike_prob = 0.30

    accel_x = rng.normal(base_x, noise_scale, num_points)
    accel_y = rng.normal(base_y, noise_scale, num_points)
    accel_z = rng.normal(base_z, noise_scale, num_points)

    for i in range(num_points):
        if rng.random() < spike_prob:
            accel_x[i] += rng.choice([-1, 1]) * rng.uniform(1.8, 4.0)
            accel_y[i] += rng.choice([-1, 1]) * rng.uniform(1.8, 4.0)
            accel_z[i] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.5)

    return pd.DataFrame({
        "timestamp": timestamps,
        "accel_x": np.round(accel_x, 3),
        "accel_y": np.round(accel_y, 3),
        "accel_z": np.round(accel_z, 3),
        "light_lux": lux_series,
        "heart_rate": hr_series,
        "spo2": spo2_series,
        "temperature": temp_series,
        "humidity": hum_series,
    })


def build_fallback_analysis(metrics, rule_tips):
    """Phân tích dự phòng chi tiết khi chưa có Gemini API Key hoặc lỗi kết nối."""
    score = metrics["score"]
    rating = metrics["rating"]
    hr = metrics["hr"]
    spo2 = metrics["spo2"]
    stress = metrics["stress"]
    mov_score = metrics["mov_score"]
    mov_status = metrics["mov_status"]
    temp = metrics["temp"]
    hum = metrics["hum"]
    lux = metrics["lux"]
    duration = metrics.get("sleep_duration", 6.0)

    overall = (
        f"Giấc ngủ của bạn đạt **{score:.1f}/100 điểm** (Xếp loại: **{rating}**). "
        f"Trong thời gian theo dõi (~{duration:.1f} giờ), nhịp tim trung bình đạt {hr:.1f} BPM, "
        f"SpO2 đạt {spo2:.1f}%, mức độ căng thẳng (Stress) ở mức {stress}/10 và trạng thái vận động là '{mov_status}' ({mov_score}/10)."
    )

    key_factors = []
    # 1. Yếu tố sinh lý (Nhịp tim, SpO2, Stress)
    if spo2 < 94:
        key_factors.append({
            "factor": "Hô hấp & Nồng độ Oxy máu (SpO2)",
            "impact": "Tiêu cực",
            "analysis": f"SpO2 trung bình giảm xuống {spo2:.1f}% (< 94%), là yếu tố chính kéo giảm chất lượng giấc ngủ và làm tăng chỉ số Stress ({stress}/10)."
        })
    elif hr > 85 or stress >= 7:
        key_factors.append({
            "factor": "Nhịp tim & Mức độ Căng thẳng thần kinh",
            "impact": "Tiêu cực",
            "analysis": f"Nhịp tim khi ngủ khá cao ({hr:.1f} BPM) kèm điểm Stress {stress}/10 khiến hệ thần kinh khó chuyển sang giai đoạn ngủ sâu."
        })
    else:
        key_factors.append({
            "factor": "Chỉ số Sinh lý (Nhịp tim & SpO2)",
            "impact": "Tích cực",
            "analysis": f"Nhịp tim ({hr:.1f} BPM) và SpO2 ({spo2:.1f}%) duy trì ở ngưỡng sinh lý ổn định, hỗ trợ cơ thể phục hồi năng lượng."
        })

    # 2. Yếu tố vận động
    if mov_score >= 3:
        key_factors.append({
            "factor": "Chuyển động cơ thể (MPU6050)",
            "impact": "Tiêu cực",
            "analysis": f"Ghi nhận điểm vận động {mov_score}/10 ({mov_status}), cho thấy bạn xoay người hoặc trằn trọc nhiều lần trong đêm."
        })
    else:
        key_factors.append({
            "factor": "Chuyển động cơ thể (MPU6050)",
            "impact": "Tích cực",
            "analysis": f"Điểm vận động thấp ({mov_score}/10 - {mov_status}), phản ánh cơ thể nằm yên và duy trì giấc ngủ liên tục."
        })

    # 3. Yếu tố môi trường
    env_issues = []
    if temp > 28.0:
        env_issues.append(f"nhiệt độ cao ({temp:.1f}°C)")
    elif temp < 20.0:
        env_issues.append(f"nhiệt độ lạnh ({temp:.1f}°C)")
    if hum > 70.0:
        env_issues.append(f"độ ẩm cao ({hum:.1f}%)")
    elif hum < 40.0:
        env_issues.append(f"không khí khô ({hum:.1f}%)")
    if lux > 20.0:
        env_issues.append(f"ánh sáng mạnh ({lux:.1f} lux)")

    if env_issues:
        key_factors.append({
            "factor": "Môi trường phòng ngủ (Nhiệt độ, Độ ẩm, Ánh sáng)",
            "impact": "Cần chú ý",
            "analysis": "Phát hiện yếu tố môi trường chưa tối ưu: " + ", ".join(env_issues) + ". Đây có thể là nguyên nhân ngoại cảnh gây trằn trọc và tăng nhịp tim."
        })
    else:
        key_factors.append({
            "factor": "Môi trường phòng ngủ (Nhiệt độ, Độ ẩm, Ánh sáng)",
            "impact": "Tích cực",
            "analysis": f"Nhiệt độ ({temp:.1f}°C), độ ẩm ({hum:.1f}%) và ánh sáng ({lux:.1f} lux) đều nằm trong vùng lý tưởng cho giấc ngủ."
        })

    env_recs = []
    habit_recs = []

    if temp > 28.0:
        env_recs.append(f"Hạ nhiệt độ điều hòa hoặc bật quạt thông gió để đưa nhiệt độ phòng từ {temp:.1f}°C về vùng lý tưởng 24–26°C.")
    elif temp < 20.0:
        env_recs.append(f"Phòng khá lạnh ({temp:.1f}°C), nên đóng bớt cửa lùa và sử dụng chăn giữ nhiệt phù hợp.")
    if hum > 70.0:
        env_recs.append(f"Độ ẩm {hum:.1f}% khá cao gây bí bách; nên bật chế độ hút ẩm (Dry) hoặc mở thoáng cửa trước khi ngủ 30 phút.")
    elif hum < 40.0:
        env_recs.append(f"Độ ẩm {hum:.1f}% hơi khô; nên đặt chậu nước nhỏ hoặc bật máy tạo ẩm nhẹ để tránh khô mũi họng.")
    if lux > 20.0:
        env_recs.append(f"Cường độ sáng {lux:.1f} lux ức chế tiết Melatonin; hãy tắt đèn trần, kéo rèm cản sáng hoặc chỉ dùng đèn ngủ ánh sáng vàng dưới 5 lux.")
    if not env_recs:
        env_recs.append("Duy trì cấu hình nhiệt độ (24–26°C), độ ẩm (50–65%) và không gian tối như hiện tại.")

    if hr > 85 or stress >= 6:
        habit_recs.append("Tránh sử dụng cà phê, trà đậm, nước tăng lực sau 15:00 và không tập thể dục cường độ cao trong vòng 2 giờ trước khi ngủ.")
        habit_recs.append("Thực hiện bài tập thở 4-7-8 hoặc giãn cơ nhẹ 10 phút trước khi lên giường để hạ nhịp tim và giảm chỉ số Stress.")
    if spo2 < 94:
        habit_recs.append("Thử nằm nghiêng sang một bên và kê gối vừa phải để mở rộng đường thở, hạn chế ngủ ngửa gây ngáy hoặc tụt SpO2.")
    if mov_score >= 3:
        habit_recs.append("Tắt điện thoại/màn hình xanh ít nhất 30 phút trước khi ngủ để não bộ dễ đi vào giấc ngủ sâu, giảm trằn trọc.")
    if not habit_recs:
        habit_recs.append("Duy trì giờ đi ngủ và thức dậy cố định mỗi ngày để củng cố nhịp sinh học tự nhiên.")

    for tip in rule_tips:
        if tip not in env_recs and tip not in habit_recs:
            habit_recs.append(tip)

    return {
        "source": "rule_fallback",
        "overall_assessment": overall,
        "key_factors": key_factors,
        "environment_recommendations": env_recs,
        "habit_recommendations": habit_recs,
        "error_detail": None,
    }


def analyze_sleep_with_gemini(metrics, rule_tips=None, api_key=None, model_name="gemini-2.5-flash"):
    """Gửi dữ liệu giấc ngủ lên Google Gemini API để phân tích chuyên sâu & cá nhân hóa.

    Tham số:
      metrics (dict): Các chỉ số giấc ngủ (score, rating, hr, spo2, stress, mov_score,
                      mov_status, temp, hum, lux, sleep_duration).
      rule_tips (list): Các gợi ý cơ bản từ hệ luật if-else.
      api_key (str): Google Gemini API Key.
      model_name (str): Tên model Gemini (mặc định: gemini-2.5-flash).

    Trả về:
      dict: Cấu trúc đánh giá chất lượng, phân tích nhân tố chính và khuyến nghị cá nhân hóa.
    """
    rule_tips = rule_tips or []
    resolved_key = (api_key or load_api_key_from_env()).strip()

    if not resolved_key:
        fallback = build_fallback_analysis(metrics, rule_tips)
        fallback["error_detail"] = "Chưa cung cấp GEMINI_API_KEY (Đang hiển thị bản phân tích từ Hệ chuyên gia dự phòng)."
        return fallback

    prompt = f"""Bạn là một Chuyên gia Y học Giấc ngủ và Kỹ sư AIoT. Hãy phân tích dữ liệu giấc ngủ thu thập từ hệ thống cảm biến (MAX30100, MPU6050, DHT22, BH1750) và mô hình Machine Learning dưới đây để tư vấn cho người dùng bằng tiếng Việt dễ hiểu, khoa học và cá nhân hóa.

DỮ LIỆU GIẤC NGỦ TỔNG HỢP:
- Điểm chất lượng giấc ngủ (AI Random Forest dự đoán): {metrics['score']:.1f} / 100
- Xếp loại: {metrics['rating']}
- Thời lượng ngủ ước tính: {metrics.get('sleep_duration', 6.0):.1f} giờ
- Nhịp tim trung bình (MAX30102): {metrics['hr']:.1f} BPM
- Nồng độ Oxy trong máu SpO2 trung bình (MAX30102): {metrics['spo2']:.1f} %
- Mức độ căng thẳng / Stress Level (tính từ HRV & SpO2): {metrics['stress']} / 10
- Điểm vận động cơ thể (MPU6050 Actigraphy): {metrics['mov_score']} / 10 (Trạng thái: {metrics['mov_status']})
- Nhiệt độ phòng ngủ trung bình (DHT22): {metrics['temp']:.1f} °C
- Độ ẩm phòng ngủ trung bình (DHT22): {metrics['hum']:.1f} %
- Cường độ ánh sáng phòng ngủ (BH1750): {metrics['lux']:.1f} Lux

YÊU CẦU ĐẦU RA:
Hãy trả về DUY NHẤT một chuỗi JSON hợp lệ (không bọc trong markdown ```json) theo đúng cấu trúc sau:
{{
  "overall_assessment": "Đoạn văn (3-4 câu) đánh giá tổng quan chất lượng giấc ngủ, giải thích ý nghĩa điểm số {metrics['score']:.1f}/100 và tóm tắt bức tranh chung của đêm ngủ.",
  "key_factors": [
    {{
      "factor": "Tên yếu tố (ví dụ: Nhiệt độ & Độ ẩm phòng ngủ / Nhịp tim & Mức độ Stress / Chuyển động cơ thể / Nồng độ Oxy máu SpO2)",
      "impact": "Chỉ nhận 1 trong 3 giá trị: Tích cực | Cần chú ý | Tiêu cực",
      "analysis": "Phân tích sâu mối liên hệ nhân quả giữa các chỉ số cảm biến (ví dụ nhiệt độ/ánh sáng tác động thế nào tới sự trằn trọc và nhịp tim của người dùng kèm số liệu cụ thể)."
    }}
  ],
  "environment_recommendations": [
    "Khuyến nghị cụ thể 1 để tối ưu môi trường phòng ngủ (nhiệt độ, độ ẩm, ánh sáng, thông khí) dựa trên số liệu thực tế",
    "Khuyến nghị cụ thể 2..."
  ],
  "habit_recommendations": [
    "Khuyến nghị cụ thể 1 để cải thiện thói quen trước khi ngủ, tư thế ngủ hoặc thư giãn thần kinh dựa trên nhịp tim, SpO2, mức độ stress và vận động",
    "Khuyến nghị cụ thể 2..."
  ]
}}
"""

    is_openrouter = resolved_key.startswith("sk-or-")

    if is_openrouter:
        candidate_models = []
        if model_name:
            or_m = model_name if "/" in model_name else f"google/{model_name}"
            candidate_models.append(or_m)
        for backup_m in [
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-lite-001",
        ]:
            if backup_m not in candidate_models:
                candidate_models.append(backup_m)
    else:
        candidate_models = [model_name]
        for backup_m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            if backup_m not in candidate_models:
                candidate_models.append(backup_m)

    last_error = None
    for current_model in candidate_models:
        if is_openrouter:
            url = "https://openrouter.ai/api/v1/chat/completions"
            payload_dict = {
                "model": current_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {resolved_key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Sleep AIoT Monitor",
            }
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={resolved_key}"
            payload_dict = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "responseMimeType": "application/json",
                },
            }
            headers = {"Content-Type": "application/json"}

        payload_bytes = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                if is_openrouter:
                    text_out = (
                        resp_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    ).strip()
                else:
                    text_out = (
                        resp_data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    ).strip()

                if text_out.startswith("```"):
                    text_out = text_out.strip("`")
                    if text_out.lower().startswith("json"):
                        text_out = text_out[4:].strip()

                # Trích xuất khối JSON nếu có ký tự thừa xung quanh
                start_idx = text_out.find("{")
                end_idx = text_out.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    text_out = text_out[start_idx : end_idx + 1]

                parsed = json.loads(text_out)
                source_label = f"gemini via OpenRouter ({current_model})" if is_openrouter else f"gemini ({current_model})"
                return {
                    "source": source_label,
                    "overall_assessment": parsed.get("overall_assessment", ""),
                    "key_factors": parsed.get("key_factors", []),
                    "environment_recommendations": parsed.get("environment_recommendations", []),
                    "habit_recommendations": parsed.get("habit_recommendations", []),
                    "error_detail": None,
                }
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
                err_json = json.loads(err_body)
                err_msg = err_json.get("error", {}).get("message", err_body)
            except Exception:
                err_msg = str(e)
            last_error = f"HTTP {e.code} ({current_model}): {err_msg}"
            # Nếu lỗi 400/401/403 do API key không hợp lệ thì dừng luôn không thử model khác
            if e.code in (400, 401, 403) and ("API key" in err_msg or "auth" in err_msg.lower() or "key" in err_msg.lower()):
                break
        except Exception as e:
            last_error = f"{current_model}: {str(e)}"

    fallback = build_fallback_analysis(metrics, rule_tips)
    fallback["error_detail"] = f"Không gọi được Gemini API ({last_error}). Đã chuyển sang chế độ phân tích dự phòng."
    return fallback
