"""sleep_rules.py

Module đánh giá điểm chất lượng giấc ngủ và tạo khuyến nghị cải thiện.
Dựa trên luật if-then kết hợp các chỉ số môi trường (DHT22) và sinh lý (MAX30102).
"""
import numpy as np
import pandas as pd

def evaluate_sleep_quality(score, temp_c=None, hum_pct=None, hr_bpm=None):
  """Đánh giá chất lượng giấc ngủ và sinh khuyến nghị.

  Tham số:
    score (float): Điểm chất lượng giấc ngủ thử nghiệm (thang 0 - 100).
    temp_c (float, optional): Nhiệt độ phòng ngủ (°C).
    hum_pct (float, optional): Độ ẩm phòng ngủ (%).
    hr_bpm (int/float, optional): Nhịp tim trung bình khi ngủ (bpm).

  Trả về:
    dict: Gồm 'rating' (Xếp loại), 'score' (Điểm), và 'recommendations' (Danh
    sách gợi ý).
  """
  # 1. Phân loại mức độ chất lượng giấc ngủ theo thang điểm 0 - 100
  if score >= 80:
    rating = "Tốt (Good)"
  elif score >= 60:
    rating = "Trung bình (Fair)"
  else:
    rating = "Kém (Poor)"

  recommendations = []

  # 2. Quy tắc đánh giá môi trường phòng ngủ (từ cảm biến DHT22)
  if temp_c is not None:
    if temp_c > 28.0:
      recommendations.append(
          f"Nhiệt độ phòng hiện tại khá cao ({temp_c:.1f}°C). Nên bật điều"
          " hòa/quạt hạ xuống mức 24–26°C để dễ vào giấc sâu hơn."
      )
    elif temp_c < 20.0:
      recommendations.append(
          f"Nhiệt độ phòng hơi lạnh ({temp_c:.1f}°C). Bạn nên đắp thêm chăn ấm"
          " hoặc đóng bớt cửa sổ."
      )

  if hum_pct is not None:
    if hum_pct > 70.0:
      recommendations.append(
          f"Độ ẩm phòng cao ({hum_pct:.1f}%). Không khí ẩm có thể gây ngột"
          " ngạt, nên mở hé cửa thông gió hoặc dùng chế độ hút ẩm (Dry)."
      )
    elif hum_pct < 40.0:
      recommendations.append(
          f"Độ ẩm phòng quá thấp ({hum_pct:.1f}%). Không khí khô dễ gây khô"
          " họng, bạn nên bật máy phun sương tạo ẩm."
      )

  # 3. Quy tắc nhịp tim nghỉ ngơi khi ngủ (từ cảm biến MAX30102)
  if hr_bpm is not None:
    if hr_bpm > 85:
      recommendations.append(
          f"Nhịp tim khi ngủ ở mức cao ({int(hr_bpm)} bpm). Cần tránh sử dụng"
          " caffeine, thức uống có cồn hoặc vận động mạnh sát giờ ngủ."
      )
    elif hr_bpm < 50:
      recommendations.append(
          f"Nhịp tim khi ngủ khá thấp ({int(hr_bpm)} bpm). Nếu kèm triệu chứng"
          " mệt mỏi vào ban ngày, bạn nên theo dõi kỹ hơn."
      )

  # 4. Nếu mọi thông số đều ở ngưỡng lý tưởng
  if not recommendations:
    recommendations.append(
        "Môi trường phòng ngủ và nhịp tim nghỉ ngơi đều ở trạng thái lý tưởng."
        " Hãy tiếp tục duy trì!"
    )

  return {"score": round(score, 1), "rating": rating, "tips": recommendations}


# --- Khối code kiểm thử nhanh khi chạy trực tiếp file này ---
if __name__ == "__main__":
  # Thử nghiệm với một trường hợp mẫu
  test_result = evaluate_sleep_quality(
      score=65.5, temp_c=29.2, hum_pct=72.0, hr_bpm=88
  )

  print("=== KẾT QUẢ KIỂM THỬ SLEEP RULES ===")
  print(f"Điểm số: {test_result['score']} / 100")
  print(f"Xếp loại: {test_result['rating']}")
  print("Khuyến nghị:")
  for i, tip in enumerate(test_result["tips"], 1):
    print(f"  {i}. {tip}")
    
# tính Stress Level (Dựa trên Biến thiên nhịp tim - HRV và SpO2)
def calculate_stress_level(heart_rates, spo2_levels):
    """
    Tính điểm Stress Level (thang 1-10) từ mảng nhịp tim và SpO2 thực tế.
    """
    # Lọc bỏ các giá trị lỗi từ cảm biến (0 hoặc quá vô lý)
    valid_hr = [hr for hr in heart_rates if 40 < hr < 180]
    valid_spo2 = [spo2 for spo2 in spo2_levels if 50 < spo2 <= 100]
    
    if len(valid_hr) < 5 or len(valid_spo2) < 5:
        return 5 # Trả về mức trung bình nếu thiếu dữ liệu
        
    # Tính Biến thiên nhịp tim (HRV) thông qua độ lệch chuẩn
    hr_std = np.std(valid_hr)
    avg_spo2 = np.mean(valid_spo2)
    
    # Logic 1: HRV càng cao -> Cơ thể càng thư giãn -> Stress thấp
    # Logic 2: HRV càng thấp -> Cơ thể như cỗ máy -> Stress cao
    # Giả sử hr_std lý tưởng khi ngủ là ~5.0
    stress_score = 10 - (hr_std * 1.5)
    
    # Phạt điểm nặng nếu SpO2 trung bình sụt giảm (dấu hiệu ngáy/ngưng thở gây stress cho não)
    if avg_spo2 < 95:
        stress_score += (95 - avg_spo2) * 2.0
        
    # Ép giới hạn điểm chuẩn vào khung từ 1 đến 10
    stress_score = max(1, min(10, stress_score))
    
    return round(stress_score)
  
# phân tích Điểm vận động từ MPU6050 (Actigraphy Score)
def calculate_movement_score(df_xyz):
    """
    Đầu vào: DataFrame chứa 3 cột 'accel_x', 'accel_y', 'accel_z' của MPU6050 trong 1 phút.
    Đầu ra: Điểm vận động (0-10) và Nhãn trạng thái.
    """
    if df_xyz.empty or len(df_xyz) < 5:
        return 0, "Không đủ dữ liệu"

    # Bước 1: Tính gia tốc tổng hợp để loại bỏ góc nghiêng tĩnh (trọng lực)
    magnitudes = np.sqrt(
        df_xyz['accel_x']**2 + 
        df_xyz['accel_y']**2 + 
        df_xyz['accel_z']**2
    )
    
    # Bước 2: Lấy đạo hàm (tính sự chênh lệch) giữa các điểm thời gian liên tiếp
    # Cơ thể nằm im -> delta xấp xỉ 0.
    deltas = np.abs(np.diff(magnitudes))
    
    # Ngưỡng lọc nhiễu nhịp thở và rung động môi trường (Có thể tinh chỉnh thực tế)
    THRESHOLD = 1.0
    
    # Đếm số lần cử động vượt ngưỡng
    movement_spikes = np.sum(deltas > THRESHOLD)
    # Tính tổng bình phương gia tốc của các cử động đó (Năng lượng vận động)
    movement_energy = np.sum(deltas[deltas > THRESHOLD]**2)

    # Bước 3: Đánh giá trạng thái
    if movement_spikes == 0:
        return 0, "Ngủ im lìm"
        
    elif 1 <= movement_spikes <= 3 and movement_energy < 5.0:
        # Cử động rời rạc, năng lượng thấp -> Chuyển động vô thức khi ngủ sâu/REM
        return 2, "Chuyển động vô thức"
        
    else:
        # Cử động liên tục hoặc năng lượng rất lớn -> Đang trằn trọc/Thức
        # Quy đổi thành thang điểm 10
        score = min(10, int(movement_energy * 0.2))
        # Nếu đã có cử động mạnh, bét nhất cũng phải tính 3 điểm
        score = max(3, score) 
        return score, "Trằn trọc / Thức"

# Ví dụ cách bạn gọi hàm này với data từ SQLite (đã query vào DataFrame df_sensor_1min):
# score, status = calculate_movement_score(df_sensor_1min[['accel_x', 'accel_y', 'accel_z']])
# print(f"Điểm: {score}, Trạng thái: {status}")    
