"""sleep_rules.py

Module đánh giá điểm chất lượng giấc ngủ và tạo khuyến nghị cải thiện.
Dựa trên luật if-then kết hợp các chỉ số môi trường (DHT22) và sinh lý (MAX30102).
"""


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