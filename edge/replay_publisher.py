"""replay_publisher.py

Script giả lập phát lại dữ liệu cảm biến từ file CSV lên MQTT Broker.
"""

from datetime import datetime, timezone
import json
import os
import time
import paho.mqtt.client as mqtt
import pandas as pd

BROKER_HOST = "localhost"  # Mosquitto broker
BROKER_PORT = 1883
TOPIC_SENSOR = "aiot/sleep/sensors"
DEVICE_ID = "simulated_sleep_esp32"
DELAY_SECONDS = 3

# Tìm file dữ liệu sạch
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/data_cleaned.csv"
)
if not os.path.exists(CSV_PATH):
  CSV_PATH = "data_cleaned.csv"

print(f"[*] Đang nạp dữ liệu từ: {CSV_PATH}")
try:
  df = pd.read_csv(CSV_PATH)
  print(f"[+] Đọc thành công {len(df)} dòng dữ liệu!")
except Exception as e:
  print(f"[!] Lỗi đọc CSV: {e}")
  exit(1)

# Khởi tạo MQTT Client tương thích cả paho v1 & v2
try:
  client = mqtt.Client(
      callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=DEVICE_ID
  )
except AttributeError:
  client = mqtt.Client(client_id=DEVICE_ID)


def on_connect(c, userdata, flags, rc, properties=None):
  if rc == 0:
    print(f"[+] Kết nối thành công tới MQTT Broker: {BROKER_HOST}")
  else:
    print(f"[!] Lỗi kết nối MQTT: {rc}")


client.on_connect = on_connect

try:
  client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
  client.loop_start()
except Exception as e:
  print(f"[!] Không kết nối được Mosquitto Broker ({BROKER_HOST}:{BROKER_PORT})")
  print("    -> Bạn nhớ kiểm tra xem dịch vụ mosquitto đã bật chưa nhé!")
  exit(1)

print(f"\n[>>>] BẮT ĐẦU PHÁT DỮ LIỆU LÊN TOPIC: {TOPIC_SENSOR}")
print("[*] Nhấn Ctrl + C để dừng bất kỳ lúc nào.\n")

try:
  while True:
    for index, row in df.iterrows():
      now_iso = (
          datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
      )
      act_val = float(row.get("Physical Activity Level", 50))
      movement_norm = round(act_val / 100.0, 2)

      payload = {
          "device_id": DEVICE_ID,
          "timestamp": now_iso,
          "heart_rate_bpm": int(row.get("Heart Rate", 72)),
          "spo2_percent": 98,
          "movement": movement_norm,
          "temperature_c": 26.5,
          "humidity_percent": 60.0,
      }

      msg = json.dumps(payload, ensure_ascii=False)
      client.publish(TOPIC_SENSOR, msg)
      print(f"[{now_iso}] [Dòng {index + 1}/{len(df)}] Đã gửi: {msg}")

      time.sleep(DELAY_SECONDS)

except KeyboardInterrupt:
  print("\n[*] Đã dừng chương trình.")
finally:
  client.loop_stop()
  client.disconnect()