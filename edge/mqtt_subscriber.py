import paho.mqtt.client as mqtt
import os
import sqlite3
import json
from datetime import datetime

# ================= CẤU HÌNH MQTT =================
MQTT_BROKER = "192.168.137.1"  # Thay bằng IP máy tính của bạn (giống IP trong config.h)
MQTT_PORT = 1883
MQTT_TOPIC = "sleep_aiot/body"

# Lấy thư mục gốc của project (lùi lại 1 cấp từ thư mục 'edge')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Trỏ chính xác vào data/database/sleep_records.db
DB_NAME = os.path.join(BASE_DIR, "data", "database", "sleep_records.db")

def init_db():
    """Khởi tạo database và tạo bảng nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Đã kết nối Database: {DB_NAME}")

def insert_data(data):
    """Lưu một dòng dữ liệu JSON vào bảng SQLite"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Lấy thời gian thực của máy tính lúc nhận được tin nhắn
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO sensor_data 
            (timestamp, accel_x, accel_y, accel_z, light_lux, heart_rate, spo2, temperature, humidity) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_time,
            data.get("accel_x", 0.0),
            data.get("accel_y", 0.0),
            data.get("accel_z", 0.0),
            data.get("light_lux", 0.0),
            data.get("heart_rate", 0.0),
            data.get("spo2", 0),
            data.get("temperature", 0.0),
            data.get("humidity", 0.0)
        ))
        
        conn.commit()
        conn.close()
        print(f"[{current_time}] Đã lưu CSDL: HeartRate={data.get('heart_rate')} | SpO2={data.get('spo2')}")
    except Exception as e:
        print(f"❌ Lỗi ghi Database: {e}")

# ================= CÁC HÀM CALLBACK MQTT =================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✅ Đã kết nối MQTT Broker tại {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 Đang lắng nghe topic: {MQTT_TOPIC} ...\n")
    else:
        print(f"❌ Kết nối thất bại, mã lỗi: {reason_code}")

def on_message(client, userdata, msg):
    try:
        # Giải mã gói tin JSON từ ESP32
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        # Gọi hàm lưu vào SQLite
        insert_data(data)
        
    except json.JSONDecodeError:
        print(f"❌ Lỗi: Bản tin không phải chuẩn JSON - {msg.payload}")
    except Exception as e:
        print(f"❌ Lỗi xử lý tin nhắn: {e}")

# ================= CHƯƠNG TRÌNH CHÍNH =================
if __name__ == "__main__":
    # 1. Chuẩn bị database
    init_db()
    
    # 2. Khởi tạo MQTT Client (Sử dụng API v2 vì bạn đang dùng paho-mqtt bản mới 2.2.0)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    
    # 3. Gắn các hàm sự kiện (callback)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # 4. Kết nối và chạy vòng lặp vô tận
    print("Đang khởi động hệ thống thu thập dữ liệu...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng thu thập dữ liệu.")
        client.disconnect()
    except Exception as e:
        print(f"❌ Không thể kết nối tới Broker: {e}")