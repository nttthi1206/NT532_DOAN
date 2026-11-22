"""simulate_device_db.py

Script giả lập thiết bị ESP32 bơm trực tiếp dữ liệu cảm biến vào SQLite Database
(data/database/sleep_records.db) mỗi 2 giây.
Dùng để test màn hình "Theo dõi Real-time" và "Tạo báo cáo tổng kết" trên Streamlit
mà không cần mạch ESP32 thật hay cài đặt MQTT Broker.
"""

import argparse
from datetime import datetime
import os
import sqlite3
import sys
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "data", "database", "sleep_records.db")

SCENARIOS = {
    "good": {
        "name": "Giấc ngủ sâu lý tưởng",
        "hr": 64.0, "hr_std": 5.5, "spo2": 98.0,
        "temp": 25.0, "hum": 55.0, "lux": 2.0,
        "mov": "low",
    },
    "hot": {
        "name": "Phòng nóng ẩm & đèn sáng gây trằn trọc",
        "hr": 83.0, "hr_std": 2.8, "spo2": 96.5,
        "temp": 30.5, "hum": 78.0, "lux": 85.0,
        "mov": "high",
    },
    "stress": {
        "name": "Stress cao & nhịp tim nhanh",
        "hr": 92.0, "hr_std": 1.2, "spo2": 97.0,
        "temp": 26.0, "hum": 58.0, "lux": 8.0,
        "mov": "medium",
    },
    "apnea": {
        "name": "SpO2 sụt giảm (Nghi ngờ ngáy / ngưng thở khi ngủ)",
        "hr": 86.0, "hr_std": 2.0, "spo2": 91.5,
        "temp": 25.5, "hum": 62.0, "lux": 5.0,
        "mov": "high",
    },
}


def init_db():
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
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
    """)
    conn.commit()
    conn.close()


def generate_single_sample(cfg, rng):
    hr = float(np.clip(rng.normal(cfg["hr"], cfg["hr_std"]), 48.0, 140.0))
    spo2 = int(np.clip(round(rng.normal(cfg["spo2"], 0.8)), 85, 100))
    temp = float(np.clip(rng.normal(cfg["temp"], 0.15), 16.0, 38.0))
    hum = float(np.clip(rng.normal(cfg["hum"], 0.6), 25.0, 95.0))
    lux = float(np.clip(rng.normal(cfg["lux"], max(0.3, cfg["lux"] * 0.05)), 0.0, 1000.0))

    base_x, base_y, base_z = -0.5, 0.2, 9.7
    if cfg["mov"] == "low":
        noise = 0.06
        spike = False
    elif cfg["mov"] == "medium":
        noise = 0.25
        spike = rng.random() < 0.12
    else:
        noise = 0.55
        spike = rng.random() < 0.35

    ax = rng.normal(base_x, noise)
    ay = rng.normal(base_y, noise)
    az = rng.normal(base_z, noise)
    if spike:
        ax += rng.choice([-1, 1]) * rng.uniform(1.8, 3.8)
        ay += rng.choice([-1, 1]) * rng.uniform(1.8, 3.8)
        az += rng.choice([-1, 1]) * rng.uniform(1.5, 3.2)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accel_x": round(float(ax), 3),
        "accel_y": round(float(ay), 3),
        "accel_z": round(float(az), 3),
        "light_lux": round(lux, 1),
        "heart_rate": round(hr, 1),
        "spo2": spo2,
        "temperature": round(temp, 1),
        "humidity": round(hum, 1),
    }


def insert_row(row):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sensor_data
        (timestamp, accel_x, accel_y, accel_z, light_lux, heart_rate, spo2, temperature, humidity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["timestamp"],
        row["accel_x"],
        row["accel_y"],
        row["accel_z"],
        row["light_lux"],
        row["heart_rate"],
        row["spo2"],
        row["temperature"],
        row["humidity"],
    ))
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Giả lập ESP32 bơm dữ liệu vào SQLite")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="hot",
        help="Chọn kịch bản giả lập: good (ngủ ngon), hot (nóng & trằn trọc), stress (căng thẳng), apnea (tụt SpO2)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=35,
        help="Số bản ghi khởi tạo ban đầu để có sẵn biểu đồ ngay lập tức (mặc định: 35)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Chu kỳ bơm dữ liệu real-time tính bằng giây (mặc định: 2.0s). Đặt = 0 nếu chỉ muốn bơm batch rồi thoát.",
    )
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    rng = np.random.default_rng()
    init_db()

    print(f"🧪 Kịch bản giả lập: [{args.scenario}] - {cfg['name']}")
    print(f"📂 Database: {DB_NAME}")

    if args.batch > 0:
        print(f"⏳ Đang khởi tạo nhanh {args.batch} mẫu dữ liệu vào SQLite...")
        for _ in range(args.batch):
            insert_row(generate_single_sample(cfg, rng))
        print(f"✅ Đã bơm xong {args.batch} mẫu dữ liệu!")

    if args.interval <= 0:
        return

    print(f"📡 Bắt đầu phát dữ liệu Real-time mỗi {args.interval}s (Nhấn Ctrl+C để dừng)...\n")
    try:
        while True:
            sample = generate_single_sample(cfg, rng)
            insert_row(sample)
            print(
                f"[{sample['timestamp']}] HR={sample['heart_rate']} BPM | "
                f"SpO2={sample['spo2']}% | Temp={sample['temperature']}°C | "
                f"Hum={sample['humidity']}% | Lux={sample['light_lux']} lx"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng giả lập thiết bị.")


if __name__ == "__main__":
    main()
