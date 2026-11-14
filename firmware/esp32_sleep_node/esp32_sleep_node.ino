#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <BH1750.h>
#include "MAX30100_PulseOximeter.h"
#include <Wire.h>
#include <ArduinoJson.h>
#include "config.h"
#include <DHT.h>
#define DHTPIN 4      // Chân DATA của DHT nối vào GPIO 4 của ESP32 (Sửa nếu cần)
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_MPU6050 mpu;
BH1750 lightMeter;
PulseOximeter pox;

unsigned long lastMsgTime = 0;
const long interval = 2000; 

// Cần thiết để báo hiệu khi MAX30100 bắt được nhịp
void onBeatDetected() {
  Serial.println("♥ Đã phát hiện mạch đập!");
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Đang kết nối WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // Chờ kết nối
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi đã kết nối thành công!");
  Serial.print("Địa chỉ IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  // Lặp lại cho đến khi kết nối MQTT thành công
  while (!client.connected()) {
    Serial.print("Đang cố gắng kết nối MQTT Broker...");
    
    // Tạo một ID ngẫu nhiên cho ESP32 để tránh trùng lặp thiết bị
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    
    // Thử kết nối
    if (client.connect(clientId.c_str())) {
      Serial.println(" THÀNH CÔNG!");
    } else {
      Serial.print(" THẤT BẠI, mã lỗi (state): ");
      Serial.print(client.state());
      Serial.println(" -> Thử lại sau 5 giây");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Tăng tốc độ I2C lên 400kHz để tránh nghẽn cổ chai khi đọc 3 cảm biến
  Wire.begin();
  Wire.setClock(400000); 
  
  setup_wifi();
  
  client.setServer(mqtt_server, mqtt_port);
  client.setBufferSize(512); // Ép mở rộng bộ đệm MQTT tránh rớt gói JSON

  if (!mpu.begin()) Serial.println("Lỗi MPU6050!");
  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) Serial.println("Lỗi BH1750!");
  
  dht.begin();
  Serial.println("Đã khởi tạo DHT22.");
  
  Serial.print("Khởi tạo MAX30100...");
  if (!pox.begin()) {
    Serial.println(" LỖI!");
  } else {
    Serial.println(" OK");
    // Đăng ký hàm callback bắt nhịp tim
    pox.setOnBeatDetectedCallback(onBeatDetected);
    // Tùy chỉnh LED hồng ngoại
    pox.setIRLedCurrent(MAX30100_LED_CURR_11MA);
  }
}

void loop() {
  // Yêu cầu thư viện MAX30100 liên tục cập nhật trạng thái (Tuyệt đối không dùng delay trong loop)
  pox.update();

  // Kiểm tra và duy trì kết nối MQTT
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();
  if (now - lastMsgTime >= interval) {
    lastMsgTime = now;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    float lux = lightMeter.readLightLevel();
    float heartRate = pox.getHeartRate();
    uint8_t spo2 = pox.getSpO2();
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();
    // Nếu lỗi đọc DHT (nan) thì gán mặc định bằng 0 để tránh lỗi văng JSON
    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("Lỗi đọc cảm biến DHT!");
      humidity = 0.0;
      temperature = 0.0;
    }

    StaticJsonDocument<384> doc;
    doc["accel_x"] = a.acceleration.x;
    doc["accel_y"] = a.acceleration.y;
    doc["accel_z"] = a.acceleration.z;
    doc["light_lux"] = lux;
    doc["heart_rate"] = heartRate;
    doc["spo2"] = spo2;
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;

    char jsonBuffer[384];
    serializeJson(doc, jsonBuffer);
    
    Serial.print("Publish message: ");
    Serial.println(jsonBuffer);
    client.publish(mqtt_topic, jsonBuffer);
  }
}