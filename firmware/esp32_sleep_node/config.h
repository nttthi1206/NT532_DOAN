#pragma once // Chống lỗi khai báo trùng lặp (Bắt buộc trong C/C++)

// 1. CÁC THƯ VIỆN BẮT BUỘC
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <BH1750.h>
#include "MAX30100_PulseOximeter.h"

// 2. CẤU HÌNH MẠNG & MQTT
// Sử dụng const để biến thành hằng số, tối ưu bộ nhớ RAM cho ESP32
const char* ssid = "esp32";
const char* password = "11122111";
const char* mqtt_server = "192.168.137.1"; 
const int mqtt_port = 1883;
const char* mqtt_topic = "sleep_aiot/body";