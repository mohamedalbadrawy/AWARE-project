// ============================================================
//  CameraWebServer.ino  —  ESP32-CAM (AI Thinker) - CAMERA 1
//  مشروع: AI-Driven Smart Traffic Management System
//
//  Endpoints:
//  http://192.168.137.101/capture     ← صورة واحدة (للتأكد)
//  http://192.168.137.101:81/stream   ← MJPEG stream (لـ Python)
// ============================================================

#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ============================================================
//  ⚙️ إعدادات الكاميرا الأولى
// ============================================================
const char* CAM_ID = "cam1";                     // ← "cam1" للكاميرا الأولى
IPAddress STATIC_IP (192, 168, 137, 101);        // ← IP الكاميرا الأولى
#define USE_VGA                                  // ← جودة VGA (640x480)
// ============================================================

const char*  WIFI_SSID     = "hamo";
const char*  WIFI_PASSWORD = "123456789";
IPAddress    GATEWAY       (192, 168, 137,   1);
IPAddress    SUBNET        (255, 255, 255, 0);
IPAddress    DNS           (8,   8,   8,   8);

const unsigned long WIFI_TIMEOUT_MS = 15000;
const unsigned long WIFI_CHECK_MS   =  5000;

// forward declaration
void startCameraServer();

// ============================================================
//  initCamera()
// ============================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel  = LEDC_CHANNEL_0;
  config.ledc_timer    = LEDC_TIMER_0;
  config.pin_d0        = Y2_GPIO_NUM;
  config.pin_d1        = Y3_GPIO_NUM;
  config.pin_d2        = Y4_GPIO_NUM;
  config.pin_d3        = Y5_GPIO_NUM;
  config.pin_d4        = Y6_GPIO_NUM;
  config.pin_d5        = Y7_GPIO_NUM;
  config.pin_d6        = Y8_GPIO_NUM;
  config.pin_d7        = Y9_GPIO_NUM;
  config.pin_xclk      = XCLK_GPIO_NUM;
  config.pin_pclk      = PCLK_GPIO_NUM;
  config.pin_vsync     = VSYNC_GPIO_NUM;
  config.pin_href      = HREF_GPIO_NUM;
  config.pin_sccb_sda  = SIOD_GPIO_NUM;
  config.pin_sccb_scl  = SIOC_GPIO_NUM;
  config.pin_pwdn      = PWDN_GPIO_NUM;
  config.pin_reset     = RESET_GPIO_NUM;
  config.xclk_freq_hz  = 20000000;
  config.pixel_format  = PIXFORMAT_JPEG;

  if (psramFound()) {
#ifdef USE_VGA
    config.frame_size   = FRAMESIZE_VGA;   // 640x480
#else
    config.frame_size   = FRAMESIZE_QVGA;  // 320x240
#endif
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM-%s] Init failed: 0x%x\n", CAM_ID, err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s && s->id.PID == OV2640_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
    s->set_contrast(s, 1);
  }

  Serial.printf("[CAM-%s] Init OK\n", CAM_ID);
  return true;
}

// ============================================================
//  connectWiFi() — Static IP
// ============================================================
bool connectWiFi() {
  if (!WiFi.config(STATIC_IP, GATEWAY, SUBNET, DNS)) {
    Serial.printf("[CAM-%s] Static IP config failed!\n", CAM_ID);
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[CAM-%s] Connecting to WiFi", CAM_ID);

  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - t0 > WIFI_TIMEOUT_MS) {
      Serial.println("\n[WiFi] TIMEOUT");
      return false;
    }
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.printf("[CAM-%s] Connected!\n", CAM_ID);
  Serial.printf("[CAM-%s] IP      : %s\n", CAM_ID, WiFi.localIP().toString().c_str());
  Serial.printf("[CAM-%s] Capture : http://%s/capture\n", CAM_ID, WiFi.localIP().toString().c_str());
  Serial.printf("[CAM-%s] Stream  : http://%s:81/stream\n", CAM_ID, WiFi.localIP().toString().c_str());
  return true;
}

// ============================================================
//  setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.printf("\n[SYS] ESP32-CAM (%s) starting...\n", CAM_ID);
  Serial.printf("[SYS] Target IP: %s\n", STATIC_IP.toString().c_str());

  // Reset hardware الكاميرا
  pinMode(32, OUTPUT);
  digitalWrite(32, LOW);  delay(500);
  digitalWrite(32, HIGH); delay(100);

  if (!initCamera()) {
    Serial.println("[SYS] Camera failed — restarting in 3s...");
    delay(3000);
    ESP.restart();
  }

  if (!connectWiFi()) {
    Serial.printf("[CAM-%s] No WiFi — ESP32-Main will use Fallback.\n", CAM_ID);
    return;
  }

  startCameraServer();
  Serial.printf("[CAM-%s] All servers ready!\n", CAM_ID);
}

// ============================================================
//  loop() — WiFi Reconnect تلقائي
// ============================================================
void loop() {
  static unsigned long lastCheck = 0;

  if (millis() - lastCheck >= WIFI_CHECK_MS) {
    lastCheck = millis();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.printf("[CAM-%s] WiFi lost — reconnecting...\n", CAM_ID);
      WiFi.disconnect();
      delay(1000);

      if (connectWiFi()) {
        startCameraServer();
        Serial.printf("[CAM-%s] Stream restored.\n", CAM_ID);
      } else {
        Serial.println("[SYS] Still no WiFi. ESP32-Main stays in Fallback.");
      }
    }
  }
}