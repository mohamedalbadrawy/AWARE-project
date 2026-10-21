// ============================================================
//  traffic_esp32.ino  —  ESP32 Main Controller
//  مشروع: AI-Driven Smart Traffic Management System
//
//  ── المعمارية ──────────────────────────────────────────────
//   CAM1 (Lane1)        CAM2 (Lane2)
//       │                   │
//       └── Python/YOLOv8 ──┘
//                │
//         HTTP POST /update
//         {lane1, lane2, emergency, ambulance_lane}
//                │
//        ESP32 Main WebServer  (192.168.1.100)
//        ┌────────┴────────┐
//      Lane 1           Lane 2
//  Green/Yellow/Red  Green/Yellow/Red
//   LCD 0x27           LCD 0x26
//        └────────┬────────┘
//           LM393 GPIO34
//
//  ── Modes ──────────────────────────────────────────────────
//  AI Mode        : Python بيبعت بيانات عادية
//  Fallback Mode  : مفيش بيانات +10ث → 15ث ثابتة لكل لين
//  Emergency Mode : صوت + أسعاف مرئي → افتح للأسعاف
//
//  ── Fix: Non-blocking Yellow Phase ────────────────────────
//  بدل delay(3000) → millis() based yellow phase
//  server.handleClient() شغال طول الوقت حتى في الأصفر
// ============================================================
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>
// ============================================================
//  ⚙️ إعدادات الشبكة — Static IP
// ============================================================
const char*  WIFI_SSID     = "hamo";
const char*  WIFI_PASSWORD = "123456789";
IPAddress    STATIC_IP (192, 168, 137, 100);
IPAddress    GATEWAY   (192, 168, 137,   1);
IPAddress    SUBNET    (255, 255, 255, 0);
IPAddress    DNS       (8,   8,   8,   8);
const unsigned long WIFI_TIMEOUT_MS = 15000;
// ============================================================
//  ⚙️ GPIO — إشارات المرور
//  ⚠️ غيّر على حسب توصيلك
// ============================================================
#define LANE1_GREEN   26
#define LANE1_YELLOW  25
#define LANE1_RED     33
#define LANE2_GREEN   14
#define LANE2_YELLOW  27
#define LANE2_RED     13
// ============================================================
//  ⚙️ سنسور الصوت LM393
// ============================================================
#define SOUND_SENSOR_PIN    34
const unsigned long SOUND_DEBOUNCE_MS = 500;
// ============================================================
//  ⚙️ I2C LCD — SDA=21 | SCL=22
// ============================================================
LiquidCrystal_I2C lcd1(0x27, 16, 2);  // Lane 1
LiquidCrystal_I2C lcd2(0x26, 16, 2);  // Lane 2
// ============================================================
//  ⚙️ Timing
// ============================================================
const unsigned long FALLBACK_TIMEOUT_MS = 10000;
const unsigned long FALLBACK_GREEN_MS   = 15000;
const unsigned long YELLOW_DURATION_MS  =  3000;
const unsigned long MIN_GREEN_MS        =  5000;
const unsigned long MAX_GREEN_MS        = 60000;
const unsigned long EMERGENCY_DURATION  = 20000;
// ============================================================
//  WebServer
// ============================================================
WebServer server(80);
// ============================================================
//  State
// ============================================================
enum SystemMode { MODE_AI, MODE_FALLBACK, MODE_EMERGENCY };
enum LaneState  { STATE_GREEN, STATE_YELLOW, STATE_RED };
SystemMode currentMode = MODE_FALLBACK;
struct Lane {
  LaneState     state;
  int           vehicleCount;
  unsigned long greenDuration;
};
Lane lane1 = { STATE_RED,   0, FALLBACK_GREEN_MS };
Lane lane2 = { STATE_GREEN, 0, FALLBACK_GREEN_MS };
unsigned long lastDataReceivedMs  = 0;
unsigned long currentPhaseStartMs = 0;
int           activeLane          = 1;
// ✅ Fix 1: Yellow Phase (Non-blocking)
bool          inYellowPhase       = false;
unsigned long yellowStartMs       = 0;
int           nextActiveLane      = 0;
// Emergency
bool          emergencyActive     = false;
unsigned long emergencyStartMs    = 0;
int           ambulanceLane       = 0;
// Sound (LM393)
bool          soundDetected       = false;
unsigned long soundFirstMs        = 0;
bool          soundConfirmed      = false;
// ============================================================
//  calcGreenDuration()
// ============================================================
unsigned long calcGreenDuration(int count) {
  unsigned long dur = (unsigned long)count * 1000UL + MIN_GREEN_MS;
  return (dur > MAX_GREEN_MS) ? MAX_GREEN_MS : dur;
}
// ============================================================
//  applyLane1 / applyLane2
// ============================================================
void applyLane1(LaneState s) {
  digitalWrite(LANE1_GREEN,  s == STATE_GREEN  ? HIGH : LOW);
  digitalWrite(LANE1_YELLOW, s == STATE_YELLOW ? HIGH : LOW);
  digitalWrite(LANE1_RED,    s == STATE_RED    ? HIGH : LOW);
  lane1.state = s;
}
void applyLane2(LaneState s) {
  digitalWrite(LANE2_GREEN,  s == STATE_GREEN  ? HIGH : LOW);
  digitalWrite(LANE2_YELLOW, s == STATE_YELLOW ? HIGH : LOW);
  digitalWrite(LANE2_RED,    s == STATE_RED    ? HIGH : LOW);
  lane2.state = s;
}
// ============================================================
//  updateLCDs()
//  LCD1 (0x27) → Lane 1 | LCD2 (0x26) → Lane 2
// ============================================================
void updateLCDs() {
  char line1_lcd1[17], line2_lcd1[17];
  char line1_lcd2[17], line2_lcd2[17];
  if (currentMode == MODE_EMERGENCY) {
    if (ambulanceLane == 1) {
      snprintf(line1_lcd1, 17, "** AMBULANCE!! *");
      snprintf(line2_lcd1, 17, "LANE 1 OPEN  >>>");
      snprintf(line1_lcd2, 17, "** AMBULANCE!! *");
      snprintf(line2_lcd2, 17, "LANE 2 WAIT  ---");
    } else if (ambulanceLane == 2) {
      snprintf(line1_lcd1, 17, "** AMBULANCE!! *");
      snprintf(line2_lcd1, 17, "LANE 1 WAIT  ---");
      snprintf(line1_lcd2, 17, "** AMBULANCE!! *");
      snprintf(line2_lcd2, 17, "LANE 2 OPEN  >>>");
    } else {
      snprintf(line1_lcd1, 17, "** EMERGENCY ** ");
      snprintf(line2_lcd1, 17, "Lane1:  RED     ");
      snprintf(line1_lcd2, 17, "** EMERGENCY ** ");
      snprintf(line2_lcd2, 17, "Lane2:  RED     ");
    }
  } else {
    const char* modeLine = (currentMode == MODE_FALLBACK)
                           ? " FALLBACK  MODE "
                           : "    AI  MODE    ";
    snprintf(line1_lcd1, 17, "%s", modeLine);
    snprintf(line1_lcd2, 17, "%s", modeLine);
    const char* s1 = (lane1.state == STATE_GREEN)  ? "GREEN " :
                     (lane1.state == STATE_YELLOW) ? "YELLOW" : "RED   ";
    const char* s2 = (lane2.state == STATE_GREEN)  ? "GREEN " :
                     (lane2.state == STATE_YELLOW) ? "YELLOW" : "RED   ";
    if (currentMode == MODE_AI) {
      snprintf(line2_lcd1, 17, "L1:%-6s C:%-3d", s1, lane1.vehicleCount);
      snprintf(line2_lcd2, 17, "L2:%-6s C:%-3d", s2, lane2.vehicleCount);
    } else {
      snprintf(line2_lcd1, 17, "Lane1: %-6s   ", s1);
      snprintf(line2_lcd2, 17, "Lane2: %-6s   ", s2);
    }
  }
  lcd1.setCursor(0, 0); lcd1.print(line1_lcd1);
  lcd1.setCursor(0, 1); lcd1.print(line2_lcd1);
  lcd2.setCursor(0, 0); lcd2.print(line1_lcd2);
  lcd2.setCursor(0, 1); lcd2.print(line2_lcd2);
}
// ============================================================
//  activateEmergency()
// ============================================================
void activateEmergency(int ambLane) {
  emergencyActive  = true;
  emergencyStartMs = millis();
  ambulanceLane    = ambLane;
  currentMode      = MODE_EMERGENCY;
  inYellowPhase    = false; // ✅ اقطع أي yellow phase شغالة
  if (ambLane == 1) {
    applyLane1(STATE_GREEN);
    applyLane2(STATE_RED);
    Serial.println("[EMERGENCY] Lane 1 GREEN — ambulance passing");
  } else if (ambLane == 2) {
    applyLane1(STATE_RED);
    applyLane2(STATE_GREEN);
    Serial.println("[EMERGENCY] Lane 2 GREEN — ambulance passing");
  } else {
    applyLane1(STATE_RED);
    applyLane2(STATE_RED);
    Serial.println("[EMERGENCY] ALL RED — siren only");
  }
  updateLCDs();
}
// ============================================================
//  ✅ Fix 1: startYellowPhase() — بتبدأ فاز الأصفر
//  بتتعمل مرة واحدة لما يحين وقت التبديل
// ============================================================
void startYellowPhase() {
  if (inYellowPhase) return;
  inYellowPhase  = true;
  yellowStartMs  = millis();
  nextActiveLane = (activeLane == 1) ? 2 : 1;
  applyLane1(STATE_YELLOW);
  applyLane2(STATE_YELLOW);
  updateLCDs();
  Serial.printf("[YELLOW] Phase started — next: Lane %d\n", nextActiveLane);
}
// ============================================================
//  ✅ Fix 1: updateYellowPhase() — بتكمّل التبديل لما الأصفر يخلص
//  بتتعمل كل loop iteration وهي Non-blocking
// ============================================================
void updateYellowPhase() {
  if (!inYellowPhase) return;
  if (millis() - yellowStartMs < YELLOW_DURATION_MS) return;
  // ✅ خلص الأصفر — بدّل اللين
  inYellowPhase = false;
  activeLane    = nextActiveLane;
  if (activeLane == 1) {
    applyLane1(STATE_GREEN);
    applyLane2(STATE_RED);
  } else {
    applyLane1(STATE_RED);
    applyLane2(STATE_GREEN);
  }
  currentPhaseStartMs = millis();
  updateLCDs();
  unsigned long nextDur = (activeLane == 1)
                          ? lane1.greenDuration
                          : lane2.greenDuration;
  Serial.printf("[SWITCH] Lane %d GREEN for %lu s\n",
                activeLane, nextDur / 1000);
}
// ============================================================
//  WebServer Handlers
// ============================================================
void handleUpdate() {
  if (!server.hasArg("plain")) {
    server.send(400, "application/json", "{\"error\":\"no body\"}");
    return;
  }
  StaticJsonDocument<200> doc;
  if (deserializeJson(doc, server.arg("plain"))) {
    server.send(400, "application/json", "{\"error\":\"invalid json\"}");
    return;
  }
  int  c1      = doc["lane1"]          | 0;
  int  c2      = doc["lane2"]          | 0;
  bool emerg   = doc["emergency"]      | false;
  int  ambLane = doc["ambulance_lane"] | 0;
  lane1.vehicleCount  = c1;
  lane2.vehicleCount  = c2;
  lane1.greenDuration = calcGreenDuration(c1);
  lane2.greenDuration = calcGreenDuration(c2);
  lastDataReceivedMs  = millis();
  if (emerg && soundConfirmed) {
    if (!emergencyActive) activateEmergency(ambLane);
  } else if (!emergencyActive) {
    if (currentMode == MODE_FALLBACK) {
      currentMode = MODE_AI;
      Serial.println("[MODE] Switched to AI Mode");
    }
  }
  Serial.printf("[DATA] L1=%d(%lus) L2=%d(%lus) emerg=%d ambLane=%d\n",
    c1, lane1.greenDuration / 1000,
    c2, lane2.greenDuration / 1000,
    emerg, ambLane);
  updateLCDs();
  server.send(200, "application/json", "{\"status\":\"ok\"}");
}
void handleStatus() {
  StaticJsonDocument<256> doc;
  doc["mode"]           = (currentMode == MODE_AI)        ? "AI"        :
                          (currentMode == MODE_EMERGENCY)  ? "EMERGENCY" : "FALLBACK";
  doc["lane1_state"]    = (lane1.state == STATE_GREEN)    ? "GREEN"  :
                          (lane1.state == STATE_YELLOW)   ? "YELLOW" : "RED";
  doc["lane2_state"]    = (lane2.state == STATE_GREEN)    ? "GREEN"  :
                          (lane2.state == STATE_YELLOW)   ? "YELLOW" : "RED";
  doc["lane1_cars"]     = lane1.vehicleCount;
  doc["lane2_cars"]     = lane2.vehicleCount;
  doc["active_lane"]    = activeLane;
  doc["emergency"]      = emergencyActive;
  doc["ambulance_lane"] = ambulanceLane;
  doc["sound_detected"] = soundConfirmed;
  doc["yellow_phase"]   = inYellowPhase; // ✅ مفيد للـ debug
  String out;
  serializeJson(doc, out);
  server.send(200, "application/json", out);
}
void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}
// ============================================================
//  connectWiFi() — Static IP
// ============================================================
bool connectWiFi() {
  if (!WiFi.config(STATIC_IP, GATEWAY, SUBNET, DNS)) {
    Serial.println("[WiFi] Static IP config failed!");
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - t0 > WIFI_TIMEOUT_MS) {
      Serial.println("\n[WiFi] TIMEOUT — Fallback mode active");
      return false;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.printf("[WiFi] Connected! IP: %s\n",  WiFi.localIP().toString().c_str());
  Serial.printf("[API]  POST: http://%s/update\n", WiFi.localIP().toString().c_str());
  Serial.printf("[API]  GET : http://%s/status\n", WiFi.localIP().toString().c_str());
  return true;
}
// ============================================================
//  checkSoundSensor() — LM393 + Software Debounce
// ============================================================
void checkSoundSensor() {
  int val = digitalRead(SOUND_SENSOR_PIN);
  if (val == HIGH) {
    if (!soundDetected) {
      soundDetected = true;
      soundFirstMs  = millis();
    } else if (!soundConfirmed &&
               millis() - soundFirstMs >= SOUND_DEBOUNCE_MS) {
      soundConfirmed = true;
      Serial.println("[SOUND] Siren confirmed by LM393!");
    }
  } else {
    if (soundDetected) {
      soundDetected  = false;
      soundConfirmed = false;
      Serial.println("[SOUND] Siren cleared.");
    }
  }
}
// ============================================================
//  ✅ Fix 1: handleAIMode() — Non-blocking
// ============================================================
void handleAIMode() {
  if (inYellowPhase) {
    updateYellowPhase(); // تابع الأصفر لو شغال
    return;
  }
  unsigned long dur = (activeLane == 1)
                      ? lane1.greenDuration
                      : lane2.greenDuration;
  if (millis() - currentPhaseStartMs >= dur) {
    startYellowPhase();
  }
}
// ============================================================
//  ✅ Fix 1: handleFallbackMode() — Non-blocking
// ============================================================
void handleFallbackMode() {
  if (inYellowPhase) {
    updateYellowPhase();
    return;
  }
  if (millis() - currentPhaseStartMs >= FALLBACK_GREEN_MS) {
    startYellowPhase();
  }
}
// ============================================================
//  handleEmergencyMode()
// ============================================================
void handleEmergencyMode() {
  if (millis() - emergencyStartMs >= EMERGENCY_DURATION) {
    emergencyActive = false;
    ambulanceLane   = 0;
    soundConfirmed  = false;
    inYellowPhase   = false; // ✅ reset
    Serial.println("[EMERGENCY] Cleared — resuming normal operation.");
    currentMode = (lastDataReceivedMs > 0 &&
                   millis() - lastDataReceivedMs < FALLBACK_TIMEOUT_MS)
                  ? MODE_AI : MODE_FALLBACK;
    if (activeLane == 1) { applyLane1(STATE_GREEN); applyLane2(STATE_RED); }
    else                  { applyLane1(STATE_RED);   applyLane2(STATE_GREEN); }
    currentPhaseStartMs = millis();
    updateLCDs();
    Serial.printf("[MODE] Resumed: %s\n",
                  currentMode == MODE_AI ? "AI" : "FALLBACK");
  }
}
// ============================================================
//  setup()
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.println("\n[SYS] ESP32 Main — Traffic Controller starting...");
  Serial.printf("[SYS] Target IP: %s\n", STATIC_IP.toString().c_str());
  // GPIO إشارات
  pinMode(LANE1_GREEN,  OUTPUT); digitalWrite(LANE1_GREEN,  LOW);
  pinMode(LANE1_YELLOW, OUTPUT); digitalWrite(LANE1_YELLOW, LOW);
  pinMode(LANE1_RED,    OUTPUT); digitalWrite(LANE1_RED,    HIGH);
  pinMode(LANE2_GREEN,  OUTPUT); digitalWrite(LANE2_GREEN,  LOW);
  pinMode(LANE2_YELLOW, OUTPUT); digitalWrite(LANE2_YELLOW, LOW);
  pinMode(LANE2_RED,    OUTPUT); digitalWrite(LANE2_RED,    HIGH);
  // Sound Sensor
  pinMode(SOUND_SENSOR_PIN, INPUT);
  // I2C + LCDs
  Wire.begin(21, 22);
  lcd1.init(); lcd1.backlight();
  lcd2.init(); lcd2.backlight();
  lcd1.setCursor(0, 0); lcd1.print(" Traffic Ctrl 1 ");
  lcd1.setCursor(0, 1); lcd1.print("  Starting...   ");
  lcd2.setCursor(0, 0); lcd2.print(" Traffic Ctrl 2 ");
  lcd2.setCursor(0, 1); lcd2.print("  Starting...   ");
  // WiFi + WebServer
  if (connectWiFi()) {
    server.on("/update", HTTP_POST, handleUpdate);
    server.on("/status", HTTP_GET,  handleStatus);
    server.onNotFound(handleNotFound);
    server.begin();
    Serial.println("[SERVER] HTTP server started");
    lcd1.setCursor(0, 1);
    lcd1.print(WiFi.localIP().toString().c_str());
  } else {
    Serial.println("[SYS] No WiFi — Fallback mode only");
    lcd1.setCursor(0, 0); lcd1.print("  No WiFi!!!    ");
    lcd2.setCursor(0, 0); lcd2.print("  No WiFi!!!    ");
    delay(2000);
  }
  // ابدأ بـ Fallback: Lane1 أخضر
  currentMode         = MODE_FALLBACK;
  activeLane          = 1;
  currentPhaseStartMs = millis();
  lastDataReceivedMs  = 0;
  applyLane1(STATE_GREEN);
  applyLane2(STATE_RED);
  updateLCDs();
  Serial.println("[SYS] Ready! Running in FALLBACK until Python connects.");
}
// ============================================================
//  loop() — server.handleClient() شغال طول الوقت ✅
// ============================================================
void loop() {
  server.handleClient();   // ✅ شغال حتى في فاز الأصفر
  checkSoundSensor();      // ✅ شغال طول الوقت
  // Fallback Timeout
  if (currentMode == MODE_AI &&
      millis() - lastDataReceivedMs > FALLBACK_TIMEOUT_MS) {
    currentMode = MODE_FALLBACK;
    Serial.println("[MODE] No data from Python — switched to FALLBACK");
    updateLCDs();
  }
  // تشغيل المنطق حسب الـ Mode
  switch (currentMode) {
    case MODE_AI:        handleAIMode();        break;
    case MODE_FALLBACK:  handleFallbackMode();  break;
    case MODE_EMERGENCY: handleEmergencyMode(); break;
  }
  // WiFi Reconnect كل 10ث
  static unsigned long lastWifiCheck = 0;
  if (millis() - lastWifiCheck > 10000) {
    lastWifiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] Lost — reconnecting...");
      if (connectWiFi()) server.begin();
    }
  }
}