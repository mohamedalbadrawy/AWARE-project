// ============================================================
//  app_httpd.cpp  —  Camera HTTP Server
//  مشروع: AI-Driven Smart Traffic Management System
//
//  ⚠️ المكان: نفس فولدر CameraWebServer.ino
//
//  Structure:
//  CameraWebServer/
//  ├── CameraWebServer.ino   ← الملف الرئيسي
//  ├── app_httpd.cpp         ← الملف ده
//  └── camera_pins.h         ← موجود بالفعل
//
//  Ports:
//  /capture → port 80  (صورة واحدة)
//  /stream  → port 81  (MJPEG stream لـ Python)
// ============================================================

#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"

#define PART_BOUNDARY "123456789000000000000987654321"

static const char* STREAM_CONTENT_TYPE =
  "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY =
  "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
  "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static httpd_handle_t stream_httpd = NULL;
static httpd_handle_t camera_httpd = NULL;

// ============================================================
//  stream_handler() — MJPEG stream متواصل على port 81
//  Python بتتصل بيه وتقرأ منه frames
// ============================================================
static esp_err_t stream_handler(httpd_req_t* req) {
  camera_fb_t* fb       = NULL;
  esp_err_t    res      = ESP_OK;
  char*        part_buf = (char*)malloc(128);

  if (!part_buf) return ESP_ERR_NO_MEM;

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) { free(part_buf); return res; }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "15");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[STREAM] Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    // boundary
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY,
                                strlen(STREAM_BOUNDARY));
    if (res != ESP_OK) break;

    // header بحجم الصورة
    size_t hlen = snprintf(part_buf, 128, STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res != ESP_OK) break;

    // الصورة نفسها
    res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK) break;
  }

  if (fb) esp_camera_fb_return(fb);
  free(part_buf);
  return res;
}

// ============================================================
//  capture_handler() — صورة JPEG واحدة على port 80
//  للتأكد من الكاميرا عبر المتصفح أو VLC
// ============================================================
static esp_err_t capture_handler(httpd_req_t* req) {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition",
                     "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  esp_err_t res = httpd_resp_send(req,
                                  (const char*)fb->buf,
                                  fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// ============================================================
//  startCameraServer() — بتشغل السيرفرين
//  بتتـ call من setup() في CameraWebServer.ino
// ============================================================
void startCameraServer() {
  // ── سيرفر الكاميرا (port 80) → /capture ─────────────────
  httpd_config_t config   = HTTPD_DEFAULT_CONFIG();
  config.server_port      = 80;
  config.ctrl_port        = 32768;
  config.max_uri_handlers = 4;
  config.stack_size       = 8192;

  httpd_uri_t capture_uri = {
    .uri      = "/capture",
    .method   = HTTP_GET,
    .handler  = capture_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    Serial.println("[SERVER] Capture → port 80  (/capture)");
  } else {
    Serial.println("[SERVER] Capture server FAILED");
  }

  // ── سيرفر الـ Stream (port 81) → /stream ─────────────────
  httpd_config_t stream_config   = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port      = 81;
  stream_config.ctrl_port        = 32769;
  stream_config.max_uri_handlers = 2;
  stream_config.stack_size       = 8192;

  httpd_uri_t stream_uri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("[SERVER] Stream  → port 81  (/stream)");
  } else {
    Serial.println("[SERVER] Stream server FAILED");
  }
}
