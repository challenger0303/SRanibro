/* Test-only PSVR2Toolkit-shaped CAPI. Never package this with SRanibro.
 * It generates SYNTHETIC frames/gaze, not hardware or tracking evidence.
 *
 * Build on Linux:
 *   cc -std=c11 -O2 -fPIC -shared mock_psvr2_capi.c -o /tmp/sranibro-mock-capi.so
 * Select explicitly, and opt in:
 *   SRANIBRO_PSVR2_MOCK=1 PSVR2_TOOLKIT_CAPI=/tmp/sranibro-mock-capi.so ...
 *
 * The guard deliberately prevents accidental use in a normal app session.
 * No SDK, Tobii, SteamVR, or upstream implementation is embedded here.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float x, y; } Vec2;
typedef struct { float x, y, z; } Vec3;
typedef struct {
    uint32_t origin_valid;
    Vec3 origin;
    uint32_t direction_valid;
    Vec3 direction;
    uint32_t pupil_valid;
    float pupil_mm;
    uint32_t pupil_position_valid;
    Vec2 pupil_position;
    uint32_t guide_valid;
    Vec2 guide;
    uint32_t blink_valid;
    uint32_t blink;
} Eye;
typedef struct {
    int64_t timestamp;
    uint32_t counter;
    Eye left, right;
    uint32_t origin_valid;
    Vec3 origin;
    uint32_t direction_valid;
    Vec3 direction;
    uint32_t convergence_valid;
} Wearable;
typedef struct {
    int64_t timestamp;
    uint32_t counter, state;
    Vec3 left, right, combined;
    float convergence_mm;
} Foveated;
typedef struct {
    unsigned char magic[2];
    uint16_t version;
    uint32_t size;
    float exposure_left, exposure_right;
    uint32_t led, exposure_counter_left, exposure_counter_right, led_counter;
    int32_t dsp;
    Vec3 lens_left, lens_right;
    uint32_t calibration;
    Vec3 origin;
    unsigned char enabled_eye, motor_sequence, motor_strength;
    Wearable wearable;
    Foveated foveated;
} Gaze;

_Static_assert(sizeof(bool) == 1, "exported bool ABI");
_Static_assert(sizeof(Eye) == 72, "eye ABI");
_Static_assert(sizeof(Wearable) == 192, "wearable ABI");
_Static_assert(sizeof(Gaze) == 0x148, "gaze ABI");
_Static_assert(offsetof(Gaze, wearable) == 80, "wearable offset");
_Static_assert(offsetof(Gaze, foveated) == 272, "foveated offset");

static bool initialized;
static int64_t start_us, last_gaze_us, last_image_us;
static uint32_t gaze_counter, image_counter;
static unsigned char image[0x100 + 400 * 200];

static int64_t now_us(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (int64_t)time.tv_sec * 1000000 + time.tv_nsec / 1000;
}

int psvr2_toolkit_init(void) {
    const char *opt_in = getenv("SRANIBRO_PSVR2_MOCK");
    if (!opt_in || strcmp(opt_in, "1") != 0) return -4;
    initialized = true;
    start_us = now_us();
    last_gaze_us = last_image_us = 0;
    gaze_counter = image_counter = 0;
    return 0;
}

void psvr2_toolkit_deinit(void) { initialized = false; }
bool psvr2_toolkit_get_driver_active(void) { return initialized; }

bool psvr2_toolkit_gaze_status(Gaze *gaze, uint32_t timeout_ms) {
    (void)timeout_ms;
    int64_t now = now_us();
    if (!initialized || !gaze || now - last_gaze_us < 8333) return false;
    last_gaze_us = now;
    memset(gaze, 0, sizeof(*gaze));
    memcpy(gaze->magic, "GS", 2);
    gaze->version = 1;
    gaze->size = sizeof(*gaze);
    gaze->enabled_eye = 2;
    gaze->wearable.timestamp = now - start_us;
    gaze->wearable.counter = ++gaze_counter;
    float sweep = ((int)(gaze_counter % 120) - 60) / 600.0f;
    Eye eye = {0};
    eye.origin_valid = eye.direction_valid = eye.pupil_valid = 1;
    eye.pupil_position_valid = eye.blink_valid = 1;
    eye.direction = (Vec3){ sweep, 0.0f, 0.995f };
    eye.pupil_mm = 4.0f;
    eye.pupil_position = (Vec2){ 0.5f + sweep, 0.5f };
    eye.blink = (gaze_counter % 240) >= 230;
    gaze->wearable.left = gaze->wearable.right = eye;
    gaze->wearable.left.origin.x = -32.0f;
    gaze->wearable.right.origin.x = 32.0f;
    return true;
}

bool psvr2_toolkit_gaze_image(unsigned char **out, uint32_t timeout_ms) {
    (void)timeout_ms;
    int64_t now = now_us();
    if (!initialized || !out || now - last_image_us < 8333) return false;
    last_image_us = now;
    ++image_counter;
    memset(image, 0, 0x100);
    memcpy(image, "VI", 2);
    uint16_t version = 1, kind = 6;
    uint32_t size = sizeof(image), timestamp = (uint32_t)(now - start_us);
    memcpy(image + 2, &version, 2);
    memcpy(image + 4, &size, 4);
    memcpy(image + 8, &timestamp, 4);
    memcpy(image + 16, &kind, 2);
    int sweep = (int)(image_counter % 120) / 3 - 20;
    for (int y = 0; y < 200; ++y) {
        for (int x = 0; x < 400; ++x) {
            int eye_x = x % 200 - 100;
            int dy = y - 100, dx = eye_x - sweep;
            unsigned char pixel = 60;
            if (eye_x * eye_x + 3 * dy * dy < 75 * 75) pixel = 180;
            if (dx * dx + dy * dy < 32 * 32) pixel = 95;
            if (dx * dx + dy * dy < 15 * 15) pixel = 15;
            image[0x100 + y * 400 + x] = pixel;
        }
    }
    *out = image;
    return true;
}
