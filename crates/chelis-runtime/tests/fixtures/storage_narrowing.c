/* The header's f32 -> f16 and bf16 storage conversions, driven by
 * runtime_storage_narrowing.rs. With no arguments it reads f32 bit patterns
 * (hexadecimal) from standard input and prints "f16 bf16" in hexadecimal, one
 * line each. With two hexadecimal arguments LOW HIGH it converts every f32
 * bit pattern in [LOW, HIGH) and writes the f16 and bf16 bits of each as two
 * native-endian 16-bit words. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_runtime.h"

static float chelis_probe_f32(unsigned long long bits) {
    uint32_t narrow = (uint32_t)bits;
    float value;
    memcpy(&value, &narrow, sizeof value);
    return value;
}

int main(int argc, char **argv) {
    if (argc == 3) {
        unsigned long long low = strtoull(argv[1], NULL, 16);
        unsigned long long high = strtoull(argv[2], NULL, 16);
        static uint16_t buffer[1 << 16];
        size_t used = 0;
        for (unsigned long long bits = low; bits < high; ++bits) {
            float value = chelis_probe_f32(bits);
            buffer[used++] = chelis_f32_to_f16(value);
            buffer[used++] = chelis_f32_to_bf16(value);
            if (used == sizeof buffer / sizeof buffer[0]) {
                if (fwrite(buffer, sizeof buffer[0], used, stdout) != used) {
                    return 1;
                }
                used = 0;
            }
        }
        if (fwrite(buffer, sizeof buffer[0], used, stdout) != used) {
            return 1;
        }
        return fflush(stdout) == 0 ? 0 : 1;
    }
    unsigned long long bits;
    while (scanf("%llx", &bits) == 1) {
        float value = chelis_probe_f32(bits);
        printf("%x %x\n", (unsigned)chelis_f32_to_f16(value), (unsigned)chelis_f32_to_bf16(value));
    }
    return ferror(stdin) ? 1 : 0;
}
