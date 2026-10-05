/* Generated C's f64 -> f16 and bf16 storage conversions
 * (chelis_host_f64_to_f16 and chelis_host_f64_to_bf16, defined before this
 * text), driven by host_emit.rs's storage_narrowing_tests. With no arguments
 * it reads f64 bit patterns (hexadecimal) from standard input and prints
 * "f16 bf16" in hexadecimal, one line each. With two hexadecimal arguments
 * LOW HIGH it widens every f32 bit pattern in [LOW, HIGH) exactly to f64,
 * converts it, and writes the f16 and bf16 bits as two native-endian 16-bit
 * words. */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc == 3) {
        unsigned long long low = strtoull(argv[1], NULL, 16);
        unsigned long long high = strtoull(argv[2], NULL, 16);
        static uint16_t buffer[1 << 16];
        size_t used = 0;
        for (unsigned long long bits = low; bits < high; ++bits) {
            uint32_t narrow = (uint32_t)bits;
            float single;
            memcpy(&single, &narrow, sizeof single);
            double value = (double)single;
            buffer[used++] = chelis_host_f64_to_f16(value);
            buffer[used++] = chelis_host_f64_to_bf16(value);
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
        uint64_t wide = (uint64_t)bits;
        double value;
        memcpy(&value, &wide, sizeof value);
        printf("%x %x\n", (unsigned)chelis_host_f64_to_f16(value),
               (unsigned)chelis_host_f64_to_bf16(value));
    }
    return ferror(stdin) ? 1 : 0;
}
