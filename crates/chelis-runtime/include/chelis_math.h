#pragma once

/* Generated C no longer selects a math library. Every transcendental a
 * generated unit calls is a correctly rounded kernel ([05-OP-46]) that the
 * unit itself defines with internal linkage, so this header declares
 * nothing (spec/design/correctly_rounded_math.md section 4.2). */
