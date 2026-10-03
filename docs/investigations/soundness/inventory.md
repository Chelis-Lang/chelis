# Authority and subject inventory

Source snapshot: `38ab515f5e0d070d8d31ecfc9d114bbbb7b91ce0`, 2026-09-09.

This inventory gives the account an independently reproducible specification
membership check. It is not a list of all semantic requirements: unnumbered prose,
tables, recursive grammar cases, incorporated registry rows, and feature
interactions can impose obligations without a blockquote atom.

The existing [rejection-registry generator](../../../scripts/generate_rejection_registries.py)
defines `discover_atoms`, using normative definition lines in the thirteen
numbered chapters. At this snapshot that definition discovers **168
unique atoms**. The table below uses that exact definition-line grammar, not
cross-references or comment citations. Every atom is routed to a chapter of this
account. Routing by itself is not proof that the chapter has established its
enforcement; the chapter's record and residual disposition decide that.

The second table independently enumerates **107 level-two chapter
headings**, including subjects with no atom IDs. A heading is a reading boundary,
not a semantic requirement count. Complete chapter reading and the three detailed
accounts supply the interpretation. Historical status, examples, and roadmap
material must be distinguished from active normative rules rather than blindly
counted as requirements.

A third independent view comes from implementation surfaces: grammar/tag/type/op
universes, public entries and transports, mutations and consumers, and target and
feature configuration. The detailed chapters identify those discovery sources and
their limitations. No source-file count, atom count, or baseline certifies the
absence of an omitted interaction.

## Named atom definitions

| Atom | Definition location at the snapshot | Primary account |
|---|---|---|
| `[01-CID-1]` | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 136 | [Language](language.md) |
| `[03-META-1]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 81 | [Language](language.md) |
| `[03-META-2]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 111 | [Language](language.md) |
| `[03-META-3]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 122 | [Language](language.md) |
| `[03-PROG-1]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1019 | [Language](language.md) |
| `[03-PROG-2]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1042 | [Language](language.md) |
| `[03-PROG-3]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1061 | [Language](language.md) |
| `[03-ROLE-1]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1098 | [Language](language.md) |
| `[03-ROLE-2]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1112 | [Language](language.md) |
| `[03-ROLE-3]` | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1131 | [Language](language.md) |
| `[04-DTYPE-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 59 | [Language](language.md) |
| `[04-TGT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 195 | [Runtime](runtime.md) |
| `[04-ADT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 395 | [Language](language.md) |
| `[04-ADT-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 408 | [Language](language.md) |
| `[04-ADT-3]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 417 | [Language](language.md) |
| `[04-ADT-4]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 430 | [Language](language.md) |
| `[04-PAT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 456 | [Language](language.md) |
| `[04-INF-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 732 | [Language](language.md) |
| `[04-INF-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 766 | [Language](language.md) |
| `[04-INF-3]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 781 | [Language](language.md) |
| `[04-INF-4]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 794 | [Language](language.md) |
| `[04-INF-5]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 821 | [Language](language.md) |
| `[04-INF-6]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 837 | [Language](language.md) |
| `[04-INF-7]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 858 | [Language](language.md) |
| `[04-INF-8]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 883 | [Language](language.md) |
| `[04-SHAPE-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 1682 | [Runtime](runtime.md) |
| `[04-LIT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 1811 | [Language](language.md) |
| `[04-DTYPE-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2175 | [Language](language.md) |
| `[04-FIT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2238 | [Integration](integration.md) |
| `[04-FIT-11]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2267 | [Integration](integration.md) |
| `[04-FIT-12]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2273 | [Integration](integration.md) |
| `[04-FIT-18]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2294 | [Integration](integration.md) |
| `[04-FIT-13]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2311 | [Integration](integration.md) |
| `[04-FIT-14]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2331 | [Integration](integration.md) |
| `[04-FIT-15]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2338 | [Integration](integration.md) |
| `[04-FIT-16]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2346 | [Integration](integration.md) |
| `[04-FIT-17]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2360 | [Integration](integration.md) |
| `[04-FIT-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2397 | [Integration](integration.md) |
| `[04-FIT-9]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2413 | [Integration](integration.md) |
| `[04-FIT-10]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2424 | [Integration](integration.md) |
| `[04-EFF-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2456 | [Language](language.md) |
| `[04-LIN-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2634 | [Language](language.md) |
| `[04-LIN-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2649 | [Language](language.md) |
| `[04-LIN-3]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2660 | [Language](language.md) |
| `[04-LIN-4]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2667 | [Language](language.md) |
| `[04-LIN-5]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2677 | [Language](language.md) |
| `[04-LIN-6]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2685 | [Language](language.md) |
| `[04-LIN-7]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2693 | [Language](language.md) |
| `[04-LIN-8]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2705 | [Language](language.md) |
| `[04-NUM-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2799 | [Runtime](runtime.md) |
| `[04-NUM-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2805 | [Runtime](runtime.md) |
| `[04-NUM-3]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2818 | [Runtime](runtime.md) |
| `[04-NUM-4]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2826 | [Runtime](runtime.md) |
| `[04-NUM-5]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2835 | [Runtime](runtime.md) |
| `[04-NUM-6]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2842 | [Runtime](runtime.md) |
| `[04-NUM-7]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2847 | [Runtime](runtime.md) |
| `[04-NUM-8]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2862 | [Runtime](runtime.md) |
| `[04-NUM-9]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2954 | [Runtime](runtime.md) |
| `[04-NUM-10]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2980 | [Runtime](runtime.md) |
| `[04-NUM-11]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3003 | [Runtime](runtime.md) |
| `[04-NUM-12]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3019 | [Runtime](runtime.md) |
| `[04-NUM-13]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3036 | [Runtime](runtime.md) |
| `[04-NUM-14]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3046 | [Runtime](runtime.md) |
| `[04-NUM-15]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3067 | [Runtime](runtime.md) |
| `[04-NUM-16]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3083 | [Runtime](runtime.md) |
| `[04-TOT-1]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3141 | [Language](language.md) |
| `[04-TOT-2]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3147 | [Language](language.md) |
| `[04-TOT-3]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3151 | [Language](language.md) |
| `[04-TOT-4]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3156 | [Language](language.md) |
| `[04-TOT-5]` | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3169 | [Language](language.md) |
| `[05-OP-17]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 105 | [Runtime](runtime.md) |
| `[05-OP-18]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 114 | [Runtime](runtime.md) |
| `[05-OP-19]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 119 | [Runtime](runtime.md) |
| `[05-OP-40]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 201 | [Runtime](runtime.md) |
| `[05-OP-20]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 244 | [Runtime](runtime.md) |
| `[05-OP-21]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 253 | [Runtime](runtime.md) |
| `[05-OP-22]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 258 | [Runtime](runtime.md) |
| `[05-OP-11]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 310 | [Runtime](runtime.md) |
| `[05-OP-12]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 328 | [Runtime](runtime.md) |
| `[05-OP-13]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 349 | [Runtime](runtime.md) |
| `[05-OP-14]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 359 | [Runtime](runtime.md) |
| `[05-OP-15]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 384 | [Runtime](runtime.md) |
| `[05-OP-16]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 394 | [Runtime](runtime.md) |
| `[05-OP-29]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 399 | [Runtime](runtime.md) |
| `[05-OP-30]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 425 | [Runtime](runtime.md) |
| `[05-AXIS-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 454 | [Runtime](runtime.md) |
| `[05-RWIN-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 543 | [Runtime](runtime.md) |
| `[05-RWIN-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 626 | [Runtime](runtime.md) |
| `[05-OP-39]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 633 | [Runtime](runtime.md) |
| `[05-OP-65]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 699 | [Runtime](runtime.md) |
| `[05-DIM-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 720 | [Runtime](runtime.md) |
| `[05-DIM-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 729 | [Runtime](runtime.md) |
| `[05-DIM-3]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 743 | [Runtime](runtime.md) |
| `[05-MOV-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 865 | [Runtime](runtime.md) |
| `[05-OP-7]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 907 | [Runtime](runtime.md) |
| `[05-SHAPE-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 940 | [Runtime](runtime.md) |
| `[05-OP-45]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 958 | [Runtime](runtime.md) |
| `[05-OP-8]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 984 | [Runtime](runtime.md) |
| `[05-OP-37]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1011 | [Runtime](runtime.md) |
| `[05-RNG-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1042 | [Runtime](runtime.md) |
| `[05-OP-41]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1080 | [Runtime](runtime.md) |
| `[05-OP-26]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1114 | [Runtime](runtime.md) |
| `[05-OP-27]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1124 | [Runtime](runtime.md) |
| `[05-OP-28]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1129 | [Runtime](runtime.md) |
| `[05-OP-36]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1137 | [Runtime](runtime.md) |
| `[05-OP-43]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1201 | [Runtime](runtime.md) |
| `[05-SPARSE-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1264 | [Runtime](runtime.md) |
| `[05-SPARSE-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1282 | [Runtime](runtime.md) |
| `[05-OP-66]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1389 | [Runtime](runtime.md) |
| `[05-HOST-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1417 | [Runtime](runtime.md) |
| `[05-HOST-4]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1470 | [Runtime](runtime.md) |
| `[05-HOST-3]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1505 | [Runtime](runtime.md) |
| `[05-OP-38]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1515 | [Runtime](runtime.md) |
| `[05-OP-9]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1542 | [Runtime](runtime.md) |
| `[05-OP-10]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1561 | [Runtime](runtime.md) |
| `[05-OP-25]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1576 | [Runtime](runtime.md) |
| `[05-HOST-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1623 | [Runtime](runtime.md) |
| `[05-OP-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1636 | [Runtime](runtime.md) |
| `[05-OP-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1671 | [Runtime](runtime.md) |
| `[05-OP-3]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1691 | [Runtime](runtime.md) |
| `[05-OP-4]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1726 | [Runtime](runtime.md) |
| `[05-OP-5]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1738 | [Runtime](runtime.md) |
| `[05-OP-31]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1760 | [Runtime](runtime.md) |
| `[05-OP-32]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1875 | [Runtime](runtime.md) |
| `[05-OP-33]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1941 | [Runtime](runtime.md) |
| `[05-OP-44]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2227 | [Runtime](runtime.md) |
| `[05-OP-34]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2384 | [Runtime](runtime.md) |
| `[05-OP-35]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2417 | [Runtime](runtime.md) |
| `[05-OP-6]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2768 | [Runtime](runtime.md) |
| `[05-OP-23]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2787 | [Runtime](runtime.md) |
| `[05-OP-24]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2800 | [Runtime](runtime.md) |
| `[05-OP-42]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2821 | [Runtime](runtime.md) |
| `[05-OP-64]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2851 | [Runtime](runtime.md) |
| `[05-OP-46]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2885 | [Runtime](runtime.md) |
| `[05-OP-47]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2920 | [Runtime](runtime.md) |
| `[05-OP-48]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2946 | [Runtime](runtime.md) |
| `[05-OP-49]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 2975 | [Runtime](runtime.md) |
| `[05-OP-50]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3010 | [Runtime](runtime.md) |
| `[05-OP-51]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3037 | [Runtime](runtime.md) |
| `[05-OP-52]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3097 | [Runtime](runtime.md) |
| `[05-OP-53]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3131 | [Runtime](runtime.md) |
| `[05-OP-54]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3171 | [Runtime](runtime.md) |
| `[05-OP-55]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3208 | [Runtime](runtime.md) |
| `[05-OP-56]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3247 | [Runtime](runtime.md) |
| `[05-OP-57]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3284 | [Runtime](runtime.md) |
| `[05-OP-58]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3328 | [Runtime](runtime.md) |
| `[05-OP-59]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3353 | [Runtime](runtime.md) |
| `[05-OP-60]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3379 | [Runtime](runtime.md) |
| `[05-OP-61]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3419 | [Runtime](runtime.md) |
| `[05-OP-62]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3445 | [Runtime](runtime.md) |
| `[05-OP-63]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3471 | [Runtime](runtime.md) |
| `[05-UNS-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3843 | [Integration](integration.md) |
| `[05-UNS-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3855 | [Integration](integration.md) |
| `[05-UNS-3]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3862 | [Integration](integration.md) |
| `[05-UNS-4]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3867 | [Integration](integration.md) |
| `[05-UNS-5]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3872 | [Integration](integration.md) |
| `[05-UNS-6]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3881 | [Integration](integration.md) |
| `[05-OBS-1]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3893 | [Integration](integration.md) |
| `[05-OBS-2]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3901 | [Integration](integration.md) |
| `[05-OBS-3]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3907 | [Integration](integration.md) |
| `[05-OBS-4]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3950 | [Integration](integration.md) |
| `[05-OBS-5]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3959 | [Integration](integration.md) |
| `[05-OBS-6]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3964 | [Integration](integration.md) |
| `[05-OBS-7]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 4010 | [Integration](integration.md) |
| `[05-OBS-8]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 4022 | [Integration](integration.md) |
| `[05-OBS-9]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 4032 | [Integration](integration.md) |
| `[05-OBS-10]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 4038 | [Integration](integration.md) |
| `[05-OBS-11]` | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 4043 | [Integration](integration.md) |

## Chapter subject reconciliation

The account columns identify investigation ownership, not a transfer of language
or implementation authority. Cross-cutting subjects deliberately have more than
one chapter; their assumptions must connect in the overview.

| Chapter subject | Source location at the snapshot | Account |
|---|---|---|
| 1. What Chelis Is | [spec/00-context.md](../../../spec/00-context.md), line 3 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 2. What Chelis Is Not | [spec/00-context.md](../../../spec/00-context.md), line 18 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 3. Audience | [spec/00-context.md](../../../spec/00-context.md), line 28 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 4. Core Bet | [spec/00-context.md](../../../spec/00-context.md), line 39 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 5. Design Principles | [spec/00-context.md](../../../spec/00-context.md), line 45 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 6. Dual Syntax | [spec/00-context.md](../../../spec/00-context.md), line 71 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 7. Type System Scope | [spec/00-context.md](../../../spec/00-context.md), line 91 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 8. Computational Model | [spec/00-context.md](../../../spec/00-context.md), line 105 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 9. Backends and Execution | [spec/00-context.md](../../../spec/00-context.md), line 120 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 10. Key Influences | [spec/00-context.md](../../../spec/00-context.md), line 130 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 11. Reading Order | [spec/00-context.md](../../../spec/00-context.md), line 142 | [Language](language.md) / [overview](../soundness_obligations.md) |
| 1. Hard language constraints | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 25 | [Language](language.md) |
| 2. Filesystem and manifest layer | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 148 | [Language](language.md) |
| 3. Identifier conventions: Surf | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 272 | [Language](language.md) |
| 4. Identifier conventions: Rust | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 405 | [Language](language.md) |
| 5. Identifier conventions: Python | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 420 | [Language](language.md) |
| 6. Module conventions | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 435 | [Language](language.md) |
| 7. Function naming patterns | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 500 | [Language](language.md) |
| 8. Documentation conventions | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 730 | [Language](language.md) |
| 9. Project-cutting conventions | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 923 | [Language](language.md) |
| 10. Test naming | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 949 | [Language](language.md) |
| 11. Resolved decisions | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 987 | [Language](language.md) |
| 12. Enforcement | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 1092 | [Language](language.md) |
| 13. References | [spec/01-nomenclature.md](../../../spec/01-nomenclature.md), line 1364 | [Language](language.md) |
| 0. Notation | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 3 | [Language](language.md) |
| 1. Keywords | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 92 | [Language](language.md) |
| 2. Operator Precedence | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 114 | [Language](language.md) |
| 3. Punchlist Decisions | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 160 | [Language](language.md) |
| 4. Formal Grammar (PEG) | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 1085 | [Language](language.md) |
| 5. Complete Desugaring Reference | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 1443 | [Language](language.md) |
| 6. Parsing and Disambiguation Rules | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 1617 | [Language](language.md) |
| 7. Deep Vocabulary Boundary | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 1662 | [Language](language.md) |
| 8. Example Programs | [spec/02-surf-syntax.md](../../../spec/02-surf-syntax.md), line 1671 | [Language](language.md) |
| 1. Node Structure | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 11 | [Language](language.md) |
| 2. Tag Vocabulary | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 322 | [Language](language.md) |
| 3. Built-In Scope | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 626 | [Language](language.md) |
| 4. Function Application | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 659 | [Language](language.md) |
| 5. Pipe Semantics | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 724 | [Language](language.md) |
| 6. Canonical Form | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 746 | [Language](language.md) |
| 7. Grammar (PEG) | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 970 | [Language](language.md) |
| 8. Validation Rules | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1144 | [Language](language.md) |
| 9. Examples | [spec/03-deep-syntax.md](../../../spec/03-deep-syntax.md), line 1175 | [Language](language.md) |
| 0. Checked Deep Contract | [spec/04-type-system.md](../../../spec/04-type-system.md), line 8 | [Language](language.md) |
| 1. Type Representation | [spec/04-type-system.md](../../../spec/04-type-system.md), line 23 | [Language](language.md) |
| 2. Algebraic Data Types | [spec/04-type-system.md](../../../spec/04-type-system.md), line 356 | [Language](language.md) |
| 3. Hindley-Milner Inference | [spec/04-type-system.md](../../../spec/04-type-system.md), line 721 | [Language](language.md) |
| 4. Tensor Type Algebra | [spec/04-type-system.md](../../../spec/04-type-system.md), line 1021 | [Language](language.md) / [runtime](runtime.md) / [integration](integration.md) |
| 5. Precision Type Rules | [spec/04-type-system.md](../../../spec/04-type-system.md), line 1736 | [Language](language.md) / [runtime](runtime.md) / [integration](integration.md) |
| 6. Fitness Scoring | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2210 | [Language](language.md) / [runtime](runtime.md) / [integration](integration.md) |
| 7. Effects | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2436 | [Language](language.md) |
| 8. Linearity | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2523 | [Language](language.md) / [runtime](runtime.md) / [integration](integration.md) |
| 9. Numeric Value Semantics | [spec/04-type-system.md](../../../spec/04-type-system.md), line 2797 | [Runtime](runtime.md) |
| 10. Checker Totality | [spec/04-type-system.md](../../../spec/04-type-system.md), line 3139 | [Language](language.md) |
| 1. Design Principles | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 9 | [Runtime](runtime.md) |
| 2. RISC Primitives (Tier 1) | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 86 | [Runtime](runtime.md) |
| 3. Derived Built-Ins (Tier 2) | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 1063 | [Runtime](runtime.md) |
| 4. Standard Lowerings (Tier 2 → Tier 1) | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3495 | [Runtime](runtime.md) |
| 5. AD Completeness | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3715 | [Runtime](runtime.md) |
| 6. Reference Implementations | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3759 | [Runtime](runtime.md) |
| 7. The Unsupported-Case Response Contract | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3835 | [Integration](integration.md) / [runtime](runtime.md) |
| 8. Observation And Formatting Contract | [spec/05-risc-primitives.md](../../../spec/05-risc-primitives.md), line 3891 | [Integration](integration.md) / [runtime](runtime.md) |
| 1. Overview | [spec/06-transformations.md](../../../spec/06-transformations.md), line 3 | [Runtime](runtime.md) |
| 2. grad -- Reverse-Mode Automatic Differentiation | [spec/06-transformations.md](../../../spec/06-transformations.md), line 15 | [Runtime](runtime.md) |
| 3. vmap -- Vectorized Map | [spec/06-transformations.md](../../../spec/06-transformations.md), line 389 | [Runtime](runtime.md) |
| 4. jit -- Just-In-Time Compilation | [spec/06-transformations.md](../../../spec/06-transformations.md), line 511 | [Runtime](runtime.md) |
| 5. Optimization Passes | [spec/06-transformations.md](../../../spec/06-transformations.md), line 553 | [Runtime](runtime.md) |
| 6. Transformation Composition | [spec/06-transformations.md](../../../spec/06-transformations.md), line 704 | [Runtime](runtime.md) |
| 7. Formal Semantics of grad | [spec/06-transformations.md](../../../spec/06-transformations.md), line 766 | [Runtime](runtime.md) |
| 8. Error Conditions | [spec/06-transformations.md](../../../spec/06-transformations.md), line 884 | [Runtime](runtime.md) |
| 9. Pass Ordering and Convergence | [spec/06-transformations.md](../../../spec/06-transformations.md), line 935 | [Runtime](runtime.md) |
| 1. Implicit DAG Parallelism | [spec/07-concurrency.md](../../../spec/07-concurrency.md), line 3 | [Runtime](runtime.md) |
| 2. Explicit Parallelism: `par` | [spec/07-concurrency.md](../../../spec/07-concurrency.md), line 11 | [Runtime](runtime.md) |
| 3. Backend Mapping | [spec/07-concurrency.md](../../../spec/07-concurrency.md), line 20 | [Runtime](runtime.md) |
| 4. Non-Goals | [spec/07-concurrency.md](../../../spec/07-concurrency.md), line 28 | [Runtime](runtime.md) |
| 5. Related Features | [spec/07-concurrency.md](../../../spec/07-concurrency.md), line 37 | [Runtime](runtime.md) |
| 1. Backend Strategy | [spec/08-backends.md](../../../spec/08-backends.md), line 8 | [Runtime](runtime.md) |
| 2. Phase 0: C Backend | [spec/08-backends.md](../../../spec/08-backends.md), line 21 | [Runtime](runtime.md) |
| 3. Phase 1: HIP Backend | [spec/08-backends.md](../../../spec/08-backends.md), line 52 | [Runtime](runtime.md) |
| 4. Phase M: Metal Backend (macOS GPU peer) | [spec/08-backends.md](../../../spec/08-backends.md), line 229 | [Runtime](runtime.md) |
| 5. Later Integration Backends | [spec/08-backends.md](../../../spec/08-backends.md), line 434 | [Runtime](runtime.md) |
| 6. Interactive Execution | [spec/08-backends.md](../../../spec/08-backends.md), line 449 | [Runtime](runtime.md) |
| 7. Backend Selection | [spec/08-backends.md](../../../spec/08-backends.md), line 461 | [Runtime](runtime.md) |
| 8. Invariants | [spec/08-backends.md](../../../spec/08-backends.md), line 471 | [Runtime](runtime.md) |
| 1. Scope | [spec/09-tide.md](../../../spec/09-tide.md), line 9 | [Integration](integration.md) / [runtime](runtime.md) |
| 2. Interactive Execution Strategy | [spec/09-tide.md](../../../spec/09-tide.md), line 21 | [Integration](integration.md) / [runtime](runtime.md) |
| 3. Latency Policy | [spec/09-tide.md](../../../spec/09-tide.md), line 35 | [Integration](integration.md) / [runtime](runtime.md) |
| 4. Command Semantics | [spec/09-tide.md](../../../spec/09-tide.md), line 46 | [Integration](integration.md) / [runtime](runtime.md) |
| 5. Output Expectations | [spec/09-tide.md](../../../spec/09-tide.md), line 185 | [Integration](integration.md) / [runtime](runtime.md) |
| 6. Agreement Testing | [spec/09-tide.md](../../../spec/09-tide.md), line 196 | [Integration](integration.md) / [runtime](runtime.md) |
| 7. Later Tide Work | [spec/09-tide.md](../../../spec/09-tide.md), line 205 | [Integration](integration.md) / [runtime](runtime.md) |
| 8. Phase 2e Contract Notes | [spec/09-tide.md](../../../spec/09-tide.md), line 216 | [Integration](integration.md) / [runtime](runtime.md) |
| 9. Phase 2f Contract Notes | [spec/09-tide.md](../../../spec/09-tide.md), line 226 | [Integration](integration.md) / [runtime](runtime.md) |
| 10. Phase 2g Contract Notes | [spec/09-tide.md](../../../spec/09-tide.md), line 243 | [Integration](integration.md) / [runtime](runtime.md) |
| 1. Text Forms | [spec/10-serialization.md](../../../spec/10-serialization.md), line 3 | [Integration](integration.md) / [runtime](runtime.md) |
| 2. Binary Shell Form | [spec/10-serialization.md](../../../spec/10-serialization.md), line 10 | [Integration](integration.md) / [runtime](runtime.md) |
| 3. Compiler API Wire Contract | [spec/10-serialization.md](../../../spec/10-serialization.md), line 24 | [Integration](integration.md) / [runtime](runtime.md) |
| 4. Invariant Revalidation At Decode Boundaries | [spec/10-serialization.md](../../../spec/10-serialization.md), line 296 | [Integration](integration.md) / [runtime](runtime.md) |
| 1. Python Interop | [spec/11-ffi.md](../../../spec/11-ffi.md), line 3 | [Integration](integration.md) / [runtime](runtime.md) |
| 2. C Interop | [spec/11-ffi.md](../../../spec/11-ffi.md), line 102 | [Integration](integration.md) / [runtime](runtime.md) |
| 3. Embedding the Compiler | [spec/11-ffi.md](../../../spec/11-ffi.md), line 171 | [Integration](integration.md) / [runtime](runtime.md) |
| Phases | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 8 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Red Team Checkpoints | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 34 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Trust stack expansion and reef distribution | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 53 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Differentiable programming (committed scope) | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 87 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Hydronnx — ONNX shell (committed scope) | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 111 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Kerrent — GPU kernel authorship (committed scope) | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 141 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |
| Agent Editing Surface | [spec/12-roadmap.md](../../../spec/12-roadmap.md), line 179 | [Overview: scope](../soundness_obligations.md#scope-and-authority) |

## Workspace and non-Cargo surfaces

The locked, offline Cargo metadata view contains **33 workspace members**.
Every member below has a primary discussion route. This is a membership and
ownership check, not a claim that every public function or feature combination has
been validated. Cross-cutting obligations still apply: for example, the types
crate's numeric carrier participates in the runtime account, and Python tensor
admission participates in both runtime and integration.

The feature column records declarations, not enabled features or tested support.
An empty feature set does not remove platform `cfg`, external library, build-script,
environment, or runtime configuration obligations. Development-only observation
features must preserve ordinary behavior except for their declared observations.

| Workspace member | Obligation surface | Declared Cargo features | Primary account |
|---|---|---|---|
| [chelis-unord](../../../crates/chelis-unord/Cargo.toml) | Behavior-reaching order; explicit ordered consumption | None declared | [Integration](integration.md) |
| [chelis-repr-inventory](../../../crates/chelis-repr-inventory/Cargo.toml) | Representation surface discovery and its admission evidence | None declared | [Integration](integration.md) |
| [chelis-vocab](../../../crates/chelis-vocab/Cargo.toml) | Closed vocabulary shared by semantic consumers | None declared | [Language](language.md) |
| [chelis-prove](../../../crates/chelis-prove/Cargo.toml) | Obligation collection, solver models, certificates, and verdicts | `arb`, `carcara`, `clarabel`, `cvc5-rs`, `num-rational`, `smt`, `z3` | [Integration](integration.md) |
| [chelis-compiler-api](../../../crates/chelis-compiler-api/Cargo.toml) | Public pipeline entries, contexts, schemas, and artifact admission | `emission-observer` | [Integration](integration.md) |
| [chelis-backend-c](../../../crates/chelis-backend-c/Cargo.toml) | C lowering, generated host execution, and correctly rounded kernel emission | None declared | [Runtime](runtime.md) |
| [chelis-ir](../../../crates/chelis-ir/Cargo.toml) | DAG and host IR, evaluation, transforms, and ownership verification | `lowering-trace` | [Runtime](runtime.md) |
| [chelis-deep](../../../crates/chelis-deep/Cargo.toml) | Structural syntax, roles, metadata, and direct-producer ingress | None declared | [Language](language.md) |
| [chelis-effects](../../../crates/chelis-effects/Cargo.toml) | Effect inference, declaration checking, and handlers | None declared | [Language](language.md) |
| [chelis-types](../../../crates/chelis-types/Cargo.toml) | Typing, constraints, checked artifacts, numeric and linearity facts | `checkpoint-compile-probe`, `generalize-sweep-oracle` | [Language](language.md) |
| [chelis-pred](../../../crates/chelis-pred/Cargo.toml) | Predicate representation and transport to proof consumers | None declared | [Language](language.md) |
| [chelis-macros](../../../crates/chelis-macros/Cargo.toml) | Expansion, hygiene, and generated bindings | None declared | [Language](language.md) |
| [chelis-surf](../../../crates/chelis-surf/Cargo.toml) | Surface parsing, desugaring, formatting, and resugaring | None declared | [Language](language.md) |
| [chelis-runtime](../../../crates/chelis-runtime/Cargo.toml) | C ABI, representation, metadata, ownership, and foreign adoption | `ownership-ledger` | [Runtime](runtime.md) |
| [chelis-backend-hip](../../../crates/chelis-backend-hip/Cargo.toml) | Device code, launch domains, and host/device agreement | None declared | [Runtime](runtime.md) |
| [chelis-image-id](../../../crates/chelis-image-id/Cargo.toml) | Exact loaded-image identity | None declared | [Integration](integration.md) |
| [chelis-pipeline-core](../../../crates/chelis-pipeline-core/Cargo.toml) | Pipeline stage identity and shared orchestration | None declared | [Integration](integration.md) |
| [chelis-reef](../../../crates/chelis-reef/Cargo.toml) | Dependency resolution, installation, source closure, and lock identity | None declared | [Integration](integration.md) |
| [chelis-shell](../../../crates/chelis-shell/Cargo.toml) | Package contract and shared source/manifest validation | None declared | [Integration](integration.md) |
| [chelis-std-bundle](../../../crates/chelis-std-bundle/Cargo.toml) | Bundled standard-library identity and source selection | None declared | [Integration](integration.md) |
| [chelis-validate](../../../crates/chelis-validate/Cargo.toml) | Validation scope and truthful findings | None declared | [Integration](integration.md) |
| [chelis-version](../../../crates/chelis-version/Cargo.toml) | Compiler/version identity exposed to consumers | None declared | [Integration](integration.md) |
| [chelis-python](../../../crates/chelis-python/Cargo.toml) | Embedding, conversion, protocol negotiation, and tensor admission | `default`, `extension-module` | [Integration](integration.md) |
| [chelis-conformance](../../../crates/chelis-conformance/Cargo.toml) | Contract selection, comparison, and conformance reports | None declared | [Integration](integration.md) |
| [chelis-backend-metal](../../../crates/chelis-backend-metal/Cargo.toml) | Metal code, device limits, and host/device agreement | None declared | [Runtime](runtime.md) |
| [chelis-lint](../../../crates/chelis-lint/Cargo.toml) | Traversal coverage, findings, and autofix preservation | None declared | [Integration](integration.md) |
| [chelis-tide](../../../crates/chelis-tide/Cargo.toml) | Interactive state, edits, queries, and latency/report contracts | `smt` | [Integration](integration.md) |
| [chelis-lsp](../../../crates/chelis-lsp/Cargo.toml) | Document snapshots, analysis scope, and diagnostic provenance | None declared | [Integration](integration.md) |
| [chelis-cove](../../../crates/chelis-cove/Cargo.toml) | Structured editing, whole-module admission, and preimages | None declared | [Integration](integration.md) |
| [tree-sitter-chelis](../../../tree-sitter-chelis/Cargo.toml) | Editor grammar and generated-parser agreement | None declared | [Language](language.md) |
| [chelis-cli](../../../crates/chelis-cli/Cargo.toml) | Command routing, failures, output, test/proof orchestration | `chelis-prove`, `default`, `smt` | [Integration](integration.md) |
| [chelisup](../../../crates/chelisup/Cargo.toml) | Toolchain installation, selection, shims, and rollback | None declared | [Integration](integration.md) |
| [chelis-e2e](../../../crates/chelis-e2e/Cargo.toml) | Executable agreement and acceptance evidence | `hip-local-gpu` | [Integration](integration.md) |

Cargo membership does not enumerate the whole product. These additional
source-discovery boundaries must remain visible:

| Additional surface | Discovery source | Account and residual boundary |
|---|---|---|
| Python package and packaging metadata | [bindings/python](../../../bindings/python) | [Integration](integration.md): Python facade and native-extension agreement, conversions, wheel/runtime identity. |
| Editor extensions and separately packaged grammars | [editors](../../../editors), [grammars](../../../grammars), [tree-sitter-chelis](../../../tree-sitter-chelis) | [Language](language.md) and [integration](integration.md): grammar/query parity and document identity; this account is not an execution audit of every editor packaging combination. |
| Public headers, generated source, and build scripts | [runtime](../../../crates/chelis-runtime), [backends](../../../spec/08-backends.md), workspace build targets | [Runtime](runtime.md): ABI attribution, safe code generation, compiler flags, foreign dependencies, and generated-consumer route closure. |
| Bundled and external language libraries | [standard-library bundler](../../../crates/chelis-std-bundle), [semantic registries](../../../spec/registry), Reef source closure | [Runtime](runtime.md) covers registered semantics; [integration](integration.md) covers dependency and installed-source identity. External shell implementations are not independently audited here. |
| Test discovery, guard generators, CI, and release artifacts | [scripts](../../../scripts), [workflows](../../../.github/workflows), [phase oracles](../../phase_oracles.md) | [Integration](integration.md): execution receipts, negative controls, skip/failure accounting, exact artifact identity. This source account does not certify hosted policy or an actual release. |
| Toolchain/environment definitions and native dependencies | [Cargo.lock](../../../Cargo.lock), [flake.nix](../../../flake.nix), package/build manifests | [Integration](integration.md) and [runtime](runtime.md): correct selected dependencies and explicit trusted assumptions; no theorem about Rust/C compilers, solvers, operating systems, drivers, or hardware is supplied. |
| Planned/new entry kinds | [Roadmap scope](../soundness_obligations.md#scope-horizon) | New syntax, IR layers, model import, capability checks, and targets extend the relevant independent inventories before they can inherit an assurance claim. |

## Historical-class crosswalk

The live `tracking` label query on 2026-09-09 returned **35 hubs**, across open
and closed issues. The label describes a hub as a class **or a plan**, not a work
item. This table reconciles all 35 navigation entries with the obligation map;
it does not claim that every hub is a distinct soundness class or that an
issue's title remains an accurate description of current code.

This is deliberately not a full issue-thread status, parentage, or closure audit.
In particular, the [launch ledger](https://github.com/Chelis-Lang/chelis/issues/1362)
selects a release subset and carries evidence from its own named commits. Neither
its device/Python exclusions nor a closed historical witness erase the broader
language/toolchain obligation. Unparented defects also remain in scope when their
behavior falls under the contracts.

| Tracking hubs | Historical subject | Obligation account |
|---|---|---|
| [#703](https://github.com/Chelis-Lang/chelis/issues/703), [#730](https://github.com/Chelis-Lang/chelis/issues/730), [#1170](https://github.com/Chelis-Lang/chelis/issues/1170), [#1226](https://github.com/Chelis-Lang/chelis/issues/1226) | Loud rejection instead of fabricated execution | [Unsupported domains, typed failure, concrete type-application settlement](runtime.md) |
| [#727](https://github.com/Chelis-Lang/chelis/issues/727), [#729](https://github.com/Chelis-Lang/chelis/issues/729) | Numeric authority and exact values | [Operation, dtype, width, domain, reduction, and adjoint contracts](runtime.md) |
| [#728](https://github.com/Chelis-Lang/chelis/issues/728), [#732](https://github.com/Chelis-Lang/chelis/issues/732), [#883](https://github.com/Chelis-Lang/chelis/issues/883), [#912](https://github.com/Chelis-Lang/chelis/issues/912) | Observation, roots, and diagnostics | [Faithful values and errors, manifested roots, location and report provenance](integration.md) |
| [#731](https://github.com/Chelis-Lang/chelis/issues/731), [#874](https://github.com/Chelis-Lang/chelis/issues/874), [#908](https://github.com/Chelis-Lang/chelis/issues/908) | Checker coverage and valid representations | [All semantic positions and entry carriers; no unvisited or inadmissible state accepted as checked](language.md) |
| [#1024](https://github.com/Chelis-Lang/chelis/issues/1024) | Canonical source and round trips | [Grammar, normalization, migration, representability exceptions](language.md) |
| [#893](https://github.com/Chelis-Lang/chelis/issues/893), [#1286](https://github.com/Chelis-Lang/chelis/issues/1286) | Runtime storage and ownership | [Representation, capacity, aliasing, transfer, lifetime and reclaim](runtime.md) |
| [#1277](https://github.com/Chelis-Lang/chelis/issues/1277) | Runtime extent relationships | [Independent claims and observations, witness transport, guard timing](runtime.md) |
| [#1372](https://github.com/Chelis-Lang/chelis/issues/1372) | Graph reconstruction | [Every side annotation and dependency preserved or revalidated by every rewrite](runtime.md) |
| [#909](https://github.com/Chelis-Lang/chelis/issues/909) | Typed host callability | [Function identity, closures, concrete application types, calling convention and ownership](runtime.md) |
| [#1341](https://github.com/Chelis-Lang/chelis/issues/1341), [#1479](https://github.com/Chelis-Lang/chelis/issues/1479) | Behavior-reaching order | [Explicit semantic ordering; deterministic observations under the stated contract](integration.md) |
| [#1439](https://github.com/Chelis-Lang/chelis/issues/1439) | Artifact and installation path identity | [Generated/installed artifacts resolve the right environment at their use boundary](integration.md) |
| [#754](https://github.com/Chelis-Lang/chelis/issues/754), [#788](https://github.com/Chelis-Lang/chelis/issues/788), [#1096](https://github.com/Chelis-Lang/chelis/issues/1096) | Independent cross-lane and ecosystem evidence | [Actual execution, independent semantic checks, shell contract and enforcement boundary](integration.md) |
| [#694](https://github.com/Chelis-Lang/chelis/issues/694), [#733](https://github.com/Chelis-Lang/chelis/issues/733), [#740](https://github.com/Chelis-Lang/chelis/issues/740), [#803](https://github.com/Chelis-Lang/chelis/issues/803), [#808](https://github.com/Chelis-Lang/chelis/issues/808), [#1568](https://github.com/Chelis-Lang/chelis/issues/1568) | Measurement, review, coverage and governance | [Truthful claims, enforced review scope, measured coverage, negative controls, future provenance distinguished from active enforcement](integration.md) |
| [#828](https://github.com/Chelis-Lang/chelis/issues/828), [#830](https://github.com/Chelis-Lang/chelis/issues/830) | Performance and progress commitments | [Explicit resource/progress domains and measurements; optimization still preserves semantics](integration.md) |
| [#1129](https://github.com/Chelis-Lang/chelis/issues/1129), [#1362](https://github.com/Chelis-Lang/chelis/issues/1362) | Release and delivery coordination | [Historical landing/release ledgers are navigation and acceptance scope, not language authority](../soundness_obligations.md) |

The direction of reasoning is from the contract to the issue, not from the issue
to the contract's outer limit. The account additionally covers ordinary binding
and abstraction, operation/AD correctness, effect control, boundary admission,
cache/context identity, tool transactions, proof assumptions, and external trust
without requiring a pre-existing tracking hub for each.

## Reproducing the membership check

Use the checkout's managed Python and the existing parser; this reads the current
numbered sources and fails if their normative atom IDs are duplicated:

```sh
.venv/bin/python -B -c 'from pathlib import Path; from scripts.generate_rejection_registries import discover_atoms; print("\n".join(discover_atoms(Path("spec"))))'
```

The source-heading inventory was read with `rg -n '^## ' spec/[0-9][0-9]*.md`.
Workspace membership and feature names come from
`cargo metadata --no-deps --locked --offline --format-version 1`, selecting the
packages in `workspace_members`. The historical crosswalk comes from
`gh issue list --repo Chelis-Lang/chelis --state all --label tracking --limit 200 --json number,title,url`.
The actual diagnostic receipt and any reconciliation limitations are recorded in
the overview. Adding an atom to this table alone is never enforcement evidence.
