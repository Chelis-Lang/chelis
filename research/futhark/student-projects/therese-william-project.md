# University of Copenhagen Computer Science Department: The Worlds Best Futhark Formatter

## Metadata
- **Authors:** Therese Lyngby & William Henrich Due
- **Venue/Year:** University of Copenhagen, 2024
- **Advisor:** Troels Henriksen

## Summary
This report details the extension of a rudimentary formatter for the Futhark programming language. The formatter is designed to take Futhark programs and make them "pretty" while preserving the original syntax tree, layout, and comment structure. The authors define specific properties that the formatter must have to produce nicely formatted programs, including nondestructiveness, idempotence, and comment order retention. The formatter is implemented using Haskell's PrettyPrinter library and a custom monad called FmtM, which handles formatting, comment insertion, and layout adaptation. The correctness of the formatter is asserted through property-based testing using the Futhark repository's test programs.

## Key Contributions
- Design and implementation of a nonconfigurable Futhark formatter that preserves original layout and comment structure
- Development of a custom formatting monad (FmtM) that handles comment insertion, adaptive formatting, and layout management
- Definition of properties for formatter correctness including nondestructiveness, idempotence, and comment order retention
- Property-based testing methodology using real Futhark programs from the repository
- Discussion of challenges in comment placement and adaptive formatting strategies

## Technical Approach
The formatter uses Haskell's PrettyPrinter library for the actual formatting work. The core innovation is the FmtM monad, which combines reader and state monads to handle formatting context. The reader part contains layout information (single-line vs multi-line), while the state part manages comments, pending comments, and tracking the last output type.

The formatter uses adaptive functions that change behavior based on layout context. For example, the `line` primitive becomes a space in single-line mode, and `indent` becomes a no-op. This allows users to write multi-line formatting that automatically adapts to single-line contexts.

Comment handling is particularly complex since comments aren't part of the syntax tree. The formatter extracts comments during parsing and reinserts them during formatting, maintaining their original order and placement context (prepended vs trailing). The system uses a pending comment mechanism to handle trailing comments that appear on the same line as code.

## Results
The formatter passes all property-based tests for nondestructiveness, idempotence, and comment order retention. Testing was conducted using 2305 test programs from the Futhark repository. The formatter successfully preserves syntax trees, maintains comment order, and produces aesthetically pleasing output in most cases.

However, some issues remain, particularly with pattern matching on records where the formatter adds redundant information due to parser limitations. The authors note that comment placement could be improved to reduce the burden on users.

## Relevance
This work is relevant to language design and compiler development as it demonstrates techniques for building robust code formatters that preserve program semantics while improving readability. The adaptive formatting approach and comment handling strategies could be applied to other programming languages. The property-based testing methodology using real programs provides a model for verifying formatter correctness. The work also highlights challenges in handling language features that don't map cleanly to syntax trees, such as comments and optional syntax elements.
