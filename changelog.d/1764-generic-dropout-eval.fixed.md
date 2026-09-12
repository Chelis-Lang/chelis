Generic fixed-rate dropout helpers now evaluate through the checked execution
plan, preserving concrete float dtypes, gradient replay, and the following
random draw. This fixes the evaluator's `unknown runtime name dropout` error
for source-static casts such as `cast(0.5, p)`.
