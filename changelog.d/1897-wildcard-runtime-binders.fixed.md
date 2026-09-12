Fixed evaluator closure setup incorrectly equating independent wildcard tensor
axes or arguments. Functions accepting `tensor[*, *, f32]` now accept nonsquare
and empty shapes; genuine repeated named-dimension constraints remain enforced.
