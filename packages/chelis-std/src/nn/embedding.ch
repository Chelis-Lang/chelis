module Std.Nn.Embedding
export (forward)
sig forward: &tensor[batch, seq, int64] -> &tensor[vocab, hidden, p] -> tensor[batch, seq, hidden, p]
def forward[batch, seq, vocab, hidden, p](ids, table) = gather(table, ids, 0)
