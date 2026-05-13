module Std.Nn.Embedding
export (forward)
def forward[batch, seq, vocab, hidden, p](ids: &tensor[batch, seq, int64], table: &tensor[vocab, hidden, p]) -> tensor[batch, seq, hidden, p] = gather(table, ids, 0)
