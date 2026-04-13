module Std.Nn.Embedding
export (forward)
def forward[batch, seq, vocab, hidden](ids: tensor[batch, seq, int64], table: tensor[vocab, hidden, f32]) -> tensor[batch, seq, hidden, f32] = gather(table, ids, 0)
