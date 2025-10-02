1) Data pipeline (you implement)

Goal: turn raw text into integer sequences and (input, target) pairs.

Steps

Read text → build sorted vocab chars and mappings stoi, itos.

Encode text → torch.LongTensor of ids.

For a chosen seq_len = T, create sliding windows:

x[i] = ids[i : i+T]

y[i] = ids[i+1 : i+T+1] (shifted by 1)

Split train/val (e.g., last 5% for val) or a random split.

DataLoader with batch_first=True returning (B, T) tensors.

Shape checks (acceptance)

Batch (xb, yb) is LongTensor with shape (B, T).

max(xb) < vocab_size, max(yb) < vocab_size.

2) Model (you implement)

Architecture:
Embedding(vocab_size, E) → LSTM(E, H, num_layers=L, batch_first=True, dropout=(L>1?d:0)) → Linear(H, vocab_size).

Forward contract

Input: x of shape (B, T) (long).

Output: logits of shape (B, T, V), and hidden state.

No softmax; you’ll pass raw logits to CrossEntropyLoss.

Sanity assertion

If B=4, T=128, V=65, logits.shape == (4, 128, 65).

3) Training loop (you implement)

Optimizer: AdamW(model.parameters(), lr=3e-3)

Loss: CrossEntropyLoss() with reshapes:

loss = CE(logits.view(B*T, V), targets.view(B*T))

Optional: gradient clipping clip_grad_norm_(params, 1.0)

Track: train loss per epoch; evaluate on val set each epoch.

Target ranges (with the tiny dataset)

With E=256, H=512, L=2, T=128, B=64, epochs=5:

Expect train CE to fall below ~2.2 and val near ~2.3–2.6 on small corpora.

Perplexity exp(loss) ≈ 9–13. (Smaller dataset → higher, noisier.)

4) Inference / sampling (you implement)

Greedy or temperature sampling loop:

Algorithm sketch

Start from a prompt string (e.g., "To be"). Encode to ids on device.

Keep hidden=None initially. Repeatedly:

Feed the last token (shape (B=1, T=1)) through the model.

Take the last-step logits logits[:, -1, :] / temperature.

Sample next_id = multinomial(softmax(logits)) (or argmax).

Append to output and use it as next input token.

Stop after N steps; decode ids→chars.

Acceptance

Loop generates text without crashing.

Temperature ∈ {0.7, 1.0, 1.3} changes diversity as expected.

5) Hyperparameters to start

seq_len T = 128 (try 64 for speed)

batch_size B = 64

embed_dim E = 256

hidden_size H = 512 (try 256 for speed; 1024 for capacity)

num_layers L = 2

dropout d = 0.1 (effective only if L > 1)

epochs = 5 (do 2–3 for a smoke test)

lr = 3e-3

clip = 1.0

device = cuda if available else cpu

6) What to record (your lab notes)

Make a small table:

Config (E/H/L/T)	Train CE	Val CE	Val PPL	Sample (first 100 chars)
256/256/1/128	…	…	…	“…”
256/512/2/128	…	…	…	“…”
256/512/2/64	…	…	…	“…”

Add short bullets:

Effect of hidden size on val perplexity and sample quality.

Effect of sequence length on stability/quality vs speed.

Does clipping help avoid spikes? (Watch loss curves.)

7) Quick math refresh (why no softmax layer?)

Logits at each step: 
𝑧
𝑡
=
𝑊
ℎ
𝑡
+
𝑏
∈
𝑅
𝑉
z
t
	​

=Wh
t
	​

+b∈R
V

CrossEntropyLoss internally does log_softmax(z_t) + NLL, so you don’t add softmax in your model.

Perplexity 
p
p
l
=
𝑒
CE
ppl=e
CE

8) Fast checkpoints / acceptance tests

Unit check: pass a tiny batch (B=2, T=3) through model; verify shapes match.

Overfit a tiny slice: train on 256 contiguous chars for ~200 steps; loss should drop < 0.5 (confirms your pipeline).

Sampling: with temperature 0.7, you should see prompt-coherent continuations (not just repeated letters).

9) Next labs you can chain after this

Many-to-one (sequence classification): reuse LSTM, pool over time (last/mean/max), add a classifier head.

Seq2Seq (unaligned many-to-many): encoder LSTM → decoder LSTM with teacher forcing; try reversing strings or date formats.

Minimal task checklist (so you can tick items)

 Build vocab, encode text; dataset yields (B, T) pairs

 LSTM model forward returns (B, T, V) logits

 CE loss over reshaped logits/targets

 Training loop with AdamW + clipping

 Eval loop (val CE & ppl)

 Sampling with temperature

 Two ablations (H size; seq_len) + brief notes

If you want, I can add a tiny experiments.md template and hatch scripts like train_lm and sample_lm that just call your modules (no model code), so you can focus on implementations.