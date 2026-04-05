# RNN/GRU Code Demos

If the project venv has not been attached yet, run:

```bash
PYTHON_BIN=python3.12 SHARED_ENV_NAME=py312-torch-cu130 \
  bash .agents/skills/shared-python-env/scripts/setup_shared_env.sh attach "$(pwd)"
```

Then run the demos from the repo root with the attached shared environment:

```bash
source .venv/bin/activate
python notes/RNN_GRU_基础/code/plain_rnn_demo.py
python notes/RNN_GRU_基础/code/gru_demo.py
python notes/RNN_GRU_基础/code/long_memory_benchmark.py
```

## Easy Toy Demo

`plain_rnn_demo.py` and `gru_demo.py` train tiny sentiment models on a very
small dataset with short sentences such as `movie is good` and
`movie is not good`.

This setup is intentionally easy. Its main purpose is to show the common
recurrent mechanism, not to prove a large performance gap. On such short and
simple small-sample tasks, plain RNN and GRU can look very similar because
both models can memorize the pattern.

## Harder Comparison Demo

`long_memory_benchmark.py` is the more discriminative comparison.

- The important bit is written near the beginning of the sequence.
- Training sees only short sequences.
- Evaluation includes much longer sequences.
- The script prints an accuracy table for plain RNN and GRU, plus the gap.

This benchmark is more informative because it stresses memory retention and
length generalization. That is the kind of setting where GRU's gated update
often matters more in practice. It still does not prove that GRU must always
beat plain RNN; it simply gives the comparison a harder and more honest task.
