# My Parameter Golf Journey

## Where I'm At

I'm competing in OpenAI's Model Craft Challenge: Parameter Golf — training the best language model that fits inside a 16MB artifact in under 10 minutes on 8xH100s, evaluated by bits per byte on the FineWeb validation set.

**Current best:** `val_bpb ~2.02`  
**Hardware:** RTX 3080 (10GB)  
**Key wins so far:** batch size reduction + careful LR tuning

Getting to 2.02 on a consumer 3080 involved a fair amount of constraint juggling — 10GB VRAM means batch sizes that would make a cloud GPU laugh, so a lot of early iteration was just finding the regime where things don't OOM before they converge. Still a long way from the SOTA crowd (currently pushing below 1.12), but the local iteration loop has been genuinely useful for understanding what actually moves the needle before throwing money at H100 time.

---

## The Challenge

**OpenAI Model Craft Challenge: Parameter Golf** is a competition to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. The organisers are excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...).

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.

Leaderboard submissions are limited to 10 minutes on 8xH100s to keep things accessible compute-wise. However, the challenge also welcomes submissions that don't meet the compute limitation in the 'Non-record Submissions' section — pushing the infinite frontier of parameter-limited performance is fair game too.

**OpenAI is sponsoring $1,000,000 in compute credits** to help people get started. To request a compute grant: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form). Make sure to choose the appropriate level, write sufficient justification, and submit with an email tied to an OpenAI / ChatGPT account.

The challenge runs from **March 18th to April 30th**.

---

## Participant Form

If you enjoy solving very difficult technical problems, introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps with attribution and reaching out about opportunities with OpenAI. Completing the form is not required to participate.

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor.

In June, OpenAI plans to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors.

---

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, there's a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version without MLX support. Ask [Codex](https://openai.com/codex/) to refactor it — the change is straightforward. It may still be fairly slow, so jumping straight to cloud GPUs with Runpod is worth considering.

First, clone the repository, create a fresh Python environment, and install the packages:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download the cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`. By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass `--train-shards 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, the fixed first-50k-document set. The smoke command above skips periodic validation and prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with local tests, switch to a remote CUDA machine. OpenAI is partnering with Runpod to make setup easy.

#### Launching a 1xH100 Pod

1. [Create a Runpod account](https://console.runpod.io/deploy) and set up an SSH key in the Settings tab.

2. Create a new GPU Cloud Pod with whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (SXM variant specifically), but test and iterate on cheaper SKUs first — an 8xH100 box runs around $20/hour.

3. Start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access and deploy.

On your remote machine:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download the cached FineWeb dataset:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the full validation split plus 80 training shards (8B tokens). Pass `--train-shards N` for a smaller subset while iterating.

Launch your first training run:

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. To override: `MAX_WALLCLOCK_SECONDS=0`.

The script prints `train_loss` step logs during training and `val_loss`, `val_bpb`, and compressed model size at the end. For periodic validation logs during a run, set `VAL_LOSS_EVERY=200`. The baseline config should land around `val_bpb ~1.2` with a compressed model size under 16MB.

For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](data/README.md).

---

## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script. The cap is decimal 16MB (16,000,000 total bytes, not 16 MiB). No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

Not automatically for every submission, but top leaderboard entries will be verified over time. Non-reproducible results can be disqualified. If you find a record isn't reproducible, raise a GitHub Issue.

**What counts as 'external compute'?**

Tuning Adam hyperparameters across runs is fine. Brute-forcing seeds or otherwise sneaking in additional compute unfairly is not. Use your best judgment — there's no penalty for asking questions.

**What are the restrictions on evaluation?**

Submissions can't take more than 10 minutes on 8xH100 to evaluate (in addition to the 10 minutes of training time). Evaluation at any sequence length is allowed. You cannot access any training data during evaluation unless you pay for those bits within the 16MB limit. You cannot access validation data during training.

One clarification on test-time training: you are only allowed to test-time train on validation set tokens you've already evaluated your model on, since those tokens have already been graded.

**What is the process for accepting new submissions?**

Submissions are accepted chronologically by PR creation time. The leaderboard may take time to update due to verification. Submissions must exceed the SOTA record with sufficient statistical significance to be accepted. Otherwise they may be accepted as 'non-record submissions' if sufficiently unique or interesting.

**Can I import XYZ package or library?**

Yes, so long as it doesn't violate the rules on evaluation, compute, training time, or code size. Include a `requirements.txt` in your records folder and mention setup instructions in your README. You can't sneak in extra compute or capabilities through custom libraries, but importing FlashAttention etc. is completely fine.

---

## Submission Process

New SOTA records must fulfil the following criteria:

1. Beat the existing SOTA by at least 0.005 nats, demonstrated at `p < 0.01` via run logs. This requirement is waived for submissions that improve speed through systems optimisation without changing the ML.

2. If changes are made to the tokenizer or dataset, prove with certainty that `val_bpb` is correctly calculated. Tokenizer edits will be examined carefully.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be a pull request adding a new folder to the appropriate `/records` subfolder, containing:

1. A `README.md` explaining the submission in reasonable detail.
2. A `submission.json` file with your name, GitHub ID, `val_bpb`, and related metadata.
3. A train log showing a statistically significant win (typically an average over 3 runs).
4. A `train_gpt.py` script and any other dependencies. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that don't beat the existing SOTA but still satisfy the 16MB artifact limit. Weird or out-of-the-box ideas, unoptimized solutions, and interesting negative results are all welcome. Include a `requirements.txt` and detailed justification in your README.

The unlimited compute track accepts runs not intended to meet the 10-minute cutoff — just note as such in your README.

#### PRs on Core Code

`train_gpt.py` and `train_gpt_mlx.py` are intended as good starting points, not SOTA configs. PRs that tune, improve, or simplify these scripts without significantly increasing complexity are welcome, but the best models should live in `/records`.

---

## Support

Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit **#parameter-golf-discussions** and **#parameter-golf-announcements**.

This repository adapts code from `modded-nanogpt` — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
