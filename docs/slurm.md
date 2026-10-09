# SLURM

Submit a training command as a batch job with the slurm command.
Options before, arguments after `--` are forwarded to the training command.

## Jobs

```bash
# distilbert
ehpt slurm fine-tune --job-name distilbert-head-only --time 00:10:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m distilbert/distilbert-base-cased -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name distilbert-fine-tune --time 00:15:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m distilbert/distilbert-base-cased -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name distilbert-prompt-tune-random --time 00:15:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m distilbert/distilbert-base-cased -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name distilbert-prompt-tune-pretrained --time 00:15:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m distilbert/distilbert-base-cased -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

# mmBERT
ehpt slurm fine-tune --job-name mmbert-head-only --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-small -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name mmbert-fine-tune --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-small -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name mmbert-prompt-tune-random --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-small -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name mmbert-prompt-tune-pretrained --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-small -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name mmbert-fine-tune --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-base -d multinerd -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name mmbert-head-only --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-base -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name mmbert-prompt-tune-random --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-base -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name mmbert-prompt-tune-pretrained --time 00:30:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m jhu-clsp/mmBERT-base -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

# EuroBERT
ehpt slurm fine-tune --job-name eurobert-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-210m -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name eurobert-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-210m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-210m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-210m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name eurobert-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-610m -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name eurobert-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-610m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-610m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-610m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name eurobert-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-2.1B -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name eurobert-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-2.1B -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-2.1B -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name eurobert-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m EuroBERT/EuroBERT-2.1B -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

# DeBERTa v3
ehpt slurm fine-tune --job-name deberta-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-xsmall -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name deberta-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-xsmall -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-xsmall -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-xsmall -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name deberta-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-small -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name deberta-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-small -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-small -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-small -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name deberta-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-base -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name deberta-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-base -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-base -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-base -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

ehpt slurm fine-tune --job-name deberta-head-only --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-large -d multinerd --head-only -t 20000 -l debug -s 42
ehpt slurm fine-tune --job-name deberta-fine-tune --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-large -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-random --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-large -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name deberta-prompt-tune-pretrained --time 01:00:00 --mem 16GB --cpus 32 --gres gpu:h200-141g:1 -- -m microsoft/deberta-v3-large -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42

# Pythia (GPTNeoX)
ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-70m -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-70m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-70m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-70m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-70m -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-160m -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-160m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-160m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-160m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-160m -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-410m -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-410m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-410m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-410m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-410m -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1b -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1.4b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1.4b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1.4b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1.4b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-1.4b -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-2.8b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-2.8b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-2.8b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-2.8b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-2.8b -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gptneox-few-shot --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-6.9b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gptneox-fine-tune --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-6.9b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-random --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-6.9b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gptneox-prompt-tune-pretrained --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-6.9b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gptneox-qlora --time 03:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m EleutherAI/pythia-6.9b -d multinerd -t 20000 -l debug -s 42

# Qwen3.5
ehpt slurm few-shot --job-name qwen35-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-0.8B -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name qwen35-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-0.8B -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-0.8B -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-0.8B -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name qwen35-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-0.8B -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name qwen35-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-2B -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name qwen35-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-2B -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-2B -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-2B -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name qwen35-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-2B -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name qwen35-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-4B -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name qwen35-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-4B -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-4B -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-4B -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name qwen35-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-4B -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name qwen35-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-9B -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name qwen35-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-9B -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-9B -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name qwen35-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-9B -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name qwen35-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m Qwen/Qwen3.5-9B -d multinerd -t 20000 -l debug -s 42

# Llama 3
ehpt slurm few-shot --job-name llama32-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-1B-Instruct -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name llama32-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-1B-Instruct -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-1B-Instruct -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-1B-Instruct -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name llama32-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-1B-Instruct -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name llama32-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-3B-Instruct -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name llama32-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-3B-Instruct -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-3B-Instruct -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-3B-Instruct -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name llama32-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.2-3B-Instruct -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name llama32-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.1-8B-Instruct -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name llama32-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.1-8B-Instruct -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.1-8B-Instruct -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name llama32-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.1-8B-Instruct -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name llama32-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m meta-llama/Llama-3.1-8B-Instruct -d multinerd -t 20000 -l debug -s 42

# Gemma 3
ehpt slurm few-shot --job-name gemma3-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-270m-it -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gemma3-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-270m-it -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-270m-it -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-270m-it -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gemma3-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-270m-it -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gemma3-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-1b-it -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gemma3-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-1b-it -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-1b-it -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-1b-it -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gemma3-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-1b-it -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name gemma3-few-shot --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-4b-it -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name gemma3-fine-tune --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-4b-it -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-random --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-4b-it -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name gemma3-prompt-tune-pretrained --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-4b-it -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name gemma3-qlora --time 04:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/gemma-3-4b-it -d multinerd -t 20000 -l debug -s 42

# Flan-T5
ehpt slurm few-shot --job-name flant5-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-small -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name flant5-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-small -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-small -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-small -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name flant5-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-small -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name flant5-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-base -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name flant5-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-base -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-base -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-base -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name flant5-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-base -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name flant5-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-large -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name flant5-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-large -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-large -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-large -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name flant5-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-large -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name flant5-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xl -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name flant5-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xl -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xl -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xl -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name flant5-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xl -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name flant5-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xxl -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name flant5-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xxl -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xxl -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name flant5-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xxl -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name flant5-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/flan-t5-xxl -d multinerd -t 20000 -l debug -s 42

# T5Gemma 2
ehpt slurm few-shot --job-name t5gemma-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-270m-270m -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name t5gemma-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-270m-270m -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-270m-270m -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-270m-270m -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name t5gemma-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-270m-270m -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name t5gemma-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-1b-1b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name t5gemma-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-1b-1b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-1b-1b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-1b-1b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name t5gemma-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-1b-1b -d multinerd -t 20000 -l debug -s 42

ehpt slurm few-shot --job-name t5gemma-few-shot --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-4b-4b -d multinerd -l debug -s 42
ehpt slurm fine-tune --job-name t5gemma-fine-tune --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-4b-4b -d multinerd -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-random --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-4b-4b -d multinerd --prefix-init random -t 20000 -l debug -s 42
ehpt slurm prompt-tune --job-name t5gemma-prompt-tune-pretrained --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-4b-4b -d multinerd --prefix-init pretrained -t 20000 -l debug -s 42
ehpt slurm qlora --job-name t5gemma-qlora --time 05:00:00 --mem 32GB --cpus 32 --gres gpu:h200-141g:1 -- -m google/t5gemma-2-4b-4b -d multinerd -t 20000 -l debug -s 42
```
