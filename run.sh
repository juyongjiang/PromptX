# Reflexion
# python main.py --run_name "test_reflexion_gpt3.5" \
#   --root_dir "data/trajectory" \
#   --dataset_path ./data/benchmarks/humaneval-py-test.jsonl \
#   --strategy "reflexion" \
#   --language "py" \
#   --model "gpt-3.5-turbo-0613" \
#   --pass_at_k "1" \
#   --max_iters "8" \
#   --verbose 2>&1 | tee test_reflexion_gpt3.5.log

# python main.py \
#   --run_name "test_reflexion_gpt4" \
#   --root_dir "data/trajectory" \
#   --dataset_path ./data/benchmarks/humaneval-py-test.jsonl \
#   --strategy "reflexion" \
#   --language "py" \
#   --model "gpt-4" \
#   --pass_at_k "1" \
#   --max_iters "8" \
#   --verbose 2>&1 | tee test_reflexion_gpt4.log


# LATS
python main.py \
  --run_name "test_lats_gpt4" \
  --root_dir "data/trajectory" \
  --dataset_path ./data/benchmarks/humaneval-py-test.jsonl \
  --strategy "lats" \
  --language "py" \
  --model "gpt-4" \
  --pass_at_k "1" \
  --max_iters "8" \
  --verbose 2>&1 | tee test_lats_gpt4.log

# python main.py \
#   --run_name "test_lats_gpt3.5" \
#   --root_dir "data/trajectory" \
#   --dataset_path ./data/benchmarks/humaneval-py-test.jsonl \
#   --strategy "lats" \
#   --language "py" \
#   --model "gpt-3.5-turbo-0613" \
#   --pass_at_k "1" \
#   --max_iters "8" \
#   --verbose 2>&1 | tee test_lats_gpt3.5.log