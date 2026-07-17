#1.先运行模型，生成回答
bash scripts/run_smoke_test.sh 2 DeepSeek-V4-Pro
#或者是完整实验
bash scripts/run_tablebench_experiments.sh


#2.运行一次评估，得到官方的指标分数
python3 scripts/convert_react_to_tablebench_eval.py   --raw_results_file outputs/DeepSeek-V4-Pro_tablebench/raw_react_results.jsonl   --output_dir outputs/DeepSeek-V4-Pro_tablebench/ 

#3.再运行一次 LLM as a judge，得到 LLM 评估的指标分数
python scripts/judge_tablebench_eval.py \
  --results_file outputs/<model>_tablebench/tablebench_eval_results.jsonl \
  --judge_model <judge-model-name>  