# :robot: PromptCoder

<p align="center" width="100%">
<img src="assets/promptcoder.png" alt="PromptCoder" style="width: 50%; min-width: 100px; display: block; margin: auto;">
</p>

## Setup

To get started:

1. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

2. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

3. Set the scripts and run paper experiments
```bash
sh run_lats.sh
sh run_reflexion.sh
```


## Optional Arguments

run `python main.py --help` for more details

* `--root_dir`: The root logging directory
* `--run_name`: The name of the run logging file 
* `--dataset_path`: The path to the benchmark dataset
* `--strategy`: Strategy: `dfs`, `mcts`, `reflexion`
* `--language`: Strategy: `py` or `rs`
* `--model`: OpenAI models only for now, including `gpt-3.5-turbo` and `GPT-4`
* `--pass_at_k`: `Pass@k` metric, default=1
* `--max_iters`: The maximum number of self-improvement iterations, default=8
* `--expansion_factor`: The expansion factor for the reflexion UCS and A* strategy, default=3
* `--is_leetcode`: To run the leetcode benchmark
* `--verbose`: To print live logs


## Trajectories & Evaluation

`root/` contains all the trajectories. Please use `get_acc.py` with the `log path` to get the actual accuracy.

```bash
python get_acc.py
```

<p align="center" width="100%">
<img src="assets/results.jpg" alt="GPT-3.5 and GPT-4 Pass@1 accuracy on HumanEval. Prompting with LATS achieves the highest performance. We sample 5 solutions during expansion for 8 iterations." style="width: 50%; min-width: 100px; display: block; margin: auto;">
</p>


## Demo

The public demo is available at [https://huggingface.co/spaces/AIatUIUC/CodeLATS](https://huggingface.co/spaces/AIatUIUC/CodeLATS). The source code can be reviewed at `demo` folder.

![The demo is an implementation of Language Agent Tree Search (LATS) (https://arxiv.org/abs/2310.04406) built specifically for generating code in the form of python functions.](assets/demo.png)


## Reference
- CoT: Chain-of-thought prompting elicits reasoning in large language models [[Paper](https://arxiv.org/abs/2201.11903)]
- ToT: Tree of thoughts: Deliberate problem solving with large language models [[Paper](http://arxiv.org/abs/2305.10601)] [[Code](https://github.com/princeton-nlp/tree-of-thought-llm)]
- ReAct: Synergizing Reasoning and Acting in Language Models [[Paper](http://arxiv.org/abs/2210.03629)] [[Website](https://react-lm.github.io/)] [[Code](https://github.com/ysymyth/ReAct)]
- RAP: Reasoning with language model is planning with world model [[Paper](http://arxiv.org/abs/2305.14992)] [[Website](https://github.com/Ber666/llm-reasoners)] [[Code](https://github.com/Ber666/RAP)]
- Reflexion: Language Agents with Verbal Reinforcement Learning [[Paper](https://arxiv.org/abs/2303.11366)] [[Code](https://github.com/noahshinn/reflexion)] 
- LATS: Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models [[Paper](https://arxiv.org/abs/2310.04406v2)] [[Code](https://github.com/andyz245/LanguageAgentTreeSearch)]