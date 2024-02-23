import os
import argparse
from reflexion import run_reflexion
from mcts import run_mcts
from dfs import run_dfs
from utils import read_jsonl, read_jsonl_gz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str, help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str, help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str, help="Strategy: `dfs`, `mcts`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument("--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int, help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int, help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int, help="The expansion factor for the reflexion UCS and A* strategy", default=3)
    parser.add_argument("--is_leetcode", action='store_true', help="To run the leetcode benchmark")  # Temporary
    parser.add_argument("--verbose", action='store_true', help="To print live logs")
    # TODO: implement this
    # parser.add_argument("--is_resume", action='store_true', help="To resume run")
    # parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    args = parser.parse_args()
    return args

def stragey_factory(strategy: str, **kwargs):
    delete_keys = ["expansion_factor"]
    for key in delete_keys:
        if key in kwargs:
            del kwargs[key] # delete the key from the dictionary
    if strategy == "reflexion":
        return run_reflexion(**kwargs)
    elif strategy == "mcts":
        return run_mcts(**kwargs)
    elif strategy == "dfs":
        return run_dfs(**kwargs)
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


if __name__ == "__main__":
    args = get_args()
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir): # trajectory/
        os.makedirs(args.root_dir)
    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "") # humaneval-py-test.jsonl -> humaneval-py-test
    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name) # test_reflexion_gpt3
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl")

    # print starting message
    if args.verbose:
        print(f"""Starting run with the following parameters:
                  strategy: {args.strategy}
                  pass@k: {args.pass_at_k}""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(f"Dataset path `{args.dataset_path}` is not supported")
    print(f"Loaded {len(dataset)} examples") # [{}, ...] a list of json objects
    
    # start the run and evaluate with pass@k
    stragey_factory(args.strategy, # reflexion or mcts or dfs
        dataset=dataset, # [{}, ...]
        model_name=args.model, # gpt-3.5-turbo
        language=args.language, # py, rs
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path, # .jsonl file to present the results
        verbose=args.verbose, # strarting message
        expansion_factor=args.expansion_factor,
        is_leetcode=args.is_leetcode
    )

    print(f"Done! Check out the logs in `{log_path}`")