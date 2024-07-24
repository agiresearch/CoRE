import os

def get_vehicle_result_file(args):
    result_file = os.path.join(args.results_dir, args.task, f"{args.model_name.replace('/','_')}_flow_results.jsonl")
    if not os.path.exists(os.path.join(args.results_dir, args.task)):
        os.makedirs(os.path.join(args.results_dir, args.task))
    return result_file
