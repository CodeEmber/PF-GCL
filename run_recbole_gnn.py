import argparse

from recbole_gnn.quick_start import run_recbole_gnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    args, _ = parser.parse_known_args()
    args.config_file_list = [
        "properties/config.yaml",
    ]
    if args.config_files:
        args.config_file_list.extend(args.config_files.split(","))
    if args.dataset in ["yelp", "amazon-kindle-store", "amazon-books"]:
        args.config_file_list.append(f"properties/{args.dataset}.yaml")
    run_recbole_gnn(
        model=args.model, dataset=args.dataset, config_file_list=args.config_file_list
    )
