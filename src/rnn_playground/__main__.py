from pathlib import Path
import argparse
import sys
from rnn_playground.utils import FileLoader


def _existing_file(p: str) -> Path:
    path = Path(p)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {p}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rnn-playground",
        description="Char-level LSTM lab: train and sample from a tiny corpus."
    )
    # pass your dataset file
    parser.add_argument(
        "-d", "--data",
        type=_existing_file,
        default=Path("data/tiny.txt"),
        help="Path to the training text file (default: data/tiny.txt)."
    )
    # a couple of common knobs you’ll likely want soon
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (T).")
    parser.add_argument("--batch-size", type=int,
                        default=64, help="Batch size (B).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--prompt", type=str, default="To be",
                        help="Prompt for sampling.")
    parser.add_argument("--sample-len", type=int,
                        default=200, help="Generated length.")
    parser.add_argument("--temperature", type=float,
                        default=0.8, help="Sampling temperature.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # TODO: wire these into your dataset/model/train code
    fileloader = FileLoader(args.data)
    fileloader.create_vocab()
    print(f"Data file: {args.data}")
    print(
        f"seq_len={args.seq_len} batch_size={args.batch_size} epochs={args.epochs}")
    print(
        f"prompt='{args.prompt}' sample_len={args.sample_len} temperature={args.temperature}")


if __name__ == "__main__":
    sys.exit(main())
