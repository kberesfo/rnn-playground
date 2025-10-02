
import sys
from .utils import FileLoader, build_parser

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
