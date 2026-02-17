import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Validate interface metric correlations.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset CSV with interface metrics",
    )
    parser.add_argument("--threshold", type=float, default=0.995)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    pairs = [
        ("ysz_tgo_max_sigma_yy", "tgo_bc_max_sigma_yy"),
        ("ysz_tgo_mean_sed", "tgo_bc_mean_sed"),
    ]

    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            print(f"Missing columns for correlation: {a}, {b}")
            continue
        corr = float(df[[a, b]].corr().iloc[0, 1])
        if corr > args.threshold:
            print(f"WARNING: corr({a}, {b}) = {corr:.5f} > {args.threshold}")
        else:
            print(f"OK: corr({a}, {b}) = {corr:.5f}")


if __name__ == "__main__":
    main()
