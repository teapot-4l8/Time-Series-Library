#!/usr/bin/env python3
"""
Simple resampling/prep script for custom CSV datasets.

Usage examples:
  python scripts/prepare_custom_dataset.py \
    --input datasets/adjusted_data.csv \
    --output datasets/adjusted_data_1s.csv \
    --freq 1s --date_col date

The script parses the date column, sorts by time, resamples to the requested
frequency (e.g. '1s', '10s', '1min'), and forward-fills missing values.
"""
import argparse
import pandas as pd
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='input CSV path')
    p.add_argument('--output', required=True, help='output CSV path')
    p.add_argument('--freq', default='1s', help="pandas offset alias, e.g. '1s','10s','1min'")
    p.add_argument('--date_col', default='date', help='name of the datetime column')
    p.add_argument('--target', default=None, help='name of target column (optional)')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f'Failed to read {args.input}: {e}', file=sys.stderr)
        sys.exit(1)

    if args.date_col not in df.columns:
        print(f"Date column '{args.date_col}' not found in input file. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # parse dates and set index
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.set_index(args.date_col).sort_index()

    # resample and forward-fill missing values
    try:
        df_resampled = df.resample(args.freq).ffill()
    except Exception as e:
        print(f'Failed to resample with freq={args.freq}: {e}', file=sys.stderr)
        sys.exit(1)

    # reset index to have date column again
    df_resampled = df_resampled.reset_index()

    # quick sanity checks
    if args.target and args.target not in df_resampled.columns:
        print(f"Warning: target '{args.target}' not present after resampling. Columns: {list(df_resampled.columns)}")

    df_resampled.to_csv(args.output, index=False)
    print(f'Wrote {args.output} (rows: {len(df_resampled)}, columns: {len(df_resampled.columns)})')


if __name__ == '__main__':
    main()
