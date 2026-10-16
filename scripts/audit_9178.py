#!/usr/bin/env python3
"""Audit 9178 raw packet headers without inferring UTC or sampling-clock alignment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))
from packet9178 import audit_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_group(args.files), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
