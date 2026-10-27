from __future__ import annotations

import argparse
import signal

from .engine import Engine
from .store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    engine = Engine(Store(args.state_dir), args.run_id)
    signal.signal(signal.SIGTERM, lambda signum, frame: engine.stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: engine.stop.set())
    engine.execute()


if __name__ == "__main__":
    main()
