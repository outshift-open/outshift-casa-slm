"""Load and print the config file."""

import rich

from casa_slm.prompt_optimizer.config import load_config


def main() -> None:
    """Entry point."""
    config = load_config()
    rich.print(config)


if __name__ == "__main__":
    main()
