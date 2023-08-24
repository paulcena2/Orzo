# Script to run the Orzo Client

import logging
import orzo


logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG
)


def main():
    orzo.connect("ws://localhost:50000")


if __name__ == "__main__":
    main()
