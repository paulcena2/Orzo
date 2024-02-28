# Script to run the Orzo Client

import logging
import sys
sys.path.append('/Users/jbachman/Downloads/workspace/Orzo-main')
sys.path.append('/Users/jbachman/Downloads/workspace/Rigatoni-main')
import orzo


logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG
)

def main():
    orzo.connect("ws://localhost:50000")
    
if __name__ == "__main__":
    main()
