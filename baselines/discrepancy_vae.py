import sys
from baselines._crl import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--method", "dvae"]
    main()
