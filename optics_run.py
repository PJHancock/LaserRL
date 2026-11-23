"""Export a random-policy puzzle playback without starting PPO training."""
from optics_experiment import main

if __name__ == "__main__":
    import sys
    main(["render", *sys.argv[1:]])
