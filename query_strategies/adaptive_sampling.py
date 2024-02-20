import torch
import random
from .strategy import Strategy
from config import parse_args
args = parse_args()

class AdaptiveSampling(Strategy):
    def __init__(self, dataset, net):
        super(AdaptiveSampling, self).__init__(dataset, net)
        self.total_rounds = args.n_round

    def query(self, n, current_round):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs * log_probs).sum(1)  # Calculate the entropy

        sorted_indices = uncertainties.sort()[1]

        # Define percentage ranges for easy, medium, and hard samples
        easy_range = (0.7, 0.95)   # Select from the easiest 30%
        medium_range = (0.3, 0.7) # Select from the middle 40%
        hard_range = (0.05, 0.3)   # Select from the hardest 30%

        if current_round < self.total_rounds / 3:
            # Early rounds: select from the easier part
            start, end = int(easy_range[0] * len(sorted_indices)), int(easy_range[1] * len(sorted_indices))

        elif current_round < 2 * self.total_rounds / 3:
            # Middle rounds: select from the medium part
            start, end = int(medium_range[0] * len(sorted_indices)), int(medium_range[1] * len(sorted_indices))

        else:
            # Later rounds: select from the harder part
            start, end = int(hard_range[0] * len(sorted_indices)), int(hard_range[1] * len(sorted_indices))

        # Randomly select samples from the specified range to avoid redundancy
        selected = random.sample(list(sorted_indices[start:end]), n)

        return unlabeled_idxs[selected]