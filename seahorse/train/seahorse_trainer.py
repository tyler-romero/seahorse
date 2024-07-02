from data.sampling import LengthGroupedSampler
from torch.utils.data import Sampler
from transformers.trainer import Trainer, has_length


class SeahorseTrainer(Trainer):
    def _get_train_sampler(self) -> Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
            )
        else:
            return super()._get_train_sampler()
