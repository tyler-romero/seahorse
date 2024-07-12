from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset as HFDataset
from pydantic import BaseModel, Field

from seahorse.data.data_prep.llava_pretrain_cc3m import make_llava_pretrain_cc3m
from seahorse.data.data_prep.the_cauldron import make_the_cauldron
from seahorse.models.seahorse import SeahorseModel

DATASET_REGISTRY = {
    "the_cauldron": make_the_cauldron,
    "llava_pretrain_cc3m": make_llava_pretrain_cc3m,
}


class DatasetSpec(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)

    def construct(self, model: SeahorseModel) -> HFDataset:
        make_fn = DATASET_REGISTRY[self.name]
        return make_fn(model, **self.kwargs)


class DataConfig(BaseModel):
    dataset_specs: list[DatasetSpec] = Field(default=..., min_length=1)
    mandatory_columns: set[str] = {"text", "image", "length"}


def construct_dataset(data_config: DataConfig, seahorse: SeahorseModel) -> HFDataset:
    """
    Construct a dataset from a list of data loading functions. The functions should return a HFDataset
    and take only a seahorse model as an argument. The resulting datasets will be concatenated.
    It is up to the user to shuffle the dataset if needed.
    """
    ds = None
    sub_datasets = []
    for i, dataset_spec in enumerate(data_config.dataset_specs):
        ds = dataset_spec.construct(seahorse)
        if not isinstance(ds, HFDataset):
            raise ValueError(f"Expected a HFDataset, got {type(ds)} from dataset {i}")
        if not data_config.mandatory_columns.issubset(ds.column_names):
            raise ValueError(
                f"Expected columns {data_config.mandatory_columns}, got {ds.column_names} from dataset {i}"
            )
        sub_datasets.append(ds)

    return concatenate_datasets(sub_datasets)
