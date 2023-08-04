import torch
from torch.utils.data import IterableDataset
import webdataset as wds
import glob
from transformers import AutoTokenizer, AutoProcessor
from itertools import islice
import transformers
from dataclasses import dataclass, field
from itertools import islice
import braceexpand


@dataclass
class DataCollatorForVisionLanguageDataset(object):
    """Collate examples for vision-language fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.CLIPImageProcessor

    def __call__(self, batch):
        image_inputs = self.processor(images=batch[0], return_tensors="pt")
        text_inputs = self.tokenizer(batch[1], return_tensors="pt", padding=True, truncation=True)
        return {**image_inputs, **text_inputs}

class SlicedWebLoader(wds.WebLoader):
    def __init__(self, *args, num_workers, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return super().__iter__()
        else:
            worker_id = worker_info.id
            return islice(super().__iter__(), worker_id, None, self.num_workers)


dataset_files = list(braceexpand.braceexpand("/data/private/cc3m/cc3m_train/{00000..00030}.tar"))

dataset = (
    wds.WebDataset(dataset_files)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "txt")
    # .batched(256, partial=False)
    # .with_epoch(12808 // 256)
)

# tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir="/data/private/cc3m/cache")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="/data/private/cc3m/cache").image_processor

# # apply image processor and tokenizer
# collate_fn = DataCollatorForVisionLanguageDataset(tokenizer, processor)

# dataloader = wds.WebLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=None)
# dataloader.length = 12808 // 256

it = iter(dataset)

i = 0

while True:
    try:
        next(it)
        i += 1
        print(i)
    except StopIteration:
        break

print(i)

# for a in dataloader:
#     i += 1
#     # print(image.shape, caption.shape)
#     # print(image.dtype, caption.dtype)
#     # print(image.device, caption.device)
#     # print(image.keys())
#     # print(caption.keys())
#     print(a["input_ids"].shape, i)