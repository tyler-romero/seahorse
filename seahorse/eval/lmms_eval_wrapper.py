from typing import List, Tuple

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

from seahorse.models.seahorse import DEFAULT_IMAGE_TOKEN, SeahorseModel


@register_model("seahorse")
class SeahorseLmms(lmms):
    def __init__(
        self,
        model: SeahorseModel,
        batch_size: int = 1,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self._model = model
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._model.config

    @property
    def tokenizer(self):
        return self._model.tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self.model.device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        assert self._world_size == 1, "SeahorseEvalModel does not support distributed evaluation."
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Not implemented for SeahorseEvalModel.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Borrowed from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/phi3v.py
        """
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # We assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]
            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if DEFAULT_IMAGE_TOKEN in contexts[i]:
                    query = "" + contexts[i]
                else:
                    query = ""
                    for _ in range(len(visuals)):
                        query += DEFAULT_IMAGE_TOKEN + "\n"
                    query += contexts[i]
                messages = [{"role": "user", "content": query}]
                contexts[i] = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            assert len(contexts) == 1
            #
            context = contexts[0]
            tokens = self.model.tokenize_text(context)
            pixel_values = self.model.preprocess_image(visuals)

            # Setting default parameters.
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # Generate answer.
            generate_ids = self.model.generate(
                input_ids=tokens.input_ids.to(self.model.device),
                pixel_values=pixel_values.to(self.model.device),
                attention_mask=tokens.attention_mask.to(self.model.device),
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=[  # # phi3 specific generation eos tokens
                    self.tokenizer.convert_tokens_to_ids("<|assistant|>"),
                    self.tokenizer.convert_tokens_to_ids("<|end|>"),
                    self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                ],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generate_ids = generate_ids[:, tokens.input_ids.shape[1] :]

            response = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()
        return res
