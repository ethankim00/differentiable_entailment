import torch
from transformers import GPT2Tokenizer

from utils import get_verbalization_ids


class PromptEncoder(object):
    """
    Object
    """

    def __init__(self, tokenizer, pvp, label_list):
        """[summary]

        Args:
            tokenizer ([type]): Huggingface TOkenizer
            pvp ([type]): Pattern Verbalizer Pair
            label_list ([type]): [description]
        """
        # Record prompt tokens
        pattern_token_set, pattern_token_indices = set(), []
        # RoBERTa tokenizer is initiated from GPT2Tokenizer,
        # and it tokenizes same words differently in different positions:
        # e.g.  'Hello world!' -> ['Hello', 'Ġworld', '!'];
        #       'Hello', 'world' -> ['Hello'], ['world']
        # So we need to add prefix space to simulate true situations
        kwargs = (
            {"add_prefix_space": True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        )
        for idx, part in enumerate(
            pvp.PATTERN
        ):  # Iterate over pattern - looks like only one pseodutoken is used per nonzero block flag
            if (
                pvp.BLOCK_FLAG[idx] == 1
            ):  # if block flag is equal to 1 (corresponding to "the" in the pattern) -> we have a trainable pseudotoken
                token_ids = tokenizer.encode(
                    part, add_special_tokens=False, **kwargs
                )  # get token ids # encode "the" with the tokenizer
                pattern_token_set.update(token_ids)  # add to dictionary
                pattern_token_indices.extend(token_ids)

        # Record label tokens - all verbalizer ids for all label
        label_token_ids = []  # start ids of verbalizer tokens
        for label_idx, label in enumerate(label_list):  # iterate over list of labels
            verbalizers = pvp.verbalize(
                label
            )  # verbalizers (more than 1? Yes, but only one used in most PVP templates)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(
                    verbalizer, tokenizer, force_single_token=True
                )
                assert (
                    verbalizer_id != tokenizer.unk_token_id
                ), "verbalization was tokenized as <UNK>"
                label_token_ids.append(verbalizer_id)
        self.entailment_label_id = tokenizer.encode(
            "label", add_special_tokens=False, **kwargs
        )
        assert (
            len(pattern_token_set) < 50 and len(label_token_ids) < 49
        )  # Can't have more than 50 classes or 50 trainable pseudotokens

        # Convert tokens in manual prompt / label to unused tokens
        # Note that `AlbertTokenizer` or `RobertaTokenizer` doesn't have a `vocab` attribute
        if hasattr(tokenizer, "vocab") and "[unused0]" in tokenizer.vocab:
            # BERT
            self.pattern_convert = {
                token_id: tokenizer.vocab["[unused%s]" % idx]
                for idx, token_id in enumerate(pattern_token_set)
            }
            self.label_convert = {
                token_id: tokenizer.vocab["[unused%s]" % (idx + 50)]
                for idx, token_id in enumerate(label_token_ids)
            }

        else:
            # ALBERT, RoBERTa
            start_idx = tokenizer.vocab_size - 100
            self.pattern_convert = {
                token_id: start_idx + idx
                for idx, token_id in enumerate(pattern_token_set)
            }
            self.label_convert = {
                token_id: start_idx + 50 + idx
                for idx, token_id in enumerate(label_token_ids)
            }

        # Convert mlm logits to cls logits
        self.vocab_size = tokenizer.vocab_size
        self.m2c_tensor = torch.tensor(
            list(self.label_convert.values()), dtype=torch.long
        )

        # Use lookup tensor to get replace embeddings
        # pattern_token_indices = list of psuedotoken token ids
        # self.pattern_convert dicitonary mapping our new tokens to index within roberta tokenizer vocabulary
        self.lookup_tensor = torch.tensor(
            [self.pattern_convert[origin] for origin in pattern_token_indices],
            dtype=torch.long,
        )
        # self.lookup_tensor - tensor of psuedotoken indices within vocab

    def init_embed(self, model, random_=False, prompt_token_ids = None):
        """
        Initialize embeddings for a model

        Adds embeddings for the new pattern and verbalizer tokens

        Args:
            model ([type]): [description]
            random_ (bool, optional): [description]. Defaults to False.
        
        Initialize trainable pseudotokens randomly while initializing prompt and label tokens from their default embedding value

        """
        w = model.get_input_embeddings().weight.data
        print(self.pattern_convert)
        print(prompt_token_ids)
        if not prompt_token_ids:
            for origin_id, convert_id in self.pattern_convert.items():
                if random_:
                    max_val = w[convert_id].abs().max()
                    w[convert_id].uniform_(-max_val, max_val)
                else:
                    w[convert_id] = w[origin_id]
            for origin_id, convert_id in self.label_convert.items():
                if random_:
                    max_val = w[convert_id].abs().max()
                    w[convert_id].uniform_(-max_val, max_val)
                else:
                    w[convert_id] = w[origin_id]
        else:
            for origin_id, convert_id in self.pattern_convert.items():
                if not origin_id in prompt_token_ids:
                    max_val = w[convert_id].abs().max()
                    w[convert_id].uniform_(-max_val, max_val)
                else:
                    print("Initializing from pretrained embedding for {}".format(origin_id))
                    w[convert_id] = w[origin_id]
            for origin_id, convert_id in self.label_convert.items():
                if not origin_id in prompt_token_ids:
                    max_val = w[convert_id].abs().max()
                    w[convert_id].uniform_(-max_val, max_val)
                else:
                    w[convert_id] = w[origin_id]

    def add_embed_hook(self, model):
        def stop_gradient(_, grad_input, __):  # Freeze some paramters
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        trainable_ids = list(self.pattern_convert.values()) + list(
            self.label_convert.values()
        )
        print(trainable_ids)
        grad_mask = torch.zeros((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 1.0

        # Stop gradient for non trainable input embeddigns (regular words that are not psuedotokens)
        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def add_reverse_hook(self, model):
        def stop_gradient(_, grad_input, __):
            # grad_input: tuple containing a (vocab_size, hidden_dim) tensor
            return (grad_mask.to(grad_input[0].device) * grad_input[0],)

        # Train certain tokens by multiply gradients with a mask
        # reverse of above?
        trainable_ids = list(self.pattern_convert.values()) + list(
            self.label_convert.values()
        )
        print(trainable_ids)
        grad_mask = torch.ones((self.vocab_size, 1), dtype=torch.float)
        grad_mask[trainable_ids, 0] = 0.0

        return model.get_input_embeddings().register_backward_hook(stop_gradient)

    def get_replace_embeds(self, word_embeddings):
        return word_embeddings(self.lookup_tensor.to(word_embeddings.weight.device))

    def convert_mlm_logits_to_cls_logits(self, mlm_labels, logits):
        """Convert logits from masked language model to

        Args:
            mlm_labels ([type]): [description]
            logits ([type]): [description]

        Returns:
            [type]: [description]
        """
        return torch.index_select(
            logits[mlm_labels != -1], -1, self.m2c_tensor.to(logits.device)
        )
        # index into masked token logit, take last entry
