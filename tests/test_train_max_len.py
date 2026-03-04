"""
Test: max_len in the training collator (ExtractorCollator).

Shows that max_len truncates input_ids during both training and eval
batch preparation. Uses a freshly initialised Extractor (no pretrained
weights needed — we only test the collator, not the model forward pass).

Run:
    python tests/test_train_max_len.py
"""

from gliner2.model import Extractor, ExtractorConfig
from gliner2.training.trainer import ExtractorCollator, ExtractorDataset
from gliner2.training.data import InputExample


SHORT_TEXT = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
LONG_TEXT  = SHORT_TEXT * 50   # ~700 tokens

LINE_WIDTH = 64  # output box width in characters


def header(title):
    print(f"\n┌{'─' * (LINE_WIDTH - 2)}┐")
    print(f"│  {title:<{LINE_WIDTH - 4}}│")
    print(f"└{'─' * (LINE_WIDTH - 2)}┘")

def row(key, value):
    print(f"  {key:<16}{value}")


def main():
    examples = [
        InputExample(
            text=LONG_TEXT,
            entities={"company": ["Apple"], "person": ["Tim Cook"]},
        )
    ]
    dataset   = ExtractorDataset(examples, validate=False)
    batch_raw = [dataset[0]]

    for max_len in (None, 384):
        config = ExtractorConfig(
            model_name="bert-base-uncased",
            max_width=8,
            counting_layer="count_lstm",
            token_pooling="first",
            max_len=max_len,
        )
        print(f"Initialising Extractor (max_len={max_len}) …")
        model = Extractor(config)

        for is_training in (True, False):
            collator = ExtractorCollator(model.processor,
                                         is_training=is_training,
                                         max_len=model.config.max_len)
            batch   = collator(batch_raw)
            seq_len = batch.input_ids.shape[1]

            label = (
                f"{'train' if is_training else 'eval '}"
                f"  |  max_len={'none' if max_len is None else max_len}"
            )
            header(label)
            row("input_ids len", f"{seq_len} tokens")
            row("text tokens",   f"{len(batch.text_tokens[0])} tokens")

    print()


if __name__ == "__main__":
    main()
