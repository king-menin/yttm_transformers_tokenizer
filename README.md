# YTTMTransformersTokenizer
An implementation of [youtokentome](https://github.com/VKCOM/YouTokenToMe) tokenizer for [transformers](https://github.com/huggingface/transformers)
### Requirements
Tested on transformers==2.8.0
### Examples
```
>>> from tokenization_yttm import YTTMTransformersTokenizer
>>> model_path = "encoder.model"
>>> tokenizer = YTTMTransformersTokenizer.from_pretrained(vocab_file=model_path)
>>> print(tokenizer.decode(tokenizer.encode("Hello", add_special_tokens=False)))
Hello
>>> tokenizer.encode_plus("привет, чо как?")
{'input_ids': [2, 4620, 21, 15050, 9859, 3],
 'token_type_ids': [0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1]}
```