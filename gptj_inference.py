import time

import torch
import torch.nn as nn
import gptj

from gptq import *
from modelutils import *
from quant import *

from transformers import AutoTokenizer

DEV = torch.device('cuda:0')

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--text', type=str,
        help='input text'
    )
    
    parser.add_argument(
        '--min_length', type=int, default=10,
        help='The minimum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=50,
        help='The maximum length of the sequence to be generated.'
    )
    
    parser.add_argument(
        '--top_p', type=float , default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )
    
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )
    
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        model = gptj.load_quant(args.model, args.load, args.wbits)
    else:
        model = gptj.get_gptj(args.model)
        model.eval()
        
    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    print(tokenizer.decode([el.item() for el in generated_ids[0]]))
