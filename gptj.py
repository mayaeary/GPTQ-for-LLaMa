import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *


def get_gptj(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTJForCausalLM
    model = GPTJForCausalLM.from_pretrained(model, torch_dtype='auto', low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def gptj_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.wte = model.transformer.wte.to(dev)
    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, **kwargs):
            inps[cache['i']] = hidden_states
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers = model.transformer.h
    layers[0] = layers[0].cpu()
    model.transformer.wte = model.transformer.wte.cpu()
    model.transformer.ln_f = model.transformer.ln_f.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False
            )
        
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
            quantizers['transformer.h.%d.%s' % (i, name)] = (gptq[name].quantizer,scale,zero,g_idx)
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


def gptj_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name],scale,zero,g_idx = quantizers[name]
        quantizers[name],scale,zero,g_idx = quantizers[name].cpu(),scale.cpu(),zero.cpu(),g_idx.cpu()
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model

def load_quant(model, checkpoint, wbits, groupsize=-1):
    from transformers import GPTJConfig, GPTJForCausalLM
    config = GPTJConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = GPTJForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done!')

    return model

def get_tokenizer(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='GPT-J model to load.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save_safetensors', type=str, default='',
        help='Save quantized `.safetensors` checkpoint under this name.'
    )

    args = parser.parse_args()
    
    model = get_gptj(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    tick = time.time()
    quantizers = gptj_sequential(model, dataloader, DEV)
    print(time.time() - tick)

    if args.save:
        gptj_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save) 

    if args.save_safetensors:
        gptj_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        safe_save(model.state_dict(), args.save_safetensors)
