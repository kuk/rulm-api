
import os
import sys
import json
from collections import deque
from datetime import datetime
from dataclasses import (
    dataclass,
    asdict
)
from contextlib import contextmanager

from aiohttp import web

from llama_cpp import llama_cpp


######
#
#   LLAMA
#
#####


class LlamaError(Exception):
    pass


def str_bytes(str):
    return str.encode('utf8')


def bytes_str(bytes):
    return bytes.decode('utf8')


@contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    stderr_dup_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stderr_fd)

    try:
        yield
    finally:
        os.dup2(stderr_dup_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(stderr_dup_fd)


def llama_ctx(path, n_ctx=256):
    params = llama_cpp.llama_context_default_params()
    params.n_ctx = n_ctx
    params.n_parts = -1
    params.seed = -1
    params.f16_kv = True
    params.logits_all = False
    params.vocab_only = False
    params.use_mmap = True
    params.use_mlock = False
    params.embedding = False

    with suppress_stderr():
        ctx = llama_cpp.llama_init_from_file(
            path_model=str_bytes(path),
            params=params
        )

    if not ctx:
        raise LlamaError(f'failed load {path!r}')
    return ctx


@contextmanager
def llama_ctx_manager(path, n_ctx=256):
    ctx = llama_ctx(path, n_ctx)
    try:
        yield ctx
    finally:
        llama_cpp.llama_free(ctx)


def llama_tokenize(ctx, text, add_bos=True):
    n_ctx = llama_cpp.llama_n_ctx(ctx)
    tokens = (llama_cpp.llama_token * n_ctx)()
    size = llama_cpp.llama_tokenize(
        ctx=ctx,
        text=str_bytes(text),
        tokens=tokens,
        n_max_tokens=n_ctx,
        add_bos=llama_cpp.c_bool(add_bos),
    )
    if size < 0:
        raise LlamaError(f'n_tokens > n_ctx={n_ctx}')
    return n_ctx, tokens[:size]


def llama_token_text(ctx, token):
    return bytes_str(llama_cpp.llama_token_to_str(ctx, token))


def llama_eval(ctx, tokens, n_past, n_threads):
    return_code = llama_cpp.llama_eval(
        ctx=ctx,
        tokens=(llama_cpp.llama_token * len(tokens))(*tokens),
        n_tokens=llama_cpp.c_int(len(tokens)),
        n_past=llama_cpp.c_int(n_past),
        n_threads=llama_cpp.c_int(n_threads),
    )
    if return_code != 0:
        raise LlamaError(f'llama_eval return_code={return_code}')


def llama_sample(ctx, top_k, top_p, temp, repeat_penalty, last_n_tokens):
    return llama_cpp.llama_sample_top_p_top_k(
        ctx=ctx,
        last_n_tokens_data=(llama_cpp.llama_token * len(last_n_tokens))(*last_n_tokens),
        last_n_tokens_size=llama_cpp.c_int(len(last_n_tokens)),
        top_k=llama_cpp.c_int(top_k),
        top_p=llama_cpp.c_float(top_p),
        temp=llama_cpp.c_float(temp),
        repeat_penalty=llama_cpp.c_float(repeat_penalty),
    )
    

@dataclass
class LlamaCompleteRecord:
    n_past: int = None
    n_tokens: int = None
    text: str = None


def llama_complete(
        ctx, prompt,
        n_batch=8, n_threads=8,
        n_predict=16, top_k=40, top_p=0.95, temp=0.8,
        repeat_penalty=1.1, repeat_last_n=64,
):
        n_ctx, tokens = llama_tokenize(ctx, prompt)
        n_tokens = len(tokens)
        if n_tokens + n_predict > n_ctx:
            raise LlamaError(f'n_tokens={n_tokens} + n_predict={n_predict} > n_ctx={n_ctx}')

        last_n_tokens = deque(
            (llama_cpp.llama_token * repeat_last_n)(0),
            maxlen=repeat_last_n
        )
        for n_past in range(0, n_tokens, n_batch):
            batch = tokens[n_past:n_past + n_batch]
            llama_eval(ctx, batch, n_past, n_threads)
            last_n_tokens.extend(batch)
            yield LlamaCompleteRecord(n_past=n_past, n_tokens=n_tokens)

        n_past = n_tokens
        yield LlamaCompleteRecord(n_past=n_past, n_tokens=n_tokens)

        for _ in range(n_predict):
            token = llama_sample(ctx, top_k, top_p, temp, repeat_penalty, last_n_tokens)
            if token == llama_cpp.llama_token_eos():
                break

            llama_eval(ctx, [token], n_past, n_threads)
            n_past += 1
            last_n_tokens.append(token)

            text = llama_token_text(ctx, token)
            yield LlamaCompleteRecord(text=text)


#####
#
#   LOG
#
#####


def log(message, **kwargs):
    data = dict(
        datetime=datetime.utcnow().isoformat(),
        message=message,
        **kwargs
    )
    print(
        json.dumps(data, ensure_ascii=False, default=str),
        flush=True,
        file=sys.stderr
    )


#######
#
#   APP
#
#####


HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 8080))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.expanduser('~/models'))

MODEL_PARAMS = {
    'ru-alpaca-7b-f16': {
        'path': f'{MODELS_DIR}/ru_alpaca_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 256 + 512,
        'n_batch': 16,
        'n_threads': 16,
    },
    'ru-alpaca-7b-q4': {
        'path': f'{MODELS_DIR}/ru_alpaca_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 256 + 512,
        'n_batch': 16,
        'n_threads': 16
    },
    'saiga-7b-f16': {
        'path': f'{MODELS_DIR}/saiga_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'stop_pattern': '<end>'
    },
    'saiga-7b-q4': {
        'path': f'{MODELS_DIR}/saiga_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'stop_pattern': '<end>'
    },
    'saiga-7b-v2-f16': {
        'path': f'{MODELS_DIR}/saiga_v2_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'stop_pattern': '</s>'
    },
    'saiga-7b-v2-q4': {
        'path': f'{MODELS_DIR}/saiga_v2_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'stop_pattern': '</s>'
    },
}

RU_ALPACA_TEMPLATE = '''Задание: {prompt}
Ответ: '''

RU_ALPACA_TEMPLATE2 = '''### Задание: {prompt}
### Ответ: '''

SAIGA_TEMPLATE = '''<start>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. </end>
<start>user
{prompt} </end>
<start>bot
'''

SAIGA_TEMPLATE2 = '''<s>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s>
<s>user
{prompt}</s>
<s>bot
'''

MODEL_TEMPLATES = {
    'ru-alpaca-7b-f16': RU_ALPACA_TEMPLATE,
    'ru-alpaca-7b-q4': RU_ALPACA_TEMPLATE,
    'saiga-7b-f16': SAIGA_TEMPLATE,
    'saiga-7b-q4': SAIGA_TEMPLATE,
    'saiga-7b-v2-f16': SAIGA_TEMPLATE2,
    'saiga-7b-v2-q4': SAIGA_TEMPLATE2,
}


async def models_handler(request):
    data = list(MODEL_PARAMS)
    return web.json_response(data)


async def stream_sent(response, data):
    text = json.dumps(data)
    bytes = str_bytes(text)
    await response.write(bytes + b'\r\n')


def complete_until_match(records, pattern):
    text = ''
    for record in records:
        yield record

        if record.text:
            text += record.text
            if pattern in text:
                break


async def complete_handler(request):
    data = await request.json()
    prompt = data['prompt']
    model = data['model']
    max_tokens = data.get('max_tokens', 16)
    temperature = data.get('temperature', 0.8)
    log('complete', data=data)

    response = web.StreamResponse()
    await response.prepare(request)

    try:
        model_params = MODEL_PARAMS[model]
        template = MODEL_TEMPLATES[model]
        with llama_ctx_manager(
                path=model_params['path'],
                n_ctx=model_params['n_ctx']
        ) as ctx:
            prompt = template.format(prompt=prompt)
            records = llama_complete(
                ctx, prompt,
                n_batch=model_params['n_batch'],
                n_threads=model_params['n_threads'],
                n_predict=max_tokens,
                top_k=40,
                top_p=0.95,
                temp=temperature,
                repeat_penalty=1.1,
                repeat_last_n=64
            )

            stop_pattern = model_params.get('stop_pattern')
            if stop_pattern:
                records = complete_until_match(records, stop_pattern)

            for record in records:
                await stream_sent(response, asdict(record))
                
    except ConnectionResetError:
        pass

    return response
    

def main():
    app = web.Application()
    app.add_routes([
        web.get('/v1/models', models_handler),
        web.post('/v1/complete', complete_handler)
    ])
    log('run app', host=HOST, port=PORT)
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == '__main__':
    main()
