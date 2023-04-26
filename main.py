
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
    with suppress_stderr():
        size = llama_cpp.llama_tokenize(
            ctx=ctx,
            text=str_bytes(text),
            tokens=tokens,
            n_max_tokens=n_ctx,
            add_bos=llama_cpp.c_bool(add_bos),
        )
    if size < 0:
        raise LlamaError(f'n_tokens={-size} > n_ctx={n_ctx}')
    return tokens[:size]


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
        ctx, tokens,
        n_batch=8, n_threads=8,
        n_predict=16, top_k=40, top_p=0.95, temp=0.8,
        repeat_penalty=1.1, repeat_last_n=64,
):
        n_tokens = len(tokens)
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


def match_stop(records, stop, buffer_size=4):
    buffer = deque()
    text = ''

    for record in records:
        if not record.text:
            yield record
            continue

        buffer.append(record)
        text += record.text

        index = text.find(stop)
        if index >= 0:
            text = text[:index]
            if text:
                yield LlamaCompleteRecord(text=text)
            return

        if len(buffer) >= buffer_size:
            record = buffer.popleft()
            text = text[len(record.text):]
            yield record

    yield from buffer


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
SAIGA_STOP = ' <end>'

SAIGA_TEMPLATE2 = '''<s>system
Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s>
<s>user
{prompt}</s>
<s>bot
'''
SAIGA_STOP2 = '</s>'

MODEL_PARAMS = {
    'ru-alpaca-7b-f16': {
        'path': f'{MODELS_DIR}/ru_alpaca_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 256 + 512,
        'n_batch': 16,
        'n_threads': 16,
        'template': RU_ALPACA_TEMPLATE,
    },
    'ru-alpaca-7b-q4': {
        'path': f'{MODELS_DIR}/ru_alpaca_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 256 + 512,
        'n_batch': 16,
        'n_threads': 16,
        'template': RU_ALPACA_TEMPLATE,
    },
    'saiga-7b-f16': {
        'path': f'{MODELS_DIR}/saiga_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'template': SAIGA_TEMPLATE,
        'stop': SAIGA_STOP,
    },
    'saiga-7b-q4': {
        'path': f'{MODELS_DIR}/saiga_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'template': SAIGA_TEMPLATE,
        'stop': SAIGA_STOP,
    },
    'saiga-7b-v2-f16': {
        'path': f'{MODELS_DIR}/saiga_v2_llamacpp/7B/ggml-model-f16.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'template': SAIGA_TEMPLATE2,
        'stop': SAIGA_STOP2,
    },
    'saiga-7b-v2-q4': {
        'path': f'{MODELS_DIR}/saiga_v2_llamacpp/7B/ggml-model-q4.bin',
        'n_ctx': 2000,
        'n_batch': 16,
        'n_threads': 16,
        'template': SAIGA_TEMPLATE2,
        'stop': SAIGA_STOP2,
    },
}


def check_model_params(data):
    for model, params in data.items():
        assert params['n_ctx'] <= 2048
        assert os.path.exists(params['path'])


async def models_handler(request):
    data = list(MODEL_PARAMS)
    return web.json_response(data)


def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return


async def tokenize_handler(request):
    text = await request.text()
    data = safe_json_loads(text)
    if type(data) is not dict:
        raise web.HTTPBadRequest(text=f'{text!r}, json dict required')

    text = data.get('text')
    if type(text) is not str:
        raise web.HTTPBadRequest(text=f'text={text!r}, str is required')

    model = data.get('model')
    if type(model) is not str:
        raise web.HTTPBadRequest(text=f'model={model!r}, str is required')

    model_params = MODEL_PARAMS.get(model)
    if not model_params:
        raise web.HTTPBadRequest(text=f'unknown model {model!r}')

    try:
        with llama_ctx_manager(model_params['path'], model_params['n_ctx']) as ctx:
            tokens = llama_tokenize(ctx, text, add_bos=False)
            return web.json_response([
                llama_token_text(ctx, _) for _ in tokens
            ])
    except LlamaError as error:
        raise web.HTTPInternalServerError(text=str(error))


async def stream_sent(response, data):
    text = json.dumps(data)
    bytes = str_bytes(text)
    await response.write(bytes + b'\r\n')


async def complete_handler(request):
    text = await request.text()
    data = safe_json_loads(text)
    if type(data) is not dict:
        raise web.HTTPBadRequest(text=f'{text!r}, json dict required')

    prompt = data.get('prompt')
    if type(prompt) is not str:
        raise web.HTTPBadRequest(text=f'prompt={prompt!r}, str required')

    model = data.get('model')
    if type(model) is not str:
        raise web.HTTPBadRequest(text=f'model={model!r}, str required')

    model_params = MODEL_PARAMS.get(model)
    if model_params is None:
        raise web.HTTPBadRequest(text=f'unknown model {model!r}')

    n_ctx = model_params['n_ctx']
    max_tokens = data.get('max_tokens', 32)
    if type(max_tokens) is not int or not 1 <= max_tokens <= n_ctx:
        raise web.HTTPBadRequest(text=f'max_tokens={max_tokens!r}, int in [1, {n_ctx}] required')

    temperature = data.get('temperature', 0.2)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 1:
        raise web.HTTPBadRequest(text=f'temperature={temperature!r}, float in [0, 1] required')

    log(
        'complete start',
        prompt=prompt, model=model,
        max_tokens=max_tokens, temperature=temperature
    )

    response = web.StreamResponse()
    await response.prepare(request)

    with llama_ctx_manager(model_params['path'], n_ctx) as ctx:
        try:
            prompt = model_params['template'].format(prompt=prompt)
            tokens = llama_tokenize(ctx, prompt, add_bos=True)
            records = llama_complete(
                ctx, tokens,
                n_batch=model_params['n_batch'],
                n_threads=model_params['n_threads'],
                n_predict=max_tokens,
                top_k=40,
                top_p=0.95,
                temp=temperature,
                repeat_penalty=1.1,
                repeat_last_n=64
            )
            
            stop = model_params.get('stop')
            if stop:
                records = match_stop(records, stop)

            for record in records:
                if record.text:
                    item = {'text': record.text}
                else:
                    item = {'prompt_progress': record.n_past / record.n_tokens}
                await stream_sent(response, item)

        except LlamaError as error:
            await stream_sent(response, {'error': str(error)})

        except ConnectionResetError:
            pass

    return response


def main():
    check_model_params(MODEL_PARAMS)

    app = web.Application()
    app.add_routes([
        web.get('/v1/models', models_handler),
        web.post('/v1/tokenize', tokenize_handler),
        web.post('/v1/complete', complete_handler)
    ])
    log('run app', host=HOST, port=PORT)
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == '__main__':
    main()
