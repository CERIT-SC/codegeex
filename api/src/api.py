from typing import List
from fastapi import FastAPI, Request
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import uvicorn, json
import argparse
from utils import (
    is_code_generation_finished,
    cleanup_code,
)

try:
    import chatglm_cpp
    import chatglm_cpp._C as _C

    enable_chatglm_cpp = True
except:
    print(
        "[WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance. "
        "Check out https://github.com/li-plus/chatglm.cpp for more details."
    )
    enable_chatglm_cpp = False


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX2 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex2-6b",
    )
    group.add_argument(
        "--dataset-type",
        type=str,
        default="humanevalx",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    group.add_argument(
        "--cpu",
        action="store_true",
    )
    group.add_argument(
        "--half",
        action="store_true",
    )
    group.add_argument(
        "--quantize",
        type=int,
        default=None,
    )
    group.add_argument(
        "--chatglm-cpp",
        action="store_true",
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    return parser


def sync_generate(self, input_ids: List[int], gen_config: _C.GenerationConfig) -> str:
    input_ids = [x for x in input_ids]  # make a copy
    n_past = 0
    n_ctx = len(input_ids)
    code = ""

    while len(
        input_ids
    ) < n_ctx + gen_config.max_length and not is_code_generation_finished(
        code, args.dataset_type, "python"
    ):
        next_token_id = self.model.generate_next_token(
            input_ids, gen_config, n_past, n_ctx
        )
        n_past = len(input_ids)
        input_ids.append(next_token_id)
        if next_token_id == self.model.config.eos_token_id:
            break

        code = self.tokenizer.decode(input_ids[n_ctx:])

    return code


chatglm_cpp.Pipeline._sync_generate = sync_generate

app = FastAPI()


def device():
    if enable_chatglm_cpp and args.chatglm_cpp:
        print("Using chatglm-cpp to improve performance")
        dtype = "f16" if args.half else "f32"
        if args.quantize in [4, 5, 8]:
            dtype = f"q{args.quantize}_0"
        model = chatglm_cpp.Pipeline(args.model_path, dtype=dtype)
        return model

    print("chatglm-cpp not enabled, falling back to transformers")

    if not args.cpu:
        if not args.half:
            model = AutoModel.from_pretrained(
                args.model_path, trust_remote_code=True
            ).cuda()
        else:
            model = (
                AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
                .cuda()
                .half()
            )
        if args.quantize in [4, 8]:
            print(f"Model is quantized to INT{args.quantize} format.")
            model = model.half().quantize(args.quantize)
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)

    return model.eval()


@app.post("/multilingual_code_generate_adapt")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    lang = json_post_list.get("lang")
    prompt = json_post_list.get("prompt")
    max_length = json_post_list.get("max_length", 600)
    top_p = json_post_list.get("top_p", 0.95)
    temperature = json_post_list.get("temperature", 0.2)
    top_k = json_post_list.get("top_k", 0)
    n = json_post_list.get("n", 1)

    results = []

    for i in range(n):
        res_temp = temperature + ((1 - temperature) / n) * i
        response = model.generate(
            prompt,
            max_length=len(prompt) + max_length,
            do_sample=res_temp > 0,
            top_p=top_p,
            top_k=top_k,
            temperature=res_temp,
        )

        response = cleanup_code(response, args.dataset_type, lang)
        results.append(response)

    res = {
        "message": "success",
        "result": {
            "input": {
                "lang": lang,
                "n": n,
            },
            "output": {"code": results},
        },
        "status": 0,
    }

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = device()
    uvicorn.run(app, host=args.listen, port=args.port, workers=args.workers)
