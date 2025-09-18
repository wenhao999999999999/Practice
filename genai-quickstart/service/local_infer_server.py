import argparse, os, uvicorn, torch, json
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

app = FastAPI(title="Local OpenAI-Compatible")

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.2
    max_tokens: int = 512

def load_model(base, lora=None):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(base, device_map="auto", trust_remote_code=True, quantization_config=quant)
    if lora:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, lora)
    return tok, mdl

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LORA = os.getenv("LORA_ADAPTER", None)
tok, mdl = load_model(BASE, LORA)

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt = "\n".join([m.get("content","") for m in req.messages])
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=req.max_tokens, temperature=req.temperature, do_sample=True, streamer=streamer)
    thread = Thread(target=mdl.generate, kwargs=gen_kwargs)
    thread.start()
    out_text = ""
    for s in streamer:
        out_text += s
    return {"id":"chatcmpl-local","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":out_text},"finish_reason":"stop"}]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()
    if args.base: os.environ["BASE_MODEL"] = args.base
    if args.lora: os.environ["LORA_ADAPTER"] = args.lora
    uvicorn.run(app, host="0.0.0.0", port=args.port)
