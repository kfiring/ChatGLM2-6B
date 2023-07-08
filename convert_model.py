from transformers import AutoModel, AutoTokenizer
from fastllm_pytools import llm
import argparse, os


if __name__ == "__main__":    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--in-model", "-i", type=str, required=True)
    arg_parser.add_argument("--out-dir", "-o", type=str)
    arg_parser.add_argument("--quantize-bit", "-q", type=int, choices=[4,8,16], required=True)
    
    args = arg_parser.parse_args()
    
    in_model = os.path.abspath(args.in_model)
    if not os.path.exists(in_model):
        raise Exception(f"input model '{in_model}' not exist")
    
    model_name = os.path.basename(in_model)
    if os.path.isfile(in_model):
        model_name, _ = os.path.splitext(model_name)
    
    if args.quantize_bit == 4:
        model_name += "-int4"
        qbit = "int4"
    elif args.quantize_bit == 8:
        model_name += "-int8"
        qbit = "int8"
    elif args.quantize_bit == 16:
        model_name += "-fp16"
        qbit = "float16"
    model_name += ".flm"
    
    out_dir = args.out_dir
    if not out_dir:
        out_dir = "."
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"created output dir '{out_dir}'")
    
    out_model = os.path.join(out_dir, model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(in_model, trust_remote_code=True)
    orig_model = AutoModel.from_pretrained(in_model, trust_remote_code=True)
    
    orig_model = orig_model.eval()
    model = llm.from_hf(orig_model, tokenizer, dtype = qbit)
    del orig_model
    
    model.save(out_model)
    print(f"saved converted model '{out_model}'")
    