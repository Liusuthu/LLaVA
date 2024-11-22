import torch
import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

os.environ['http_proxy'] = 'http://127.0.0.1:34780'
os.environ['https_proxy'] = 'http://127.0.0.1:34780'
os.environ['all_proxy'] ='socks5://127.0.0.1:34780'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# manual_device=torch.device('cuda:1')
model_path = "liuhaotian/llava-v1.5-13b"
# model_path = "liuhaotian/llava-v1.5-7b"


# print(torch.cuda.current_device())

# print(get_model_name_from_path(model_path)) #llava-v1.5-7b
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    # device=manual_device,
)


prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)