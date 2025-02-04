import os,json,math,copy
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel,GPT2Tokenizer

def load_data(file):
    # load dataset
    with open(file, 'r') as f:
        dat = json.load(f)
    return [{"question": entry["question"], "answer": entry["answer"]} 
            for entry in dat if "question" in entry and "answer" in entry]


def load_data(file):
    # load dataset
    with open(file, 'r') as f:
        stuff = json.load(f)
    return [{"question": entry["question"], "answer": entry["answer"]} 
            for entry in stuff if "question" in entry and "answer" in entry]

def extract_answer(text):
    # extract answer between <answer> tags; fallback to last line if not found
    s_tag = "<answer>"
    e_tag = "</answer>"
    s = text.find(s_tag)
    e = text.find(e_tag, s)
    if s != -1 and e != -1:
        return text[s+len(s_tag):e].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""

def compute_log_prob(model, tokenizer, prompt, response, use_no_grad=True):
    # token-level log probs for response given prompt
    full_text = prompt + response
    inputs_full = tokenizer(full_text, return_tensors="pt")
    inputs_prompt = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids_full = inputs_full.input_ids.to(device)
    attention_mask_full = inputs_full.attention_mask.to(device)
    prompt_ids = inputs_prompt.input_ids.to(device)
    prompt_length = prompt_ids.size(1)

    #print(" shape check") 
    #print(f"input_ids_full shape: {input_ids_full.shape}")

    if use_no_grad:
        with torch.no_grad():
            outputs = model(input_ids_full, attention_mask=attention_mask_full)
            lgt = outputs.logits
    else:
        outputs = model(input_ids_full, attention_mask=attention_mask_full)
        lgt = outputs.logits

    lgt = lgt[:, :-1, :]
    target_ids = input_ids_full[:, 1:]
    response_ids = target_ids[0, prompt_length-1:]
    log_probs = F.log_softmax(lgt, dim=-1)
    response_log_probs = log_probs[0, prompt_length-1:, :].gather(1, response_ids.unsqueeze(1))
    return response_log_probs.sum()


def kl_divergence(current_model, ref_model, tokenizer, prompt, response):
    full_text = prompt + response
    inps = tokenizer(full_text, return_tensors="pt")
    device = next(current_model.parameters()).device
    ids = inps.input_ids.to(device)
    mask = inps.attention_mask.to(device)

    with torch.no_grad():
        outputs_cur = current_model(ids, attention_mask=mask)
        outputs_ref = ref_model(ids, attention_mask=mask)

    logits_cur = outputs_cur.logits
    logits_ref = outputs_ref.logits

    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.to(device).size(1)
    logits_cur_resp = logits_cur[:, prompt_len-1:-1, :]
    logits_ref_resp = logits_ref[:, prompt_len-1:-1, :]
    
    log_probs_cur = F.log_softmax(logits_cur_resp, dim=-1)
    probs_ref = F.softmax(logits_ref_resp, dim=-1)
    return F.kl_div(log_probs_cur, probs_ref, reduction='batchmean')

GROUP_SIZE = 1024 # CHANGE THIS
EPS = 0.2
KL_WEIGHT = 0.1
LR = 1e-5
EPOCHS = 3

# setup model and tokenizer
model_name="carsenk/llama3.2_1b_2025_uncensored_v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tok = AutoTokenizer.from_pretrained(model_name)

#model = GPT2LMHeadModel.from_pretrained('gpt2')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if tok.pad_token is None or tok.pad_token == tok.eos_token: tok.add_special_tokens({"pad_token": "<pad>"})
model.resize_token_embeddings(len(tok))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# setup reference models
model_old = copy.deepcopy(model)
model_old.to(device)
model_old.eval()

ref_model = copy.deepcopy(model)
ref_model.to(device)
ref_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)
qa_data = load_data("qa_data.json")

# training loop
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1} ===")
    for idx, qa in enumerate(qa_data):
        p = qa["question"] + ". answer in scientific notation if large (eg. 2.1e+8 in**2, 81, 2.00E+27, 2.2e-4 atm)\n<think> "
        ground_truth = qa["answer"].strip()
        group_log_probs = []
        group_rewards = []
        group_texts = []

        for i in range(GROUP_SIZE):
            inputs = tok(p, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            output_ids = model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        max_length=input_ids.size(1) + 50,
                                        do_sample=True,
                                        temperature=1.0,
                                        top_p=0.95,
                                        pad_token_id=tok.pad_token_id)
            
            generated_text = tok.decode(output_ids[0], skip_special_tokens=False)
            group_texts.append(generated_text)

            response_text = generated_text[len(p):]

            log_prob = compute_log_prob(model, tok, p, response_text, use_no_grad=False)
            group_log_probs.append(log_prob)

            predicted_answer = extract_answer(generated_text)
            # print(predicted_answer)
            rwd = 1.0 if ground_truth.lower() in predicted_answer.lower() else 0.0
            group_rewards.append(rwd)

        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        group_log_probs_old=[]
        for gen_text in group_texts:
            lp_old = compute_log_prob(model_old, tok, p, gen_text[len(p):], use_no_grad=True)
            group_log_probs_old.append(lp_old)

        log_probs_tensor = torch.stack(group_log_probs)
        log_probs_old_tensor = torch.stack(group_log_probs_old)
        ratios = torch.exp(log_probs_tensor - log_probs_old_tensor)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - EPS, 1 + EPS) * advantages
        loss_clip = -torch.mean(torch.min(surrogate1, surrogate2))

        kl_divs = []
        for gen_text in group_texts:
            kl = kl_divergence(model, ref_model, tok, p, gen_text[len(p):])
            kl_divs.append(kl)
        kl_penalty = torch.stack(kl_divs).mean()
        # print("Surrogate1: {:.4f}, Surrogate2: {:.4f}, LossClip: {:.4f}, KL: {:.4f}".format(surrogate1.mean().item(), surrogate2.mean().item(), loss_clip.item(), kl_penalty.item()))

        loss=loss_clip+KL_WEIGHT*kl_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_old.load_state_dict(model.state_dict())

        print(f"Sample {idx+1}: loss={loss.item():.4f}, avg reward={mean_reward.item():.2f}, kl={kl_penalty.item():.4f}")
