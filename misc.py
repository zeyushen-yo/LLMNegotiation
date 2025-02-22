import torch
from process_text import extract_from_text
from prompts import negotiation_start_prompt, negotiation_respond_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm(llm_path):
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(llm_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer


def generate_model_response(current_agent, setting, model, tokenizer, current_conversation, 
                            max_new_tokens, do_sample, temperature, agents):
    conversation = "\n".join(current_conversation)
    if len(current_conversation) == 1:
        prompt = negotiation_start_prompt.format(negotiation_setting=setting,
                                          agent1=agents[0],
                                          agent2=agents[1],
                                          current_agent=current_agent)
    else:
        prompt = negotiation_respond_prompt.format(negotiation_setting=setting,
                                            conversation_history=conversation,
                                            agent1=agents[0],
                                            agent2=agents[1],
                                            current_agent=current_agent)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    # sometimes models fail to follow the required format
    while True:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # remove the prompt part (if present) so that we return only the new response.
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        if "Answer:" in response:   
            print("prompt start here!: ", prompt)
            print("response start here!: ", response) 
            print("-------------------------------------Response end here!------------------------------------------")
            return extract_from_text(response, "Answer:")