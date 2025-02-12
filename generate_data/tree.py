import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt import get_output
from process_text import get_score, extract_from_text
from prompts import reward_judge_prompt, negotiation_start_prompt, negotiation_respond_prompt

class TreeNode:
    def __init__(self, message, agent, parent=None):
        """
        message: the text for this turn (empty for the root)
        agent: which model produced the message (we restrict agent names to 'A' and 'B')
        parent: pointer to the parent TreeNode (None for the root)
        """
        self.message = message
        self.agent = agent
        self.parent = parent
        self.children = []
        self.reward = None  # a dictionary, maps each agent to the reward of that agent


def generate_model_response(current_agent, setting, model, tokenizer, current_conversation, 
                            max_new_tokens, do_sample, temperature, agents):
    conversation = "\n".join(current_conversation)
    if len(current_conversation) == 1:
        prompt = negotiation_start_prompt(negotiation_setting=setting,
                                          agent1=agents[0],
                                          agent2=agents[1],
                                          current_agent=current_agent)
    else:
        prompt = negotiation_respond_prompt(negotiation_setting=setting,
                                            conversation_history=conversation
                                            agent1=agents[0],
                                            agent2=agents[1],
                                            current_agent=current_agent)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
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
        
    return extract_from_text(response)

def expand_node(node, depth, max_depth, span, current_agent, setting, current_conversation,
                judge_model, rl_model, rl_tokenizer, ref_model, ref_tokenizer, 
                max_new_tokens, do_sample, temperature, agents):
    """
    Recursively expands the negotiation tree.
    
    - node: current TreeNode
    - depth: current depth (0 means no turns yet)
    - span: branching factor (number of proposals/responses per turn)
    """
    if depth >= max_depth or "NEGOTIATION END" in current_conversation:
        full_conv = "\n".join(current_conversation)
        for ag in agents:
            prompt = reward_judge_prompt(negotiation_setting=setting,
                                         conversation_history=full_conv,
                                         agent1=agents[0],
                                         agent2=agents[1],
                                         current_agent=current_agent)
            output = get_output(prompt, judge_model, max_new_tokens)
            node.reward[ag] = get_score(output)
        return

    if current_agent == agent[0]:
        model, tokenizer = rl_model, rl_tokenizer
    else:
        model, tokenizer = ref_model, ref_tokenizer

    for i in range(span):
        response_text = generate_model_response(current_agent=current_agent, setting=setting, model=model, 
                                                tokenizer=tokenizer, current_conversation=current_conversation, 
                                                max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                                temperature=temperature, agents=agents)
        child_node = TreeNode(message=response_text, agent=current_agent, parent=node)
        node.children.append(child_node)
        current_conversation.append(f"{current_agent}: {response_text}")

        # alternate the agent for the next turn.
        next_agent = agents[1] if current_agent == agents[0] else agents[1]
        expand_node(child_node, depth + 1, max_depth, span, next_agent, setting, current_conversation,
                    judge_model, rl_model, rl_tokenizer, ref_model, ref_tokenizer,
                    max_new_tokens, do_sample, temperature, agents)

        current_conversation.pop()
    
    # back-propagate the reward: average of the children’s rewards.
    rewards = {}
    for ag in agents:
        rewards[ag] = sum(child.reward[ag] for child in node.children) / len(node.children)
    node.reward = rewards

def get_all_paths(node, current_path=None):
    """
    Recursively collects all paths (from the root to each leaf) in the negotiation tree.
    Each path is returned as a list of TreeNodes (excluding the root if it has an empty message).
    """
    if current_path is None:
        current_path = []
    if node.parent is not None:
        current_path = current_path + [node]
    if not node.children:
        yield current_path
    else:
        for child in node.children:
            yield from(get_all_paths(child, current_path))