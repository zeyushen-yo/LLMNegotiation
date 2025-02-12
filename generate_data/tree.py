import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt import llm_judge

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
        self.reward = None  # To be set later

    def get_full_conversation(self):
        """
        Walks up the tree (from the root) and returns the conversation history.
        Each turn is prefixed with the agent’s name.
        """
        conversation = []
        node = self
        while node is not None:
            if node.parent is not None and node.message:
                conversation.append(f"{node.agent}: {node.message}")
            node = node.parent
        conversation.reverse()
        return "\n".join(conversation)

def generate_model_response(model, tokenizer, conversation, setting, max_new_tokens):
    prompt = conversation
    if setting and "scenario" in setting:
        prompt += "\nSetting: " + setting["scenario"] + "\n"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the prompt part (if present) so that we return only the new response.
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()
        
    return response

def expand_node(node, depth, max_depth, span, agent, setting,
                rl_model, rl_tokenizer, ref_model, ref_tokenizer):
    """
    Recursively expands the negotiation tree.
    
    - node: current TreeNode
    - depth: current depth (0 means no turns yet)
    - max_depth: maximum negotiation turns
    - span: branching factor (number of proposals/responses per turn)
    - agent: the agent ("rl" or "ref") who is to speak at this node
    - setting: the negotiation setting (a dict)
    - rl_model, rl_tokenizer: the fine-tuned RL model and tokenizer
    - ref_model, ref_tokenizer: the reference model and tokenizer
    """
    if depth >= max_depth:
        full_conv = node.get_full_conversation()
        node.reward = llm_judge(full_conv, setting)
        return

    if agent == "rl":
        model, tokenizer = rl_model, rl_tokenizer
    else:
        model, tokenizer = ref_model, ref_tokenizer

    conversation = node.get_full_conversation()
    for i in range(span):
        response_text = generate_model_response(model, tokenizer, conversation, setting)
        child_node = TreeNode(message=response_text, agent=agent, parent=node)
        node.children.append(child_node)
        # Alternate the agent for the next turn.
        next_agent = "ref" if agent == "rl" else "rl"
        expand_node(child_node, depth + 1, max_depth, k, next_agent, setting,
                    rl_model, rl_tokenizer, ref_model, ref_tokenizer)
    
    # Back-propagate the reward: average of the children’s rewards.
    node.reward = sum(child.reward for child in node.children) / len(node.children)

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