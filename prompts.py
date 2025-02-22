reward_judge_prompt = '''Given a negotiation setting:

{negotiation_setting},

and a conversation history between agents {agent1} and {agent2}:

{conversation_history},

provide an evaluation of agent {current_agent}'s gain from this negotiation in terms of a number between 0 and 1. Wrap your answer in a box.
'''


negotiation_start_prompt = '''Given a negotiation setting:

{negotiation_setting}

between agents {agent1} and {agent2}. Assume you are agent {current_agent}. Please provide **only one turn** of dialogue – just your opening proposal – and nothing more. Your response should be limited to a single, complete turn without any extra commentary or continuation. Format your answer as follows:

Answer: [Your answer]
'''


negotiation_respond_prompt = '''Given a negotiation setting:

{negotiation_setting}

and a conversation history between agents {agent1} and {agent2}:

{conversation_history}.

Assume you are agent {current_agent}. Please provide **only one immediate turn** in response to the conversation history. Do not generate an extended conversation—just a single, complete turn. If this turn concludes the negotiation, append [NEGOTIATION END] (in all caps) at the end of your response. Format your answer as follows:

Answer: [Your answer]
'''


generate_setting_prompt = '''Generate a hypothetical negotiation setting between two agents {agent1} and {agent2}. You should specify the following:

1. The definition of the setting; e.g. Two agents {agent1} and {agent2} are playing a buy-sell game. Agent {agent1} is the seller and agent {agent2} is the buyer.

2. What each agent have in the beginning; e.g. Agent {agent1} has a product whose production cost is $40, and agent {agent2} has $60.

3. The goal of each agent; e.g. Agent {agent1} wants to maximize her profit, and agent {agent2} wants to minimize her payment.

The setting should be well-defined and clear. Also, it can be any reasonable setting apart from the classical buy-sell game. Provide your answer in the following format:

Answer: [Your hypothetical negotiation setting].
'''

identify_strategy_prompt = '''Given a negotiation setting:

{negotiation_setting}

and a conversation history between agents {agent1} and {agent2}:

{conversation_history}.

The following is the response from {current_agent} to this conversation history:

{response_text}.

Please identify the strategy used in this response and summarize in a few words. Provide your answer in the following format:

Answer: [Your identified strategy].
'''