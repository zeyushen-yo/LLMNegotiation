reward_judge_prompt = '''Given a negotiation setting:

{negotiation_setting},

and a conversation history between agents {agent1} and {agents2}:

{conversation_history},

provide an evaluation of agent {current_agent}'s gain from this negotiation in terms of a number between 0 and 1. Wrap your answer in a box.
'''


negotiation_start_prompt = '''Given a negotiation setting:

{negotiation_setting}

between agents {agent1} and {agent2}. Imagine you are agent {current_agent}. How would you start the conversation? 
Provide your answer in the following format:

Answer: [Your answer]

without explanation.
'''


negotiation_respond_prompt = '''Given a negotiation setting:

{negotiation_setting},

and a conversation history between agents {agent1} and {agent2}:

{conversation_history}.

Imagine you are agent {current_agent}. How would you continue the conversation? Provide your immediate response to this conversation.
Provide your answer in the following format:

Answer: [Your answer]

without explanation. If the negotiation reaches an end, write:

[NEGOTIATION END]

by the end of your answer, in capitalization.
'''


generate_setting_prompt = '''Generate a hypothetical negotiation setting between two agents {agent1} and {agent2}. You should specify the following:

1. The definition of the setting; e.g. Two agents {agent1} and {agent2} are playing a buy-sell game. Agent {agent1} is the seller and agent {agent2} is the buyer.

2. What each agent have in the beginning; e.g. Agent {agent1} has a product whose production cost is $40, and agent {agent2} has $60.

3. The goal of each agent; e.g. Agent {agent1} wants to maximize her profit, and agent {agent2} wants to minimize her payment.

The setting should be well-defined and clear. Also, it can be any reasonable setting apart from the classical buy-sell game. Provide your answer in the following format:

Answer: [Your hypothetical negotiation setting].
'''