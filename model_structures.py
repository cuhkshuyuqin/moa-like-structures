from typing import Dict

from Agents.Proposer import Proposer
from Agents.Aggregator import Aggregator
from Agents.BaseAgent import BaseAgent


def get_token_costs(agents: Dict[str, BaseAgent]):
    token_costs = ""
    total_input_tokens_count = 0
    total_output_tokens_count = 0
    for name, instance in agents.items():
        token_costs += f"{name}:\ninput:  {instance.get_total_input_tokens}\noutput: {instance.get_total_input_tokens}\n\n"
        total_input_tokens_count += instance.get_total_input_tokens
        total_output_tokens_count += instance.get_total_output_tokens
    
    token_costs += f"total:\ninput:  {total_input_tokens_count}\noutput: {total_output_tokens_count}\ntotal:  {total_input_tokens_count + total_output_tokens_count}"


def structure_0(query):
    return "A"


def structure_1(query):
    model = "Qwen/Qwen2-7B-Instruct"

    proposer = Proposer(model, query)

    return proposer.generate()


def structure_2(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])

    return layer2_1.generate()


def structure_3(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])
    layer2_2 = Aggregator(model, query, [layer1_1, layer1_2])

    layer3_1 = Aggregator(model, query, [layer2_1, layer2_2])

    return layer3_1.generate()


def structure_4(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6])

    return layer1_1.generate()

def structure_5(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])

    layer2_1 = Aggregator(model, query, [layer0_4, layer0_5, layer1_1])

    return layer2_1.generate()


def structure_6(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1])

    layer2_1 = Aggregator(model, query, [layer1_1])

    layer3_1 = Aggregator(model, query, [layer2_1])

    layer4_1 = Aggregator(model, query, [layer3_1])

    layer5_1 = Aggregator(model, query, [layer4_1])

    layer6_1 = Aggregator(model, query, [layer5_1])

    return layer6_1.generate()


def structure_self_moa_sota(query):
    model = "wzhouad/gemma-2-9b-it-WPO-HB"

    layer0_1 = Proposer(model, query, 0.7)
    layer0_2 = Proposer(model, query, 0.7)
    layer0_3 = Proposer(model, query, 0.7)
    layer0_4 = Proposer(model, query, 0.7)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4], 0.7)

    return layer1_1.generate()


def structure_self_moa_sota(query):
    model = "wzhouad/gemma-2-9b-it-WPO-HB"

    layer0_1 = Proposer(model, query, 0.7)
    layer0_2 = Proposer(model, query, 0.7)
    layer0_3 = Proposer(model, query, 0.7)
    layer0_4 = Proposer(model, query, 0.7)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4], 0.7)

    return layer1_1.generate()


def structure_token_cost_1(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6])

    token_costs = get_token_costs({
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer1_1": layer1_1,
    })

    return layer1_1.generate(), token_costs