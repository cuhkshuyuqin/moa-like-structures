from typing import Dict, List

from Agents.Proposer import Proposer
from Agents.Aggregator import Aggregator
from Agents.BaseAgent import BaseAgent
from Agents.Refiner import Refiner


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


def structure_token_cost_0(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    only_model = Proposer(model, query)

    result = only_model.generate()

    token_costs = {
        "only_model": only_model,
    }

    return result, token_costs


def structure_token_cost_1(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_token_cost_1_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_token_cost_2(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1])
    layer2_1 = Aggregator(model, query, [layer1_1])
    layer3_1 = Aggregator(model, query, [layer2_1])
    layer4_1 = Aggregator(model, query, [layer3_1])
    layer5_1 = Aggregator(model, query, [layer4_1])
    layer6_1 = Aggregator(model, query, [layer5_1])

    result = layer6_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
        "layer6_1": layer6_1,
    }

    return result, token_costs


def structure_token_cost_2_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1])
    layer2_1 = Aggregator(model, query, [layer1_1])
    layer3_1 = Aggregator(model, query, [layer2_1])
    layer4_1 = Aggregator(model, query, [layer3_1])

    result = layer4_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
    }

    return result, token_costs


def structure_token_cost_2_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1])
    layer2_1 = Aggregator(model, query, [layer1_1])
    layer3_1 = Aggregator(model, query, [layer2_1])
    layer4_1 = Aggregator(model, query, [layer3_1])

    result = layer4_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
    }

    return result, token_costs


def structure_token_cost_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2])
    layer1_2 = Aggregator(model, query, [layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_token_cost_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    layer3_1 = Aggregator(model, query, [layer2_1])

    result = layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
    }

    return result, token_costs


def structure_chain_1(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_chain_2(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_chain_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)

    result = layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
    }

    return result, token_costs


def structure_chain_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)
    layer4_1 = Aggregator(model, query, [layer3_1], temperature)

    result = layer4_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
    }

    return result, token_costs


def structure_chain_5(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)
    layer4_1 = Aggregator(model, query, [layer3_1], temperature)
    layer5_1 = Aggregator(model, query, [layer4_1], temperature)

    result = layer5_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
    }

    return result, token_costs


def structure_chain_6(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature = 0.0
    # temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)
    layer4_1 = Aggregator(model, query, [layer3_1], temperature)
    layer5_1 = Aggregator(model, query, [layer4_1], temperature)
    layer6_1 = Aggregator(model, query, [layer5_1], temperature)

    result = layer6_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
        "layer6_1": layer6_1,
    }

    return result, token_costs


def structure_chain_7(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature = 0.0
    # temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)
    layer4_1 = Aggregator(model, query, [layer3_1], temperature)
    layer5_1 = Aggregator(model, query, [layer4_1], temperature)
    layer6_1 = Aggregator(model, query, [layer5_1], temperature)
    layer7_1 = Aggregator(model, query, [layer6_1], temperature)

    result = layer7_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
        "layer6_1": layer6_1,
        "layer7_1": layer7_1,
    }

    return result, token_costs


def structure_chain_8(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    temperature = 0.0
    # temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Aggregator(model, query, [layer0_1], temperature)
    layer2_1 = Aggregator(model, query, [layer1_1], temperature)
    layer3_1 = Aggregator(model, query, [layer2_1], temperature)
    layer4_1 = Aggregator(model, query, [layer3_1], temperature)
    layer5_1 = Aggregator(model, query, [layer4_1], temperature)
    layer6_1 = Aggregator(model, query, [layer5_1], temperature)
    layer7_1 = Aggregator(model, query, [layer6_1], temperature)
    layer8_1 = Aggregator(model, query, [layer7_1], temperature)

    result = layer8_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
        "layer6_1": layer6_1,
        "layer7_1": layer7_1,
        "layer8_1": layer8_1,
    }

    return result, token_costs


def structure_star_1(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_2(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_5(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_6(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_7(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)
    layer0_7 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6, layer0_7])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer0_7": layer0_7,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_star_8(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)
    layer0_7 = Proposer(model, query)
    layer0_8 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6, layer0_7, layer0_8])

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer0_7": layer0_7,
        "layer0_8": layer0_8,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_fully_connected_2_4_2(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs


async def structure_fully_connected_2_4_2_qwen3_30b_a3b(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = await layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_fully_connected_3_9_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)
    layer0_7 = Proposer(model, query)
    layer0_8 = Proposer(model, query)
    layer0_9 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6, layer0_7, layer0_8, layer0_9])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6, layer0_7, layer0_8, layer0_9])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6, layer0_7, layer0_8, layer0_9])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer0_7": layer0_7,
        "layer0_8": layer0_8,
        "layer0_9": layer0_9,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_chain_refiner_1(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Refiner(model, query, [layer0_1], temperature)

    result = layer1_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
    }

    return result, token_costs


def structure_chain_refiner_2(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Refiner(model, query, [layer0_1], temperature)
    layer2_1 = Refiner(model, query, [layer1_1], temperature)

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_chain_refiner_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Refiner(model, query, [layer0_1], temperature)
    layer2_1 = Refiner(model, query, [layer1_1], temperature)
    layer3_1 = Refiner(model, query, [layer2_1], temperature)

    result = layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
    }

    return result, token_costs


def structure_chain_refiner_4(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Refiner(model, query, [layer0_1], temperature)
    layer2_1 = Refiner(model, query, [layer1_1], temperature)
    layer3_1 = Refiner(model, query, [layer2_1], temperature)
    layer4_1 = Refiner(model, query, [layer3_1], temperature)

    result = layer4_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
    }

    return result, token_costs


def structure_chain_refiner_5(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    # temperature = 0.0
    temperature = 0.7

    layer0_1 = Proposer(model, query)
    layer1_1 = Refiner(model, query, [layer0_1], temperature)
    layer2_1 = Refiner(model, query, [layer1_1], temperature)
    layer3_1 = Refiner(model, query, [layer2_1], temperature)
    layer4_1 = Refiner(model, query, [layer3_1], temperature)
    layer5_1 = Refiner(model, query, [layer4_1], temperature)

    result = layer5_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer1_1": layer1_1,
        "layer2_1": layer2_1,
        "layer3_1": layer3_1,
        "layer4_1": layer4_1,
        "layer5_1": layer5_1,
    }

    return result, token_costs


def structure_complete_tree_3_3(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)
    layer0_7 = Proposer(model, query)
    layer0_8 = Proposer(model, query)
    layer0_9 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_4, layer0_5, layer0_6])
    layer1_3 = Aggregator(model, query, [layer0_7, layer0_8, layer0_9])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])

    result = layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer0_7": layer0_7,
        "layer0_8": layer0_8,
        "layer0_9": layer0_9,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
    }

    return result, token_costs


def structure_complete_tree(query, branch, depth):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    if branch < 1:
        raise Exception("Invalid branch")
    if depth < 1:
        raise Exception("Invalid branch")

    complete_tree = structure_complete_tree_generate(model, query, branch, depth)

    return complete_tree.generate()

def structure_complete_tree_generate(model, query, branch, depth):
    if depth == 1:
        return Proposer(model, query)
    
    return Aggregator(model, query, [structure_complete_tree_generate(model, query, branch, depth - 1)] * branch)

def structure_fully_connected(query, shape: List[int]):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    neurons = [[Proposer(model, query) for i in shape[0]]]
    for i in range(1, len(shape)):
        neurons.append([Aggregator(model, query, neurons[i]) for j in shape[i]])

    final_aggregator = Aggregator(model, query, neurons[len(shape) - 1])

    return final_aggregator.generate()

def structure_rectangle(query, depth, length):
    return structure_fully_connected(query, [length] * depth)

async def structure_qwen_30b_a3b(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)

    result = await layer0_1.generate()

    token_costs = {
        "layer0_1": layer0_1
    }

    return result, token_costs

async def structure_fully_connected_2_4_2_qwen_30b_a3b(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = await layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs

async def structure_complete_tree_3_2_qwen_30b_a3b(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2])
    layer1_2 = Aggregator(model, query, [layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = await layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs

async def structure_compare_original(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])
    layer2_2 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])
    layer2_3 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])

    layer3_1 = Aggregator(model, query, [layer2_1, layer2_2, layer2_3])

    result = await layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
        "layer2_2": layer2_2,
        "layer2_3": layer2_3,
        "layer3_1": layer3_1,
    }

    return result, token_costs

async def structure_compare_improved_1(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_3])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2])

    layer2_1 = Aggregator(model, query, [layer1_2, layer1_3])
    layer2_2 = Aggregator(model, query, [layer1_1, layer1_3])
    layer2_3 = Aggregator(model, query, [layer1_1, layer1_2])

    layer3_1 = Aggregator(model, query, [layer2_1, layer2_2, layer2_3])

    result = await layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
        "layer2_2": layer2_2,
        "layer2_3": layer2_3,
        "layer3_1": layer3_1,
    }

    return result, token_costs

async def structure_compare_improved_2(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3])

    layer2_1 = Aggregator(model, query, [layer1_2, layer1_3])
    layer2_2 = Aggregator(model, query, [layer1_1, layer1_3])
    layer2_3 = Aggregator(model, query, [layer1_1, layer1_2])

    layer3_1 = Aggregator(model, query, [layer2_1, layer2_2, layer2_3])

    result = await layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
        "layer2_2": layer2_2,
        "layer2_3": layer2_3,
        "layer3_1": layer3_1,
    }

    return result, token_costs

async def structure_compare_improved_3(query):
    model = "qwen/qwen3-30b-a3b"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_2, layer0_3])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_3])
    layer1_3 = Aggregator(model, query, [layer0_1, layer0_2])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])
    layer2_2 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])
    layer2_3 = Aggregator(model, query, [layer1_1, layer1_2, layer1_3])

    layer3_1 = Aggregator(model, query, [layer2_1, layer2_2, layer2_3])

    result = await layer3_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer1_3": layer1_3,
        "layer2_1": layer2_1,
        "layer2_2": layer2_2,
        "layer2_3": layer2_3,
        "layer3_1": layer3_1,
    }

    return result, token_costs

async def structure_fully_connected_2_4_2_qwen3_4b(query):
    model = "Qwen/Qwen3-4B"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])
    layer1_2 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4])

    layer2_1 = Aggregator(model, query, [layer1_1, layer1_2])

    result = await layer2_1.generate()

    token_costs = {
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer1_1": layer1_1,
        "layer1_2": layer1_2,
        "layer2_1": layer2_1,
    }

    return result, token_costs