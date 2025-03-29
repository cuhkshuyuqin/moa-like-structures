from Agents.Proposer import Proposer
from Agents.Aggregator import Aggregator


def structure_0(query):
    return 'A'


def structure_1(query):
    model = "Qwen/Qwen2-7B-Instruct"

    proposer = Proposer(model, query)

    return proposer.generate()


def structure_2(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)

    layer1_1 = Aggregator(
        model, query, [layer0_1, layer0_2, layer0_3]
    )
    layer1_2 = Aggregator(
        model, query, [layer0_1, layer0_2, layer0_3]
    )
    layer1_3 = Aggregator(
        model, query, [layer0_1, layer0_2, layer0_3]
    )

    layer2_1 = Aggregator(
        model, query, [layer1_1, layer1_2, layer1_3]
    )

    return layer2_1.generate()


def structure_3(query):
    model = "Qwen/Qwen2-7B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)

    layer1_1 = Aggregator(
        model, query, [layer0_1, layer0_2]
    )
    layer1_2 = Aggregator(
        model, query, [layer0_1, layer0_2]
    )

    layer2_1 = Aggregator(
        model, query, [layer1_1, layer1_2]
    )
    layer2_2 = Aggregator(
        model, query, [layer1_1, layer1_2]
    )

    layer3_1 = Aggregator(
        model, query, [layer2_1, layer2_2]
    )

    return layer3_1.generate()
