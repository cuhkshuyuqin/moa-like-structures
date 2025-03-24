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

    proposer1 = Proposer(model, query)
    proposer2 = Proposer(model, query)

    aggregator = Aggregator(
        model, query, [proposer1, proposer2]
    )

    return aggregator.generate()
