from collections import defaultdict
from typing import Literal
from roboquant.strategies.strategy import Strategy
from roboquant.event import Event


class MultiStrategy(Strategy):
    """Combine one or more strategies. The MultiStrategy provides additional control on how to handle conflicting
    ratings for the same symbols:

    - avg: average the ratings, and optionally apply weights based on the stratetegy that generated the rating
    - first: in case of multiple ratings for a symbol, the first strategy wins
    - last:  in case of multiple ratings for a symbol, the last strategy wins. This is also the default policy
    """

    def __init__(
        self, *strategies: Strategy, policy: Literal["avg", "last", "first"] = "last", weights: list[float] | None = None
    ):
        self.strategies = list(strategies)
        self.policy = policy
        self.weights = weights or [1.0] * len(strategies)
        assert len(self.weights) == len(self.strategies), "weights and strategies should be of equal lenght"

    def give_ratings(self, event: Event) -> dict[str, float]:
        all_ratings: list[dict[str, float]] = []
        for strategy in self.strategies:
            ratings = strategy.give_ratings(event)
            all_ratings.append(ratings)

        result = defaultdict(lambda: 0.0)
        match self.policy:
            case "last":
                for ratings in all_ratings:
                    result.update(ratings)
            case "first":
                for ratings in reversed(all_ratings):
                    result.update(ratings)
            case "avg":
                symbol_weight = defaultdict(lambda: 0.0)
                for ratings, weight in zip(all_ratings, self.weights):
                    for symbol, rating in ratings.items():
                        result[symbol] += rating * weight
                        symbol_weight[symbol] += weight

                for symbol in result.keys():
                    result[symbol] /= symbol_weight[symbol]

        return result
