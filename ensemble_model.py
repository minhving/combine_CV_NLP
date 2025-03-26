from imports import *


class EnsembleAgent:
    name = "Ensemble Agent"

    def __init__(self):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.rag = RagOpenAI()
        self.random = RandomForest()
        self.pricer = None
        self.model = joblib.load('ensemble_model.pkl')
    def initialize(self):
        self.rag.initialize_openai()
        self.random.initialize()

        self.pricer = modal.Function.from_name("pricer-service", "price")
        self.pricer.keep_warm(2)
        self.pricer.remote("Phone")
    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        rag_result = self.rag.predict(description)
        frontier_result = self.pricer.remote(description)
        random_forest = self.random.predict(description)
        X = pd.DataFrame({
            'Specialist': [rag_result],
            'Frontier': [frontier_result],
            'RandomForest': [random_forest],
            'Min': [min(rag_result, frontier_result, random_forest)],
            'Max': [max(rag_result, frontier_result, random_forest)],
        })
        y = self.model.predict(X)[0]
        return y