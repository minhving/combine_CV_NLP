from imports import *

class RandomForest:
    def __init__(self):
        self.n_estimators = 100
        self.data = None
        self.price = []
        self.model_predict = None
        self.model_encode = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def initialize(self):
        self.loadData()
        self.train()
    def loadData(self):
        with open("output.json", "r") as f:
            self.data = json.load(f)
        with open('train.pkl', 'rb') as file:
            train = pickle.load(file)

        for i in range(0, 1000):
            self.price.append(train[i].price)

    def train(self):
        self.model_predict = RandomForestRegressor(n_estimators=self.n_estimators)
        self.model_predict.fit(self.data, self.price)
    def predict(self,product):

        vectors = self.model_encode.encode("Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio").astype(float).tolist()
        return self.model_predict.predict([vectors])[0]

