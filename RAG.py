from imports import *

class RagOpenAI:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.data = None
        self.openai_token = None
        self.client = None
        self.collection = None

    def load_from_json(self):
        with open("clean_output.json", "r") as f:
            self.data = json.load(f)
        ids = self.data.get('ids')
        documents = self.data.get('documents')
        metadata = self.data.get('metadata')
        vectors = self.data.get('vectors')
        return ids, documents, metadata, vectors

    def initialize_openai(self):
        load_dotenv()
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
        self.client = OpenAI()
        openai_token = os.environ['OPENAI_API_KEY']
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_token,
            model_name="text-embedding-3-large"  # Choose your embedding model
        )
        client = chromadb.PersistentClient(path="products_vectorstore")
        self.collection = client.get_or_create_collection('products')


    def description(self,item):
        text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
        return text.split("\n\nPrice is $")[0]
    def create_element(self):
        with open("train.pkl", "rb") as f:
            train = pickle.load(f)



        with open("output.json", "r") as f:
            vectors = json.load(f)


        client = chromadb.PersistentClient(path="products_vectorstore")
        collection_name = "products"
        existing_collection_names = [collection.name for collection in client.list_collections()]
        if collection_name in existing_collection_names:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
            self.collection = client.create_collection(collection_name)


        for i in tqdm(range(0, len(train), 1000)):
            documents = [self.description(item) for item in train[i: i + 1000]]
            #vectors = model.encode(documents).astype(float).tolist()
            metadatas = [{"category": item.category, "price": item.price} for item in train[i: i + 1000]]
            ids = [f"doc_{j}" for j in range(i, i + 1000)]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )

    def find_similars(self,item):
        results = self.collection.query(query_embeddings=self.model.encode(item).astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        return documents, prices

    def get_price(self,s):
        s = s.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0

    def make_context(self,similars, prices):
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message
    def messages_for(self,description, similars, prices):
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def predict(self,description):
        documents, prices = self.find_similars(description)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        return self.get_price(reply)


