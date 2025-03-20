from imports import *

class RagOpenAI:
    def __init__(self):
        self.model = 'gpt-4o-mini'
        self.data = None
        self.openai_token = None
        self.client = None
        self.collection = None

    def load_from_json(self):
        with open("output.json", "r") as f:
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


        self.data = pd.read_csv("youtube_titles.csv")

        chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Use an in-memory or persistent database
        # self.collection = chroma_client.get_or_create_collection(name="openai_embeddings", embedding_function=openai_ef)
        chroma_client.delete_collection(name="openai_embeddings")
        self.collection = chroma_client.create_collection(name="openai_embeddings", embedding_function=openai_ef)
        ids, documents, metadata, vectors = self.load_from_json()
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadata,
            embeddings=vectors
        )
    def train_test_items_split(self):
        train, test = train_test_split(self.data, test_size=0.2, random_state=42)
        return train,test

    def create_element(self):
        train,test = self.train_test_items_split()
        ids = [f"doc_{i}" for i in range(len(train))]
        documents = [train['title'].iloc[i] for i in range(len(train))]
        metadatas = [{"category": train['category_1'].iloc[i], "Vid_id": train['vid_id'].iloc[i]} for i in
                     range(len(train))]
        vectors = [
            self.client.embeddings.create(
                input=train['title'].iloc[i],
                model="text-embedding-3-large",
            ) for i in range(len(train))
        ]
        vector_values = [vector.data[0].embedding for vector in vectors]
        return ids, documents, metadatas, vector_values
    def write_to_json(self):
        # Extract embeddings from CreateEmbeddingResponse objects
        ids, documents, metadatas, vector_values = self.create_element()

        data = {
            "ids": ids,
            "documents": documents,
            "metadata": metadatas,
            "vectors": vector_values  # Use the extracted embeddings
        }

        # Save to a JSON file
        with open("output.json", "w") as f:
            json.dump(data, f, indent=4)



    def find_similars(self,product):
        vector_item = self.client.embeddings.create(input=product, model='text-embedding-3-large')
        results = self.collection.query(query_embeddings=vector_item.data[0].embedding, n_results=5)
        title = []
        vid_id = []
        for i in range(len(results['documents'][0])):
            title.append(results['documents'][0][i])
            vid_id.append(results['metadatas'][0][i]['Vid_id'])
        return title, vid_id

    def messages_for(self,item):
        system_message = "You estimate which video below is suitable for the keyword that user want.Only return video title and Video ID"
        title, vid_id = self.find_similars(item)
        user_prompt = "Here is 5 choices that you can choose from:\n"
        for i in range(len(title)):
            user_prompt += f"1. Title: {title[i]}\n Video ID: {vid_id[i]}\n"
        user_prompt += "And now the question for you:\n\n"
        user_prompt += f"Analyze and determine three titles suitable for the keyword from 5 options above: {item}\n"
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Title is , video id is"}
        ]

    def reply_from_chat_bot(self,item):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.messages_for(item)
        )
        result_from_ai = response.choices[0].message.content
        return result_from_ai


