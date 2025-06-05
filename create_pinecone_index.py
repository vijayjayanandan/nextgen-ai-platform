from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_28G6m7_2C6smHBE16GHWU2mA5G5vtW7aWSiqygbTz4WUQ4QmSXNf2NNKpTfhV4DarVfXvn")

# Create index if it doesn't exist
index_name = "ircc-documents"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index {index_name} already exists")

# Get index stats
index = pc.Index(index_name)
print(index.describe_index_stats())
