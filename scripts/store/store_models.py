from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("BM-K/KoSimCSE-bert")
    model.save("models/before_tuned/kosimcse")

    # model = SentenceTransformer("snunlp/KR-BGE-small")  # hypothetical name
    # model.save("models/before_tuned/bge-korean")

    # model = SentenceTransformer("intfloat/multilingual-e5-small")
    # model.save("models/before_tuned/e5-multilingual")
