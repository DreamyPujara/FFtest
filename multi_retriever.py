from langchain.schema import BaseRetriever
from pydantic import PrivateAttr
from langchain.schema.document import Document

class MultiCollectionRetriever(BaseRetriever):
    _retriever1: BaseRetriever = PrivateAttr()
    _retriever2: BaseRetriever = PrivateAttr()

    def __init__(self, retriever1: BaseRetriever, retriever2: BaseRetriever):
        super().__init__()
        self._retriever1 = retriever1
        self._retriever2 = retriever2

    def get_relevant_documents(self, query: str):
        print(f"get_relevant_documents --> query ---> {query}")
        docs1 = self._retriever1.get_relevant_documents(query)
        docs2 = self._retriever2.get_relevant_documents(query)
        return docs1+docs2

    async def aget_relevant_documents(self, query: str):
        print(f"aget_relevant_documents --> query ---> {query}")
        docs1 = await self._retriever1.aget_relevant_documents(query)
        docs2 = await self._retriever2.aget_relevant_documents(query)
        return docs1 + docs2
