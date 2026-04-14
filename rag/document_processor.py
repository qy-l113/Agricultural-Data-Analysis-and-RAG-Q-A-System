# rag/document_processor_local.py
import os

# ===== 强制离线模式（禁止一切联网请求）=====
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# =========================================
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class DocumentProcessorLocal:
    """使用本地 embedding 模型（不需要API，100%成功）"""

    def __init__(self, data_path=None):
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.data_path = os.path.join(project_root, 'data', 'knowledge_base')
        else:
            self.data_path = data_path

        self.vectordb_path = os.path.join(project_root, 'vectordb')

        # ===== 新增：本地模型路径 =====
        self.local_model_path = os.path.join(project_root, 'models', 'paraphrase-multilingual-MiniLM-L12-v2')

        print(f"知识库路径: {self.data_path}")
        print(f"向量库路径: {self.vectordb_path}")
        print(f"本地模型路径: {self.local_model_path}")

    def build_vectorstore(self):
        """构建向量数据库"""
        if not os.path.exists(self.data_path):
            print(f"❌ 知识库路径不存在: {self.data_path}")
            return None

        # 加载文档
        print("正在加载文档...")
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()

        if not documents:
            print(f"❌ 未找到知识文档")
            return None

        print(f"✅ 找到 {len(documents)} 个文档:")
        for doc in documents:
            rel_path = os.path.relpath(doc.metadata['source'], start=os.path.dirname(self.data_path))
            print(f"  - {rel_path}")

        # 分割文档
        print("正在分割文档...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        print(f"📄 文档分割为 {len(splits)} 个文本块")

        # ===== 关键修改：使用本地模型路径 =====
        print("🔄 正在加载本地 embedding 模型...")

        # 检查本地模型是否存在
        if os.path.exists(self.local_model_path):
            print(f"   ✅ 使用本地模型: {self.local_model_path}")
            model_name = self.local_model_path
        else:
            print(f"   ⚠️ 本地模型不存在，使用在线模型（需要联网）")
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,  # ← 改这里
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("🔄 正在创建向量数据库...")
        vectordb = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=self.vectordb_path
        )
        vectordb.persist()

        print(f"\n✅ 向量数据库构建完成！")
        print(f"   保存路径: {self.vectordb_path}")
        print(f"   文档数量: {len(documents)}")
        print(f"   文本块数: {len(splits)}")

        return vectordb

    def load_vectorstore(self):
        """加载已存在的向量数据库（用于检索，不重新构建）"""
        if not os.path.exists(self.vectordb_path):
            print(f"❌ 向量数据库不存在: {self.vectordb_path}")
            return None

        print(f"🔄 加载本地模型: {self.local_model_path}")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.local_model_path,  # ← 使用本地路径
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectordb = Chroma(
            persist_directory=self.vectordb_path,
            embedding_function=embeddings
        )
        print(f"✅ 加载向量数据库成功，共 {vectordb._collection.count()} 个向量")
        return vectordb


if __name__ == "__main__":
    processor = DocumentProcessorLocal()
    processor.build_vectorstore()