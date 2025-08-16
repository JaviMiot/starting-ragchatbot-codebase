import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config, config
import chromadb


class TestConfig:
    """Test configuration loading and validation"""
    
    def test_default_config_values(self):
        """Test default configuration values"""
        test_config = Config()
        
        # Test default values
        assert test_config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert test_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert test_config.CHUNK_SIZE == 800
        assert test_config.CHUNK_OVERLAP == 100
        assert test_config.MAX_RESULTS == 5
        assert test_config.MAX_HISTORY == 2
        assert test_config.CHROMA_PATH == "./chroma_db"
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key_123'})
    def test_env_var_loading(self):
        """Test loading configuration from environment variables"""
        test_config = Config()
        test_config.reload()  # Reload to pick up patched environment
        assert test_config.ANTHROPIC_API_KEY == 'test_api_key_123'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        test_config = Config()
        test_config.reload()  # Reload to pick up cleared environment
        assert test_config.ANTHROPIC_API_KEY == ""
    
    def test_global_config_instance(self):
        """Test that global config instance is created"""
        assert config is not None
        assert isinstance(config, Config)


class TestEnvironmentDependencies:
    """Test environment and dependency availability"""
    
    def test_chromadb_import(self):
        """Test that ChromaDB can be imported"""
        try:
            import chromadb
            assert chromadb is not None
        except ImportError:
            pytest.fail("ChromaDB is not available")
    
    def test_sentence_transformers_import(self):
        """Test that sentence transformers can be imported"""
        try:
            import sentence_transformers
            assert sentence_transformers is not None
        except ImportError:
            pytest.fail("sentence-transformers is not available")
    
    def test_anthropic_import(self):
        """Test that Anthropic client can be imported"""
        try:
            import anthropic
            assert anthropic is not None
        except ImportError:
            pytest.fail("anthropic library is not available")
    
    def test_fastapi_import(self):
        """Test that FastAPI can be imported"""
        try:
            import fastapi
            assert fastapi is not None
        except ImportError:
            pytest.fail("fastapi is not available")


class TestDatabaseConnectivity:
    """Test database and storage connectivity"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_chroma_path = os.path.join(self.temp_dir, "test_chroma")
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chromadb_client_creation(self):
        """Test that ChromaDB client can be created"""
        try:
            client = chromadb.PersistentClient(
                path=self.test_chroma_path,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            assert client is not None
        except Exception as e:
            pytest.fail(f"ChromaDB client creation failed: {e}")
    
    def test_chromadb_collection_creation(self):
        """Test that ChromaDB collections can be created"""
        try:
            client = chromadb.PersistentClient(
                path=self.test_chroma_path,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            # Create a test collection
            collection = client.get_or_create_collection(
                name="test_collection",
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            
            assert collection is not None
            assert collection.name == "test_collection"
            
        except Exception as e:
            pytest.fail(f"ChromaDB collection creation failed: {e}")
    
    def test_embedding_model_loading(self):
        """Test that embedding model can be loaded"""
        try:
            embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Test embedding generation
            test_texts = ["This is a test sentence"]
            embeddings = embedding_fn(test_texts)
            
            assert embeddings is not None
            assert len(embeddings) == 1
            assert len(embeddings[0]) > 0  # Should have some dimensions
            
        except Exception as e:
            pytest.fail(f"Embedding model loading failed: {e}")
    
    def test_chromadb_basic_operations(self):
        """Test basic ChromaDB operations work"""
        try:
            client = chromadb.PersistentClient(
                path=self.test_chroma_path,
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            collection = client.get_or_create_collection(
                name="test_operations",
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            
            # Test add operation
            collection.add(
                documents=["Test document content"],
                metadatas=[{"test": "metadata"}],
                ids=["test_id_1"]
            )
            
            # Test query operation
            results = collection.query(
                query_texts=["Test query"],
                n_results=1
            )
            
            assert results is not None
            assert "documents" in results
            assert "metadatas" in results
            assert "distances" in results
            
        except Exception as e:
            pytest.fail(f"ChromaDB basic operations failed: {e}")


class TestFileSystemAccess:
    """Test file system access and permissions"""
    
    def test_docs_folder_access(self):
        """Test access to docs folder"""
        docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
        
        if os.path.exists(docs_path):
            assert os.path.isdir(docs_path)
            assert os.access(docs_path, os.R_OK)
            
            # Test reading files in docs folder
            files = os.listdir(docs_path)
            txt_files = [f for f in files if f.endswith('.txt')]
            
            if txt_files:
                test_file = os.path.join(docs_path, txt_files[0])
                assert os.access(test_file, os.R_OK)
                
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        assert len(content) > 0
                except Exception as e:
                    pytest.fail(f"Failed to read docs file: {e}")
    
    def test_chroma_db_directory_access(self):
        """Test access to ChromaDB directory"""
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        parent_dir = os.path.dirname(chroma_path)
        
        # Test parent directory is writable
        assert os.access(parent_dir, os.W_OK)
        
        # If chroma_db exists, test it's accessible
        if os.path.exists(chroma_path):
            assert os.path.isdir(chroma_path)
            assert os.access(chroma_path, os.R_OK | os.W_OK)
    
    def test_temp_directory_access(self):
        """Test temporary directory creation"""
        import tempfile
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                assert os.path.exists(temp_dir)
                assert os.access(temp_dir, os.R_OK | os.W_OK)
                
                # Test file creation in temp directory
                test_file = os.path.join(temp_dir, "test_file.txt")
                with open(test_file, 'w') as f:
                    f.write("test content")
                
                assert os.path.exists(test_file)
                
        except Exception as e:
            pytest.fail(f"Temporary directory access failed: {e}")


class TestAPIConnectivity:
    """Test API connectivity and authentication"""
    
    @patch('anthropic.Anthropic')
    def test_anthropic_client_creation(self, mock_anthropic):
        """Test Anthropic client can be created"""
        # Mock successful client creation
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        import anthropic
        client = anthropic.Anthropic(api_key="test_key")
        
        assert client is not None
        mock_anthropic.assert_called_once_with(api_key="test_key")
    
    def test_anthropic_api_key_validation(self):
        """Test API key validation"""
        # Test empty API key
        empty_config = Config()
        empty_config.ANTHROPIC_API_KEY = ""
        
        assert empty_config.ANTHROPIC_API_KEY == ""
        
        # Test valid-looking API key format
        valid_config = Config()
        valid_config.ANTHROPIC_API_KEY = "sk-ant-api03-abcd1234"
        
        assert valid_config.ANTHROPIC_API_KEY.startswith("sk-ant-")
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-api03-test123'})
    def test_api_key_from_environment(self):
        """Test API key loading from environment"""
        test_config = Config()
        test_config.reload()  # Reload to pick up patched environment
        
        assert test_config.ANTHROPIC_API_KEY == 'sk-ant-api03-test123'
        assert test_config.ANTHROPIC_API_KEY.startswith('sk-ant-')


class TestSystemIntegration:
    """Test system-level integration points"""
    
    def test_all_modules_importable(self):
        """Test that all main modules can be imported"""
        modules_to_test = [
            'config',
            'models', 
            'vector_store',
            'document_processor',
            'search_tools',
            'ai_generator',
            'session_manager',
            'rag_system',
            'app'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_critical_dependencies_available(self):
        """Test that all critical dependencies are available"""
        critical_deps = [
            'chromadb',
            'sentence_transformers', 
            'anthropic',
            'fastapi',
            'pydantic',
            'uvicorn'
        ]
        
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                pytest.fail(f"Critical dependency {dep} is not available")
    
    def test_config_values_reasonable(self):
        """Test that configuration values are reasonable"""
        test_config = Config()
        
        # Test chunk size is reasonable (not too small or large)
        assert 100 <= test_config.CHUNK_SIZE <= 2000
        
        # Test overlap is smaller than chunk size
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
        
        # Test max results is reasonable
        assert 1 <= test_config.MAX_RESULTS <= 20
        
        # Test max history is reasonable
        assert 1 <= test_config.MAX_HISTORY <= 10
        
        # Test model names are non-empty
        assert len(test_config.ANTHROPIC_MODEL) > 0
        assert len(test_config.EMBEDDING_MODEL) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])