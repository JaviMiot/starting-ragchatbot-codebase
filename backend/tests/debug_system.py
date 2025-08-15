#!/usr/bin/env python3
"""
Comprehensive diagnostic script for RAG chatbot system.
This script tests each component in isolation to identify failure points.
"""

import sys
import os
import traceback
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results"""
    component: str
    test_name: str
    success: bool
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SystemDiagnostics:
    """Comprehensive system diagnostics for RAG chatbot"""
    
    def __init__(self):
        self.results: List[DiagnosticResult] = []
        self.failures: List[DiagnosticResult] = []
    
    def add_result(self, component: str, test_name: str, success: bool, 
                   error_message: str = None, details: Dict[str, Any] = None):
        """Add a diagnostic result"""
        result = DiagnosticResult(component, test_name, success, error_message, details)
        self.results.append(result)
        if not success:
            self.failures.append(result)
    
    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        print("🔍 Starting comprehensive system diagnostics...\n")
        
        # Test in order of dependency
        self.test_environment()
        self.test_configuration()
        self.test_dependencies()
        self.test_file_access()
        self.test_database_connectivity() 
        self.test_vector_store()
        self.test_document_processing()
        self.test_search_tools()
        self.test_ai_generator()
        self.test_rag_system()
        self.test_api_integration()
        
        self.print_summary()
    
    def test_environment(self):
        """Test environment setup"""
        print("🌍 Testing environment setup...")
        
        # Test Python version
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                self.add_result("Environment", "Python Version", True, 
                              details={"version": f"{version.major}.{version.minor}.{version.micro}"})
            else:
                self.add_result("Environment", "Python Version", False, 
                              f"Python {version.major}.{version.minor} too old, need 3.8+")
        except Exception as e:
            self.add_result("Environment", "Python Version", False, str(e))
        
        # Test environment variables
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if api_key:
                masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "SET"
                self.add_result("Environment", "API Key", True, 
                              details={"key_format": masked_key})
            else:
                self.add_result("Environment", "API Key", False, "ANTHROPIC_API_KEY not set")
        except Exception as e:
            self.add_result("Environment", "API Key", False, str(e))
    
    def test_configuration(self):
        """Test configuration loading"""
        print("⚙️  Testing configuration...")
        
        try:
            from config import config
            
            # Test config object exists
            if config:
                self.add_result("Configuration", "Config Loading", True)
            else:
                self.add_result("Configuration", "Config Loading", False, "Config object is None")
            
            # Test key configuration values
            required_attrs = ['ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL', 'EMBEDDING_MODEL', 
                            'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_RESULTS', 'CHROMA_PATH']
            
            missing_attrs = []
            for attr in required_attrs:
                if not hasattr(config, attr):
                    missing_attrs.append(attr)
            
            if not missing_attrs:
                self.add_result("Configuration", "Required Attributes", True,
                              details={"attributes_checked": len(required_attrs)})
            else:
                self.add_result("Configuration", "Required Attributes", False, 
                              f"Missing attributes: {missing_attrs}")
            
            # Test reasonable values
            try:
                if 100 <= config.CHUNK_SIZE <= 2000:
                    self.add_result("Configuration", "Chunk Size", True, 
                                  details={"chunk_size": config.CHUNK_SIZE})
                else:
                    self.add_result("Configuration", "Chunk Size", False, 
                                  f"Chunk size {config.CHUNK_SIZE} not reasonable (100-2000)")
            except Exception as e:
                self.add_result("Configuration", "Chunk Size", False, str(e))
                
        except Exception as e:
            self.add_result("Configuration", "Config Loading", False, str(e))
    
    def test_dependencies(self):
        """Test dependency imports"""
        print("📦 Testing dependencies...")
        
        dependencies = [
            'chromadb',
            'sentence_transformers', 
            'anthropic',
            'fastapi',
            'pydantic',
            'uvicorn'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep)
                self.add_result("Dependencies", f"Import {dep}", True)
            except ImportError as e:
                self.add_result("Dependencies", f"Import {dep}", False, str(e))
            except Exception as e:
                self.add_result("Dependencies", f"Import {dep}", False, f"Unexpected error: {str(e)}")
    
    def test_file_access(self):
        """Test file system access"""
        print("📁 Testing file access...")
        
        # Test docs folder
        docs_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
        try:
            if os.path.exists(docs_path):
                files = os.listdir(docs_path)
                txt_files = [f for f in files if f.endswith('.txt')]
                self.add_result("File Access", "Docs Folder", True,
                              details={"files_found": len(files), "txt_files": len(txt_files)})
                
                # Test reading a sample file
                if txt_files:
                    sample_file = os.path.join(docs_path, txt_files[0])
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.add_result("File Access", "Sample File Read", True,
                                      details={"file": txt_files[0], "content_length": len(content)})
            else:
                self.add_result("File Access", "Docs Folder", False, "Docs folder does not exist")
        except Exception as e:
            self.add_result("File Access", "Docs Folder", False, str(e))
        
        # Test ChromaDB directory writability
        try:
            chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
            parent_dir = os.path.dirname(chroma_path)
            
            if os.access(parent_dir, os.W_OK):
                self.add_result("File Access", "ChromaDB Directory", True)
            else:
                self.add_result("File Access", "ChromaDB Directory", False, 
                              "Parent directory not writable")
        except Exception as e:
            self.add_result("File Access", "ChromaDB Directory", False, str(e))
    
    def test_database_connectivity(self):
        """Test ChromaDB connectivity"""
        print("🗄️  Testing database connectivity...")
        
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Test ChromaDB client creation
            import chromadb
            
            client = chromadb.PersistentClient(
                path=os.path.join(temp_dir, "test_chroma"),
                settings=chromadb.config.Settings(anonymized_telemetry=False)
            )
            
            self.add_result("Database", "Client Creation", True)
            
            # Test embedding function
            embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self.add_result("Database", "Embedding Function", True)
            
            # Test collection creation
            collection = client.get_or_create_collection(
                name="test_collection",
                embedding_function=embedding_fn
            )
            
            self.add_result("Database", "Collection Creation", True)
            
            # Test basic operations
            collection.add(
                documents=["Test document for diagnostics"],
                metadatas=[{"test": "metadata"}],
                ids=["test_id"]
            )
            
            results = collection.query(
                query_texts=["Test query"],
                n_results=1
            )
            
            if results and results.get('documents'):
                self.add_result("Database", "Basic Operations", True,
                              details={"results_found": len(results['documents'][0])})
            else:
                self.add_result("Database", "Basic Operations", False, "No results returned")
                
        except Exception as e:
            self.add_result("Database", "ChromaDB Test", False, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_vector_store(self):
        """Test vector store functionality"""
        print("🔍 Testing vector store...")
        
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            from vector_store import VectorStore
            from models import Course, CourseChunk, Lesson
            
            # Create vector store
            vector_store = VectorStore(
                chroma_path=os.path.join(temp_dir, "test_chroma"),
                embedding_model="all-MiniLM-L6-v2",
                max_results=5
            )
            
            self.add_result("Vector Store", "Initialization", True)
            
            # Test adding course metadata
            test_course = Course(
                title="Test Course",
                course_link="http://test.com",
                instructor="Test Instructor",
                lessons=[Lesson(lesson_number=1, title="Test Lesson")]
            )
            
            vector_store.add_course_metadata(test_course)
            self.add_result("Vector Store", "Add Course Metadata", True)
            
            # Test adding course content
            test_chunks = [
                CourseChunk(
                    content="This is test content for the diagnostic system",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0
                )
            ]
            
            vector_store.add_course_content(test_chunks)
            self.add_result("Vector Store", "Add Course Content", True)
            
            # Test search functionality
            results = vector_store.search("test content")
            
            if not results.error and not results.is_empty():
                self.add_result("Vector Store", "Search Functionality", True,
                              details={"results_found": len(results.documents)})
            else:
                error_msg = results.error if results.error else "No results found"
                self.add_result("Vector Store", "Search Functionality", False, error_msg)
            
            # Test course name resolution
            resolved = vector_store._resolve_course_name("Test")
            if resolved == "Test Course":
                self.add_result("Vector Store", "Course Name Resolution", True)
            else:
                self.add_result("Vector Store", "Course Name Resolution", False,
                              f"Expected 'Test Course', got '{resolved}'")
                
        except Exception as e:
            self.add_result("Vector Store", "Vector Store Test", False, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_document_processing(self):
        """Test document processing functionality"""
        print("📄 Testing document processing...")
        
        try:
            from document_processor import DocumentProcessor
            
            processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
            self.add_result("Document Processing", "Initialization", True)
            
            # Test with sample content
            sample_content = """Course Title: Test Course
Course Link: http://test.com
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: http://lesson.com
This is the introduction lesson content. It covers basic concepts and provides an overview of the course material that students will learn.

Lesson 1: Advanced Topics
This lesson covers more advanced topics in the field."""
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_content)
                temp_file = f.name
            
            try:
                course, chunks = processor.process_course_document(temp_file)
                
                if course and course.title == "Test Course":
                    self.add_result("Document Processing", "Course Parsing", True,
                                  details={"course_title": course.title, "lessons": len(course.lessons)})
                else:
                    self.add_result("Document Processing", "Course Parsing", False,
                                  "Course not parsed correctly")
                
                if chunks and len(chunks) > 0:
                    self.add_result("Document Processing", "Content Chunking", True,
                                  details={"chunks_created": len(chunks)})
                else:
                    self.add_result("Document Processing", "Content Chunking", False,
                                  "No chunks created")
                    
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            self.add_result("Document Processing", "Document Processing Test", False, str(e))
    
    def test_search_tools(self):
        """Test search tools functionality"""
        print("🔧 Testing search tools...")
        
        try:
            from search_tools import CourseSearchTool, ToolManager
            from vector_store import SearchResults
            from unittest.mock import Mock
            
            # Create mock vector store
            mock_vector_store = Mock()
            mock_results = SearchResults(
                documents=["Test search result"],
                metadata=[{"course_title": "Test Course", "lesson_number": 1}],
                distances=[0.1],
                error=None
            )
            mock_vector_store.search.return_value = mock_results
            mock_vector_store.get_lesson_link.return_value = "http://lesson.com"
            
            # Test CourseSearchTool
            search_tool = CourseSearchTool(mock_vector_store)
            tool_def = search_tool.get_tool_definition()
            
            if tool_def and tool_def.get("name") == "search_course_content":
                self.add_result("Search Tools", "Tool Definition", True)
            else:
                self.add_result("Search Tools", "Tool Definition", False,
                              "Tool definition not correct")
            
            # Test tool execution
            result = search_tool.execute("test query", "Test Course")
            
            if isinstance(result, str) and "Test Course" in result:
                self.add_result("Search Tools", "Tool Execution", True,
                              details={"result_length": len(result)})
            else:
                self.add_result("Search Tools", "Tool Execution", False,
                              f"Unexpected result: {result}")
            
            # Test ToolManager
            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)
            
            definitions = tool_manager.get_tool_definitions()
            if len(definitions) == 1:
                self.add_result("Search Tools", "Tool Manager", True)
            else:
                self.add_result("Search Tools", "Tool Manager", False,
                              f"Expected 1 tool, got {len(definitions)}")
                
        except Exception as e:
            self.add_result("Search Tools", "Search Tools Test", False, str(e))
    
    def test_ai_generator(self):
        """Test AI generator functionality"""
        print("🤖 Testing AI generator...")
        
        try:
            from ai_generator import AIGenerator
            from unittest.mock import Mock, patch
            
            # Test with mock API key
            with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client
                
                ai_gen = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
                self.add_result("AI Generator", "Initialization", True)
                
                # Test system prompt
                if hasattr(ai_gen, 'SYSTEM_PROMPT') and len(ai_gen.SYSTEM_PROMPT) > 0:
                    self.add_result("AI Generator", "System Prompt", True,
                                  details={"prompt_length": len(ai_gen.SYSTEM_PROMPT)})
                else:
                    self.add_result("AI Generator", "System Prompt", False,
                                  "System prompt not found or empty")
                
                # Test base parameters
                if hasattr(ai_gen, 'base_params') and ai_gen.base_params.get('model'):
                    self.add_result("AI Generator", "Base Parameters", True)
                else:
                    self.add_result("AI Generator", "Base Parameters", False,
                                  "Base parameters not set correctly")
                    
        except Exception as e:
            self.add_result("AI Generator", "AI Generator Test", False, str(e))
    
    def test_rag_system(self):
        """Test RAG system integration"""
        print("🧠 Testing RAG system...")
        
        try:
            from rag_system import RAGSystem
            from config import Config
            from unittest.mock import Mock, patch
            
            # Create mock config
            mock_config = Mock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.CHROMA_PATH = "./test_chroma"
            mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock_config.MAX_RESULTS = 5
            mock_config.ANTHROPIC_API_KEY = "test_key"
            mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            mock_config.MAX_HISTORY = 2
            
            # Mock all components
            with patch('rag_system.DocumentProcessor'):
                with patch('rag_system.VectorStore'):
                    with patch('rag_system.AIGenerator'):
                        with patch('rag_system.SessionManager'):
                            rag_system = RAGSystem(mock_config)
                            
                            self.add_result("RAG System", "Initialization", True)
                            
                            # Test tool manager setup
                            if hasattr(rag_system, 'tool_manager') and rag_system.tool_manager:
                                definitions = rag_system.tool_manager.get_tool_definitions()
                                if len(definitions) >= 2:  # search and outline tools
                                    self.add_result("RAG System", "Tool Setup", True,
                                                  details={"tools_registered": len(definitions)})
                                else:
                                    self.add_result("RAG System", "Tool Setup", False,
                                                  f"Only {len(definitions)} tools registered")
                            else:
                                self.add_result("RAG System", "Tool Setup", False,
                                              "Tool manager not initialized")
                                
        except Exception as e:
            self.add_result("RAG System", "RAG System Test", False, str(e))
    
    def test_api_integration(self):
        """Test API integration points"""
        print("🌐 Testing API integration...")
        
        try:
            from app import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            self.add_result("API", "FastAPI App Creation", True)
            
            # Test health endpoint (if it exists) or basic structure
            if hasattr(app, 'routes') and len(app.routes) > 0:
                self.add_result("API", "Routes Configured", True,
                              details={"routes_count": len(app.routes)})
            else:
                self.add_result("API", "Routes Configured", False,
                              "No routes found")
                
        except Exception as e:
            self.add_result("API", "API Integration Test", False, str(e))
    
    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "="*60)
        print("🔍 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Count results by component
        component_stats = {}
        for result in self.results:
            comp = result.component
            if comp not in component_stats:
                component_stats[comp] = {"total": 0, "passed": 0, "failed": 0}
            
            component_stats[comp]["total"] += 1
            if result.success:
                component_stats[comp]["passed"] += 1
            else:
                component_stats[comp]["failed"] += 1
        
        # Print component summary
        print("\n📊 Component Status:")
        for comp, stats in component_stats.items():
            status = "✅" if stats["failed"] == 0 else "❌" if stats["passed"] == 0 else "⚠️ "
            print(f"{status} {comp}: {stats['passed']}/{stats['total']} passed")
        
        # Print failures in detail
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}):")
            print("-" * 40)
            
            for failure in self.failures:
                print(f"\n🔴 {failure.component} - {failure.test_name}")
                print(f"   Error: {failure.error_message}")
                if failure.details:
                    print(f"   Details: {failure.details}")
        else:
            print("\n✅ ALL TESTS PASSED!")
        
        # Overall status
        total_tests = len(self.results)
        passed_tests = total_tests - len(self.failures)
        
        print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if self.failures:
            print("\n💡 RECOMMENDATIONS:")
            self._print_recommendations()
    
    def _print_recommendations(self):
        """Print specific recommendations based on failures"""
        failure_types = [f.component for f in self.failures]
        
        if "Environment" in failure_types:
            print("   • Check that ANTHROPIC_API_KEY is set in .env file")
            print("   • Verify Python version is 3.8 or higher")
        
        if "Configuration" in failure_types:
            print("   • Check config.py file exists and is valid")
            print("   • Verify all required configuration values are set")
        
        if "Dependencies" in failure_types:
            print("   • Run: uv sync to install missing dependencies")
            print("   • Check that all required packages are in pyproject.toml")
        
        if "Database" in failure_types:
            print("   • Check ChromaDB installation and compatibility")
            print("   • Verify file system permissions for database directory")
            print("   • Try clearing ChromaDB directory and restarting")
        
        if "Vector Store" in failure_types:
            print("   • Check embedding model download and initialization")
            print("   • Verify ChromaDB collections can be created")
        
        if "Document Processing" in failure_types:
            print("   • Check document format in docs/ folder")
            print("   • Verify file encoding is UTF-8")
        
        if "Search Tools" in failure_types:
            print("   • Check vector store integration")
            print("   • Verify tool definitions are correct")
        
        if "AI Generator" in failure_types:
            print("   • Verify Anthropic API key is valid")
            print("   • Check network connectivity to Anthropic API")
        
        if "RAG System" in failure_types:
            print("   • Check integration between all components")
            print("   • Verify all dependencies are working individually")


def main():
    """Main diagnostic function"""
    diagnostics = SystemDiagnostics()
    diagnostics.run_all_diagnostics()
    
    # Return exit code based on results
    return 0 if not diagnostics.failures else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)