import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults data class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'meta1': 'value1'}, {'meta2': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'meta1': 'value1'}, {'meta2': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        error_msg = "Test error message"
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
    
    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        
        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test VectorStore functionality"""
    
    def setup_method(self):
        """Setup test fixtures with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, "test_chroma_db")
        self.embedding_model = "all-MiniLM-L6-v2"
        self.max_results = 5
        
        # Create mock ChromaDB client and collections
        self.mock_client = Mock()
        self.mock_course_catalog = Mock()
        self.mock_course_content = Mock()
        
        self.mock_client.get_or_create_collection.side_effect = [
            self.mock_course_catalog,
            self.mock_course_content,
            self.mock_course_catalog,  # For clear_all_data recreation
            self.mock_course_content   # For clear_all_data recreation
        ]
        
        with patch('vector_store.chromadb.PersistentClient') as mock_chroma:
            with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                mock_chroma.return_value = self.mock_client
                self.vector_store = VectorStore(
                    self.chroma_path, 
                    self.embedding_model, 
                    self.max_results
                )
    
    def teardown_method(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test VectorStore initialization"""
        assert self.vector_store.max_results == self.max_results
        assert self.vector_store.course_catalog == self.mock_course_catalog
        assert self.vector_store.course_content == self.mock_course_content
        
        # Verify collections were created
        assert self.mock_client.get_or_create_collection.call_count == 2
    
    def test_search_without_filters(self):
        """Test search without course or lesson filters"""
        # Mock successful search
        mock_chroma_results = {
            'documents': [['Test content 1', 'Test content 2']],
            'metadatas': [[{'course_title': 'Course A'}, {'course_title': 'Course B'}]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("test query")
        
        assert not results.is_empty()
        assert len(results.documents) == 2
        assert results.documents[0] == 'Test content 1'
        assert results.error is None
        
        # Verify search was called correctly
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
    
    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        # Mock course name resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        # Mock content search
        mock_chroma_results = {
            'documents': [['Course specific content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("test query", course_name="Test Course")
        
        assert not results.is_empty()
        assert len(results.documents) == 1
        assert results.documents[0] == 'Course specific content'
        
        # Verify course resolution was called
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["Test Course"],
            n_results=1
        )
        
        # Verify content search with filter
        expected_filter = {"course_title": "Test Course"}
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_chroma_results = {
            'documents': [['Lesson specific content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search("test query", lesson_number=3)
        
        assert not results.is_empty()
        assert len(results.documents) == 1
        
        # Verify filter was built correctly
        expected_filter = {"lesson_number": 3}
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_with_both_filters(self):
        """Test search with both course and lesson filters"""
        # Mock course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        mock_chroma_results = {
            'documents': [['Specific lesson content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
            'distances': [[0.1]]
        }
        self.mock_course_content.query.return_value = mock_chroma_results
        
        results = self.vector_store.search(
            "test query", 
            course_name="Test Course", 
            lesson_number=2
        )
        
        assert not results.is_empty()
        
        # Verify combined filter
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_course_not_found(self):
        """Test search with non-existent course name"""
        # Mock empty course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        results = self.vector_store.search("test query", course_name="Nonexistent Course")
        
        assert results.is_empty()
        assert "No course found matching 'Nonexistent Course'" in results.error
        
        # Verify content search was not called
        self.mock_course_content.query.assert_not_called()
    
    def test_search_with_exception(self):
        """Test search error handling"""
        self.mock_course_content.query.side_effect = Exception("ChromaDB error")
        
        results = self.vector_store.search("test query")
        
        assert results.is_empty()
        assert "Search error: ChromaDB error" in results.error
    
    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        self.mock_course_catalog.query.return_value = {
            'documents': [['Resolved Course']],
            'metadatas': [[{'title': 'Resolved Course'}]]
        }
        
        resolved = self.vector_store._resolve_course_name("partial name")
        
        assert resolved == "Resolved Course"
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["partial name"],
            n_results=1
        )
    
    def test_resolve_course_name_not_found(self):
        """Test course name resolution when no match found"""
        self.mock_course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        resolved = self.vector_store._resolve_course_name("nonexistent")
        
        assert resolved is None
    
    def test_build_filter_no_params(self):
        """Test filter building with no parameters"""
        filter_dict = self.vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self):
        """Test filter building with course only"""
        filter_dict = self.vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
    
    def test_build_filter_lesson_only(self):
        """Test filter building with lesson only"""
        filter_dict = self.vector_store._build_filter(None, 5)
        assert filter_dict == {"lesson_number": 5}
    
    def test_build_filter_both_params(self):
        """Test filter building with both parameters"""
        filter_dict = self.vector_store._build_filter("Test Course", 3)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 3}
            ]
        }
        assert filter_dict == expected
    
    def test_add_course_metadata(self):
        """Test adding course metadata"""
        # Create test course with lessons
        lessons = [
            Lesson(lesson_number=1, title="Intro", lesson_link="http://lesson1.com"),
            Lesson(lesson_number=2, title="Advanced", lesson_link="http://lesson2.com")
        ]
        course = Course(
            title="Test Course",
            course_link="http://course.com",
            instructor="Test Instructor",
            lessons=lessons
        )
        
        self.vector_store.add_course_metadata(course)
        
        # Verify catalog add was called
        self.mock_course_catalog.add.assert_called_once()
        call_args = self.mock_course_catalog.add.call_args[1]
        
        assert call_args["documents"] == ["Test Course"]
        assert call_args["ids"] == ["Test Course"]
        
        metadata = call_args["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "Test Instructor"
        assert metadata["course_link"] == "http://course.com"
        assert metadata["lesson_count"] == 2
        
        # Verify lessons_json contains lesson data
        import json
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 1
        assert lessons_data[0]["lesson_title"] == "Intro"
        assert lessons_data[0]["lesson_link"] == "http://lesson1.com"
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                content="Content chunk 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Content chunk 2",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        self.vector_store.add_course_content(chunks)
        
        # Verify content add was called
        self.mock_course_content.add.assert_called_once()
        call_args = self.mock_course_content.add.call_args[1]
        
        assert call_args["documents"] == ["Content chunk 1", "Content chunk 2"]
        assert call_args["ids"] == ["Test_Course_0", "Test_Course_1"]
        
        metadata = call_args["metadatas"]
        assert len(metadata) == 2
        assert metadata[0]["course_title"] == "Test Course"
        assert metadata[0]["lesson_number"] == 1
        assert metadata[0]["chunk_index"] == 0
    
    def test_add_course_content_empty_list(self):
        """Test adding empty course content list"""
        self.vector_store.add_course_content([])
        
        # Verify no call was made
        self.mock_course_content.add.assert_not_called()
    
    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        self.mock_course_catalog.get.return_value = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == ['Course A', 'Course B', 'Course C']
        self.mock_course_catalog.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self):
        """Test getting course titles when empty"""
        self.mock_course_catalog.get.return_value = {'ids': []}
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_course_count(self):
        """Test getting course count"""
        self.mock_course_catalog.get.return_value = {
            'ids': ['Course A', 'Course B']
        }
        
        count = self.vector_store.get_course_count()
        
        assert count == 2
    
    def test_get_lesson_link(self):
        """Test getting lesson link"""
        lessons_json = '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://lesson1.com"}]'
        
        self.mock_course_catalog.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        
        link = self.vector_store.get_lesson_link("Test Course", 1)
        
        assert link == "http://lesson1.com"
        self.mock_course_catalog.get.assert_called_once_with(ids=["Test Course"])
    
    def test_get_lesson_link_not_found(self):
        """Test getting lesson link when lesson not found"""
        lessons_json = '[{"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "http://lesson2.com"}]'
        
        self.mock_course_catalog.get.return_value = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        
        link = self.vector_store.get_lesson_link("Test Course", 1)
        
        assert link is None
    
    def test_clear_all_data(self):
        """Test clearing all data"""
        self.vector_store.clear_all_data()
        
        # Verify collections were deleted
        self.mock_client.delete_collection.assert_any_call("course_catalog")
        self.mock_client.delete_collection.assert_any_call("course_content")
        
        # Verify collections were recreated
        assert self.mock_client.get_or_create_collection.call_count >= 4  # 2 initial + 2 recreated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])