import pytest
import tempfile
import numpy as np
from pathlib import Path
import cv2
import time


# try:
#     from genocr import GenOCR, process_image, check_installation, __version__
#     from genocr.core import process_batch
#     GENOCR_AVAILABLE = True
# except ImportError:
#     GENOCR_AVAILABLE = False
#     pytest.skip("GenOCR not installed", allow_module_level=True)

from genocr import GenOCR, process_image, __version__
from genocr.core import process_batch, check_installation
GENOCR_AVAILABLE = True

@pytest.fixture
def sample_text_image():
    """Create a more realistic test image with clear text"""
    # Create a white background with black text - easier for OCR
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    
    # Add some text that should be easily readable
    cv2.putText(img, 'Hello GenOCR World!', (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, 'Testing OCR Pipeline', (50, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    return img


@pytest.fixture
def document_image():
    """Create a document-like image for testing"""
    img = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(img, 'DOCUMENT TITLE', (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Body text
    lines = [
        'This is a test document for GenOCR.',
        'It contains multiple lines of text.',
        'The AI should correct any OCR errors.',
        'Testing multi-language support: Café, résumé, naïve.'
    ]
    
    for i, line in enumerate(lines):
        y_pos = 120 + i * 40
        cv2.putText(img, line, (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add some shapes that YOLO might detect
    cv2.rectangle(img, (650, 50), (750, 150), (0, 0, 0), 2)  # Rectangle
    cv2.circle(img, (700, 250), 50, (0, 0, 0), 2)  # Circle
    
    return img


@pytest.fixture
def sample_image_file(sample_text_image):
    """Save sample image to temporary file"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, sample_text_image)
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def document_image_file(document_image):
    """Save document image to temporary file"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, document_image)
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def models_dir():
    """Path to models directory"""
    # From tests/python_bindings/, go up 2 levels to project root, then to models
    return Path(__file__).parent.parent.parent / "models"


@pytest.fixture(scope="session")
def ocr_instance(models_dir):
    """Create a single OCR instance for the test session to avoid reinitializing"""
    if not models_dir.exists():
        pytest.skip(f"Models directory not found: {models_dir}")
    
    print(f"\nInitializing GenOCR (this takes ~50s)...")
    start_time = time.time()
    ocr = GenOCR(str(models_dir))
    init_time = time.time() - start_time
    print(f"GenOCR initialized in {init_time:.1f}s")
    return ocr


class TestGenOCRInstallation:
    """Test basic installation and imports"""
    
    def test_module_imports(self):
        """Test that all expected functions can be imported"""
        assert GENOCR_AVAILABLE
        assert callable(GenOCR)
        assert callable(process_image)
        assert callable(process_batch)
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"
    
    def test_installation_check(self):
        """Test that GenOCR installation check works"""
        # This should not require models
        result = check_installation()
        assert isinstance(result, bool)


class TestGenOCRInitialization:
    """Test GenOCR initialization"""
    
    def test_init_invalid_models(self):
        """Test GenOCR initialization with invalid models directory"""
        with pytest.raises(FileNotFoundError):
            GenOCR("/nonexistent/path")
    
    def test_init_valid_models(self, models_dir):
        """Test GenOCR initialization with valid models directory"""
        if not models_dir.exists():
            pytest.skip(f"Models directory not found: {models_dir}")
        
        # This test will take ~50s due to model loading
        ocr = GenOCR(str(models_dir))
        assert ocr.models_dir == models_dir.resolve()
        assert not ocr.use_cuda  # Default should be False


class TestGenOCRProcessing:
    """Test GenOCR processing functionality"""
    
    @pytest.mark.timeout(300)  # 5 minute timeout for processing
    def test_process_image_file(self, ocr_instance, sample_image_file):
        """Test processing image file"""
        result = ocr_instance.process(sample_image_file)
        
        # Validate result format based on genocr.cpp output
        assert isinstance(result, str)
        assert len(result) > 0
        assert "=== OCR Results for" in result
        assert "--- Raw OCR Output ---" in result
        assert "--- Final Corrected Text ---" in result
        
        # Should contain some form of the text we put in
        result_lower = result.lower()
        assert any(word in result_lower for word in ["hello", "world", "genocr"])
    
    @pytest.mark.timeout(300)
    def test_process_numpy_array(self, ocr_instance, sample_text_image):
        """Test processing numpy array"""
        result = ocr_instance.process(sample_text_image)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "=== OCR Results for" in result
        
        # Should detect some text
        result_lower = result.lower()
        assert any(word in result_lower for word in ["hello", "world", "genocr", "testing"])
    
    @pytest.mark.timeout(300)
    def test_process_document(self, ocr_instance, document_image_file):
        """Test processing document-like image"""
        result = ocr_instance.process(document_image_file)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check for expected output sections
        assert "=== OCR Results for" in result
        assert "--- Raw OCR Output ---" in result
        assert "--- Final Corrected Text ---" in result
        
        # Should detect document text
        result_lower = result.lower()
        assert any(word in result_lower for word in ["document", "title", "test", "text"])
    
    def test_process_invalid_input(self, ocr_instance):
        """Test processing with invalid input"""
        with pytest.raises(ValueError):
            ocr_instance.process(123)  # Invalid input type
        
        with pytest.raises(ValueError):
            ocr_instance.process(None)
    
    def test_process_nonexistent_file(self, ocr_instance):
        """Test processing nonexistent file"""
        with pytest.raises(FileNotFoundError):
            ocr_instance.process("/nonexistent/image.png")
    
    def test_process_empty_image(self, ocr_instance):
        """Test processing empty/blank image"""
        # Create a blank white image
        blank_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        result = ocr_instance.process(blank_img)
        assert isinstance(result, str)
        # Should still return formatted output even if no text detected
        assert "=== OCR Results for" in result


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.timeout(300)
    def test_process_image_function(self, models_dir, sample_image_file):
        """Test convenience process_image function"""
        if not models_dir.exists():
            pytest.skip(f"Models directory not found: {models_dir}")
        
        result = process_image(sample_image_file, str(models_dir))
        assert isinstance(result, str)
        assert len(result) > 0
        assert "=== OCR Results for" in result
    
    @pytest.mark.timeout(600)  # Longer timeout for batch processing
    def test_process_batch_function(self, models_dir, sample_image_file):
        """Test convenience process_batch function"""
        if not models_dir.exists():
            pytest.skip(f"Models directory not found: {models_dir}")
        
        # Create a second test image
        img2 = np.ones((150, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img2, 'Batch Test Image', (50, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            cv2.imwrite(tmp2.name, img2)
            
            try:
                results = process_batch([sample_image_file, tmp2.name], str(models_dir))
                assert len(results) == 2
                assert all(isinstance(r, str) for r in results)
                assert all("=== OCR Results for" in r for r in results)
            finally:
                Path(tmp2.name).unlink(missing_ok=True)


class TestOutputFormat:
    """Test output format matches expected structure from genocr.cpp"""
    
    @pytest.mark.timeout(300)
    def test_output_sections(self, ocr_instance, document_image_file):
        """Test that output contains all expected sections"""
        result = ocr_instance.process(document_image_file)
        
        # Check required sections based on genocr.cpp
        required_sections = [
            "=== OCR Results for",
            "--- Raw OCR Output ---", 
            "--- AI Correction Process ---",
            "--- Final Corrected Text ---"
        ]
        
        for section in required_sections:
            assert section in result, f"Missing section: {section}"
    
    @pytest.mark.timeout(300)
    def test_context_detection(self, ocr_instance, document_image_file):
        """Test that YOLO context detection works"""
        result = ocr_instance.process(document_image_file)
        
        # If objects are detected, should see "Detected Context Objects:" section
        # This is optional since it depends on what YOLO detects
        if "Detected Context Objects:" in result:
            # Should have some detected objects listed
            lines = result.split('\n')
            context_section = False
            for line in lines:
                if "Detected Context Objects:" in line:
                    context_section = True
                    continue
                if context_section and line.startswith('- '):
                    # Found at least one detected object
                    break
    
    @pytest.mark.timeout(300)
    def test_raw_ocr_output(self, ocr_instance, sample_image_file):
        """Test that raw OCR output is captured"""
        result = ocr_instance.process(sample_image_file)
        
        # Should have raw OCR lines
        assert "RAW LINE:" in result
        
        # Should have some text between raw and corrected sections
        raw_start = result.find("--- Raw OCR Output ---")
        ai_start = result.find("--- AI Correction Process ---")
        
        assert raw_start < ai_start
        raw_section = result[raw_start:ai_start]
        assert len(raw_section.strip()) > len("--- Raw OCR Output ---")


@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics"""
    
    def test_initialization_time(self, models_dir):
        """Test that initialization completes within reasonable time"""
        if not models_dir.exists():
            pytest.skip(f"Models directory not found: {models_dir}")
        
        start_time = time.time()
        ocr = GenOCR(str(models_dir))
        init_time = time.time() - start_time
        
        # Should initialize within 120 seconds (allowing some buffer)
        assert init_time < 120, f"Initialization took {init_time:.1f}s, too slow!"
        print(f"Initialization completed in {init_time:.1f}s")
    
    @pytest.mark.timeout(60)
    def test_processing_speed(self, ocr_instance, sample_image_file):
        """Test that processing completes in reasonable time after initialization"""
        start_time = time.time()
        result = ocr_instance.process(sample_image_file)
        process_time = time.time() - start_time
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # After initialization, processing should be much faster
        assert process_time < 60, f"Processing took {process_time:.1f}s, too slow!"
        print(f"Processing completed in {process_time:.1f}s")


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (may take minutes)")
    config.addinivalue_line("markers", "timeout: set timeout for tests")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
