"""
MASTER PIPELINE EXECUTION SCRIPT
Complete end-to-end execution of the multimodal food label analysis framework
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse

# Setup logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Main pipeline executor for the food label analysis framework"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        self.timing = {}
        
    def validate_inputs(self) -> bool:
        """Validate that all required inputs exist"""
        logger.info("Validating inputs...")
        
        required_files = [
            ('ground_truth', self.config['ground_truth_path']),
        ]
        
        required_dirs = [
            ('raw_images', self.config['raw_images_dir']),
        ]
        
        all_valid = True
        
        for name, path in required_files:
            if not Path(path).exists():
                logger.error(f"Missing required file: {name} at {path}")
                all_valid = False
            else:
                logger.info(f"  ✓ Found {name}: {path}")
        
        for name, path in required_dirs:
            if not Path(path).exists():
                logger.error(f"Missing required directory: {name} at {path}")
                all_valid = False
            else:
                # Count files in directory
                files = list(Path(path).glob("*.jpg")) + list(Path(path).glob("*.JPG"))
                logger.info(f"  ✓ Found {name}: {path} ({len(files)} JPG files)")
        
        return all_valid
    
    def run_step_1_image_processing(self) -> bool:
        """Execute Step 1: Image Processing"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: IMAGE PROCESSING")
        logger.info("="*70)
        
        start_time = time.time()
        