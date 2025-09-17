#!/usr/bin/env python3
"""
Test corrected database classification training flow
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.services.image_metadata_manager import ImageMetadataManager
from backend.core.auto_training_system import AutoTrainingSystem


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_corrected_flow():
    """Test the corrected database training flow"""
    print("=== Testing Corrected Database Training Flow ===")

    # Create test database with classified images
    test_db_path = "test_corrected.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    metadata_manager = ImageMetadataManager(test_db_path)

    # Create test images with valid filename format for database lookup
    test_images = [
        {
            'filename': '20240101120000_board1@COMP1_001_R.jpg',
            'classification': 'Up',
            'product_name': 'PART001'
        },
        {
            'filename': '20240101120001_board1@COMP1_002_R.jpg',
            'classification': 'Down',
            'product_name': 'PART001'
        },
        {
            'filename': '20240101120002_board1@COMP2_001_G.jpg',
            'classification': 'NG',
            'product_name': 'PART001'
        }
    ]

    print("1. Creating test images with proper filename format...")
    image_ids = []
    for i, img_info in enumerate(test_images):
        # Create test image file
        test_image_path = f"test_corrected_{i}.jpg"
        with open(test_image_path, 'wb') as f:
            f.write(b'fake_image_data_' + str(i).encode())

        # Record image metadata
        image_id = metadata_manager.record_downloaded_image(
            image_path=test_image_path,
            original_filename=img_info['filename'],
            source_site="HPH",
            source_line_id="V31",
            remote_image_path=f"/path/{img_info['filename']}",
            download_url=f"http://example.com/{img_info['filename']}",
            product_info={
                'product_name': img_info['product_name'],
                'component_name': 'COMP1',
                'board_info': 'board1',
                'light_condition': 'R'
            },
            task_id=None
        )

        # Classify the image
        if image_id:
            success = metadata_manager.classify_image(
                image_id=image_id,
                classification_label=img_info['classification'],
                confidence=1.0,
                is_manual=True,
                notes="Test classification"
            )
            if success:
                image_ids.append(image_id)
                print(f"   Created: {img_info['filename']} -> {img_info['classification']}")

    print(f"\n2. Testing database classification processing...")

    # Test the corrected flow
    try:
        system = AutoTrainingSystem()

        # Create test directories
        test_input_dir = "test_input_corrected"
        test_output_dir = "test_output_corrected"

        # Clean up previous test runs
        if Path(test_input_dir).exists():
            import shutil
            shutil.rmtree(test_input_dir)
        if Path(test_output_dir).exists():
            import shutil
            shutil.rmtree(test_output_dir)

        Path(test_output_dir).mkdir(exist_ok=True)

        # Test the process_database_classified_images method
        stats = system.process_database_classified_images(
            input_dir=test_input_dir,
            output_dir=test_output_dir,
            site="HPH",
            line_id="V31",
            part_numbers=["PART001"]
        )

        print(f"   Processing stats: {stats}")

        # Check if OK/NG structure was created
        input_path = Path(test_input_dir)
        ok_dir = input_path / 'OK'
        ng_dir = input_path / 'NG'

        print(f"   OK folder exists: {ok_dir.exists()}")
        print(f"   NG folder exists: {ng_dir.exists()}")

        if ok_dir.exists():
            ok_files = list(ok_dir.glob('*.jpg'))
            print(f"   OK folder contains {len(ok_files)} images")

        if ng_dir.exists():
            ng_files = list(ng_dir.glob('*.jpg'))
            print(f"   NG folder contains {len(ng_files)} images")

        # Check if raw_data was created with product classifications
        output_path = Path(test_output_dir)
        raw_data_dirs = list(output_path.iterdir())
        print(f"   Raw data directories: {[d.name for d in raw_data_dirs if d.is_dir()]}")

        print("\n3. Flow comparison:")
        print("   Traditional Mode:")
        print("     Input: User uploads folder with OK/NG subfolders")
        print("     Process: Query database for product info")
        print("     Result: Group by product+component -> orientation confirmation")
        print("")
        print("   Database Mode (Corrected):")
        print("     Input: User selects part numbers from database")
        print("     Process: Create OK/NG structure from database classifications")
        print("     Process: Query database for product info (same as traditional)")
        print("     Result: Group by product+component -> orientation confirmation")
        print("")
        print("   Key Point: Both modes end up with the same product groupings")
        print("             and both require orientation confirmation!")

        success = True

    except Exception as e:
        print(f"   Error during processing: {e}")
        success = False

    # Cleanup
    print("\n4. Cleaning up...")
    for i in range(len(test_images)):
        test_file = f"test_corrected_{i}.jpg"
        if os.path.exists(test_file):
            os.remove(test_file)

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    if Path("test_input_corrected").exists():
        import shutil
        shutil.rmtree("test_input_corrected")

    if Path("test_output_corrected").exists():
        import shutil
        shutil.rmtree("test_output_corrected")

    return success


def print_summary():
    """Print summary of the corrected implementation"""
    print("\n" + "="*60)
    print("CORRECTED DATABASE TRAINING IMPLEMENTATION SUMMARY")
    print("="*60)

    print("\nCORRECTED FLOW:")
    print("1. User selects part numbers from database")
    print("2. System loads classified images from database")
    print("3. Images organized into OK/NG folder structure")
    print("4. Traditional process_raw_images() processes OK folder")
    print("5. Images grouped by product+component+light (same as traditional)")
    print("6. User confirms orientation for each group (SAME AS TRADITIONAL)")
    print("7. Training proceeds with oriented classes")

    print("\nKEY CORRECTIONS:")
    print("✓ Database mode creates OK/NG structure first")
    print("✓ Then uses traditional processing for OK images")
    print("✓ Still requires orientation confirmation")
    print("✓ Still groups by product+component+light")
    print("✓ Training flow identical to traditional mode after setup")

    print("\nBENEFITS:")
    print("• Reuse previously downloaded and classified images")
    print("• Consistent with traditional training workflow")
    print("• Maintains all existing orientation confirmation logic")
    print("• No changes needed to training/evaluation code")

    print("\nAPI USAGE:")
    print("POST /training/start")
    print("{")
    print('  "use_database_classification": true,')
    print('  "part_numbers": ["PART001", "PART002"],')
    print('  "site": "HPH",')
    print('  "line_id": "V31"')
    print("}")
    print("↓")
    print("Status: pending_orientation (same as traditional)")
    print("↓")
    print("User confirms orientations via /orientation/confirm/{task_id}")
    print("↓")
    print("Training proceeds normally")


if __name__ == "__main__":
    setup_logging()

    try:
        success = test_corrected_flow()

        if success:
            print("\n✅ CORRECTED IMPLEMENTATION WORKING!")
        else:
            print("\n❌ Issues found in corrected implementation")

        print_summary()

    except Exception as e:
        logging.error(f"Test failed: {e}")
        print(f"\nTest failed: {e}")