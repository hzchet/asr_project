from src.tests.test_config import TestConfig
from src.tests.test_dataloader import TestDataloader
from src.tests.test_datasets import TestDataset
from src.tests.test_text_encoder import TestTextEncoder


if __name__ == '__main__':
    test_config = TestConfig()
    test_config.test_create()
    
    test_dataloader = TestDataloader()
    test_dataloader.test_collate_fn()
    test_dataloader.test_dataloaders()
    
    test_dataset = TestDataset()
    test_dataset.test_custom_dataset()
    test_dataset.test_custom_dir_dataset()
    test_dataset.test_librispeech()
    
    test_text_encoder = TestTextEncoder()
    test_text_encoder.test_ctc_decode()
    test_text_encoder.test_beam_search()
