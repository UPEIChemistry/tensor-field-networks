from tfn import blocks


class TestPreprocessingBlock:

    def test_preprocessing_outputs_3_tensors(self, random_cartesians_and_z):

        r, z = random_cartesians_and_z
        pre_block = blocks.PreprocessingBlock(max_z=5,
                                              gaussian_config={
                                                  'width': 0.2,
                                                  'spacing': 0.2,
                                                  'min_value': -1.0,
                                                  'max_value': 15.0
                                       })
        outputs = pre_block([r, z])
        assert len(outputs) == 3
