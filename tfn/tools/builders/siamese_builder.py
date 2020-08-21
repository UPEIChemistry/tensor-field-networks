from tensorflow.python.keras import Input

from tfn.tools.builders.classifier_builder import ClassifierMixIn
from tfn.tools.builders.multi_trunk_builder import DualTrunkBuilder


class SiameseBuilder(DualTrunkBuilder, ClassifierMixIn):
    def get_inputs(self):
        return [
            Input([2, self.num_points,], name="atomic_nums", dtype="int32"),
            Input([2, self.num_points, 3], name="cartesians", dtype="float32"),
        ]

    def get_learned_output(self, inputs: list):
        z, c = inputs
        point_clouds = [
            self.point_cloud_layer([a, b])
            for a, b in zip(
                [z[:, 0], z[:, 1]], [c[:, 0], c[:, 1]]  # Split z, c into 4 arrays
            )
        ]
        inputs = self.get_dual_trunks(point_clouds)
        return point_clouds, inputs

    def get_model_output(self, point_cloud: list, inputs: list):
        one_hot, output = self.mix_dual_trunks(point_cloud, inputs)
        return self.average_votes(output)
