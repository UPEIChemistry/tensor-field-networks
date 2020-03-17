import tensorflow as tf
from kerastuner import HyperModel, HyperParameters
from tfn.layers import RadialFactory

from tfn.tools.builders import Builder
from tfn.tools.radials import get_radial_factory
from tfn.tools.jobs.config_defaults import factory_config as fconf


class HyperFactory(HyperModel):
    def __init__(self,
                 builder_cls,
                 **kwargs):
        super().__init__(
            name=kwargs.pop('name', 'hyper_factory'),
            tunable=kwargs.pop('tunable', True)
        )
        self.builder_cls = builder_cls
        self.kwargs = kwargs
        self.hp = None

    def build(self, hp: HyperParameters):
        self.hp = hp
        radial_factory = self._initialize_radial_factory()
        self._initialize_builder(radial_factory)
        model = self.builder.get_model()
        self._compile_model(model)
        return model

    def make_hyperparameter(self, name, value):
        return self.kwargs.get(name, self.hp.Fixed(name, value))

    def _initialize_builder(self, radial_factory: RadialFactory):
        self.builder: Builder = self.builder_cls(
            max_z=self.kwargs.get('max_z'),
            num_atoms=self.kwargs.get('num_atoms'),
            mu=self.kwargs.get('mu', None),
            sigma=self.kwargs.get('sigma', None),
            trainable_offsets=self.make_hyperparameter('trainable_offsets',
                                                       fconf['trainable_offsets']),
            embedding_units=self.make_hyperparameter('embedding_units', fconf['embedding_units']),
            radial_factory=radial_factory,
            residual=self.make_hyperparameter('residual', fconf['residual']),
            num_layers=self.make_hyperparameter('model_num_layers', fconf['num_layers']),
            si_units=self.make_hyperparameter('si_units', fconf['si_units']),
            activation=self.make_hyperparameter('activation', fconf['activation']),
            dynamic=self.kwargs.get('dynamic', fconf['dynamic']),
            basis_type=self.make_hyperparameter('basis_type', fconf['basis_type']),
            basis_config=self.kwargs.get('basis_config', {
                'width': self.hp.Fixed('basis_width', fconf['basis_config']['width']),
                'spacing': self.hp.Fixed('basis_spacing', fconf['basis_config']['spacing']),
                'min_value': self.hp.Fixed('basis_min', fconf['basis_config']['min_value']),
                'max_value': self.hp.Fixed('basis_max', fconf['basis_config']['max_value'])
            }),
            num_final_si_layers=self.make_hyperparameter('num_final_si_layers',
                                                         fconf['num_final_si_layers']),
            final_si_units=self.make_hyperparameter('final_si_units',
                                                    fconf['final_si_units'])
        )
        return self.builder

    def _initialize_radial_factory(self):
        radial_kwargs = dict(
            num_layers=self.make_hyperparameter('radial_num_layers', 3),
            units=self.make_hyperparameter('radial_units', 32),
            kernel_lambda=self.kwargs.get('kernel_lambda', 0.),
            bias_lambda=self.kwargs.get('bias_lambda', 0.)
        )
        return get_radial_factory(
            self.kwargs.get(
                'radial_factory',
                fconf['radial_factory']
            ),
            radial_kwargs
        )

    def _compile_model(self, model):
        loss_weights = [self.hp.Fixed('{}_weight'.format(n), 1) for n in model.output_names] or None
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.make_hyperparameter('learning_rate', 1e-3),
                epsilon=1e-8
            ),
            loss=self.make_hyperparameter('loss', 'mae'),
            loss_weights=self.kwargs.get('loss_weights', loss_weights),
            run_eagerly=self.kwargs.get('run_eagerly', fconf['run_eagerly']),
            metrics=self.kwargs.get('metrics', None)
        )
        return model
