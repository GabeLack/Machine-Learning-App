import unittest
import pandas as pd
from parameterized import parameterized
from keras.models import Sequential

from build_model import build_model_factory
from context import ModelContext

def invalid_neuron_layers():
    return [
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_float", 1.23),
        ("invalid_list", [64, 64]),
        ("invalid_dict", {'layer1': 64, 'layer2': 64}),
        ("invalid_none", None)
    ]

def invalid_tuple_neurons_layers():
    return [
        ("invalid_string_in_tuple", ('invalid',)),
        ("invalid_float_in_tuple", (1.23,)),
        ("invalid_list_in_tuple", ([64, 64],)),
        ("invalid_dict_in_tuple", ({'layer1': 64},)),
        ("invalid_none_in_tuple", (None,))
    ]

def invalid_dropout_layers():
    return [
        ("invalid_string", 'invalid'),
        ("invalid_int", 1),
        ("invalid_list", [0.2, 0.2]),
        ("invalid_dict", {'layer1': 0.2, 'layer2': 0.2}),
        ("invalid_none", None)
    ]

def invalid_tuple_dropout_layers():
    return [
        ("invalid_string_in_tuple", ('invalid',)),
        ("invalid_int_in_tuple", (1,)),
        ("invalid_list_in_tuple", ([0.2, 0.2],)),
        ("invalid_dict_in_tuple", ({'layer1': 0.2},)),
        ("invalid_none_in_tuple", (None,))
    ]

def invalid_activation():
    return [
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_list", ['relu', 'relu']),
        ("invalid_dict", {'layer1': 'relu', 'layer2': 'relu'})
    ]

def invalid_optimizer():
    return [
        ("invalid_string", 'invalid'),
        ("invalid_int", 123),
        ("invalid_list", ['invalid',]),
        ("invalid_tuple", ('invalid',)),
        ("invalid_dict", {'layer1': 'invalid', 'layer2': 'invalid'})
    ]

class TestBuildModel(unittest.TestCase):

    def test_build_model_linear(self):
        input_dim = 10
        output_activation = 'linear'
        output_dim = 1
        build_model = build_model_factory(input_dim, output_activation, output_dim)
        model = build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, output_dim))
        self.assertEqual(model.layers[-1].activation.__name__, 'linear')

    def test_build_model_sigmoid(self):
        input_dim = 10
        output_activation = 'sigmoid'
        output_dim = 1
        build_model = build_model_factory(input_dim, output_activation, output_dim)
        model = build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(model.layers[-1].activation.__name__, 'sigmoid')

    def test_build_model_softmax(self):
        input_dim = 10
        output_activation = 'softmax'
        output_dim = 3
        build_model = build_model_factory(input_dim, output_activation, output_dim)
        model = build_model()
        self.assertIsInstance(model, Sequential)
        self.assertEqual(model.output_shape, (None, output_dim))
        self.assertEqual(model.layers[-1].activation.__name__, 'softmax')

    def test_build_model_custom_params(self):
        input_dim = 10
        output_activation = 'linear'
        output_dim = 1
        build_model = build_model_factory(input_dim, output_activation, output_dim)
        model = build_model(neuron_layers=(64, 64, 64),
                            dropout_layers=(0.1, 0.2, 0.3),
                            activation='relu',
                            optimizer='adam')
        self.assertIsInstance(model, Sequential)
        # Check layer count
        self.assertEqual(len(model.layers), 7) # 3 hidden layers + 3 dropout layers + 1 output layer
        # Check general activation function on a dense layer
        self.assertEqual(model.layers[0].activation.__name__, 'relu')
        # Check optimizer
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')

    @parameterized.expand(invalid_neuron_layers())
    def test_build_model_invalid_neuron_layers(self, name, neuron_layers):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(neuron_layers=neuron_layers)

    @parameterized.expand(invalid_tuple_neurons_layers())
    def test_build_model_invalid_tuple_neurons_layers(self, name, neuron_layers):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(neuron_layers=neuron_layers)

    @parameterized.expand(invalid_dropout_layers())
    def test_build_model_invalid_dropout_layers(self, name, dropout_layers):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(dropout_layers=dropout_layers)

    @parameterized.expand(invalid_tuple_dropout_layers())
    def test_build_model_invalid_tuple_dropout_layers(self, name, dropout_layers):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(dropout_layers=dropout_layers)

    @parameterized.expand(invalid_activation())
    def test_build_model_invalid_activation(self, name, activation):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(activation=activation)

    @parameterized.expand(invalid_optimizer())
    def test_build_model_invalid_optimizer(self, name, optimizer):
        input_dim = 10
        for output_activation, output_dim in [('linear', 1), ('sigmoid', 1), ('softmax', 3)]:
            with self.subTest(output_activation=output_activation):
                build_model = build_model_factory(input_dim, output_activation, output_dim)
                with self.assertRaises(ValueError):
                    build_model(optimizer=optimizer)

if __name__ == '__main__':
    unittest.main()