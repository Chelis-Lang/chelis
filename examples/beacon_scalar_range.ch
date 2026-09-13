def neuron(x: tensor[f64]) -> tensor[f64] = relu(x)
@property bounded_neuron forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:
  (tensor_to_scalar(neuron(x)) <= 2.0f64)
