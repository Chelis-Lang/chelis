def neuron(x: tensor[f64]) -> tensor[f64] = relu(x)
@property bounded_neuron forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:
  ((x |> neuron |> tensor_to_scalar) <= 2.0f64)
