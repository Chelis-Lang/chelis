Keep runtime fold result extents independent across generic call
activations, including gradients of List-valued models. Recurrence adjoints
retain the shape of their differentiated input. Callback and caller shape
claims remain checked.
