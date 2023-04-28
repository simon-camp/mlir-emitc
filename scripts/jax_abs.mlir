module @jit_abs {
  func.func public @main(%arg0: tensor<2x3xf32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}) -> (tensor<2x3xf32> {jax.result_info = ""}) {
    %0 = stablehlo.abs %arg0 : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
