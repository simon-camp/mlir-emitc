import argparse

import jax
import jax.numpy as jnp

@jax.jit
def abs(x):
  return jnp.abs(x)

def convert_jax_function(f) -> str:
    return str(f.lower(jnp.zeros((2, 3))).compiler_ir())

def translate(output_path: str):
    with open(output_path, "w") as file:
        file.write(convert_jax_function(abs))


def main():
    parser = argparse.ArgumentParser(
        description="Genrate a jax model and convert it to stablehlo mlir dialect")
    parser.add_argument("output_path",
                        metavar="output-path",
                        help="Output path")
    args = parser.parse_args()

    translate(args.output_path)


if __name__ == "__main__":
    main()
