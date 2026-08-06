# Third-party notices

SteerDemo uses `sentence-transformers/all-MiniLM-L6-v2` for its Core ML topic
judge. The model is distributed under the Apache License 2.0. The repository
ships a converted FP16 copy of its weights and the matching tokenizer assets.
The exact upstream revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
its Apache 2.0 license is reproduced in `LICENSES/all-MiniLM-L6-v2-LICENSE.txt`.

MLX Swift, MLX Swift Examples, and Swift Transformers are resolved as Swift
package dependencies under their respective licenses; their source is not
vendored here.

The Qwen model is fetched at runtime rather than redistributed. SteerDemo pins
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` revision
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
