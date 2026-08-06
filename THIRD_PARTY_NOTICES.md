# Third-party notices

SteerDemo uses `sentence-transformers/all-MiniLM-L6-v2` for its Core ML topic
judge. The model is distributed under the Apache License 2.0. The repository
ships a converted FP16 copy of its weights and the matching tokenizer assets.
The exact upstream revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
its Apache 2.0 license is reproduced in `LICENSES/all-MiniLM-L6-v2-LICENSE.txt`.

MLX Swift and Swift Transformers are resolved as Swift package dependencies
under their respective licenses. SteerDemo also adapts the Qwen2 model source
from `ml-explore/mlx-swift-examples` revision
`9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631` to expose a residual-stream
boundary for the ActAdd research pane. That adapted source remains under the
MIT License reproduced in `LICENSES/mlx-swift-examples-LICENSE.txt`.

The Qwen model is fetched at runtime rather than redistributed. SteerDemo pins
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` revision
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
