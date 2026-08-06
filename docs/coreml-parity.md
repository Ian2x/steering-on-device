# Core ML topic-encoder parity

The same fixed-length token tensors were evaluated by PyTorch and the exported Core ML model.
Cosine similarity is computed between their normalized 384-dimensional embeddings.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Core ML Tools: `9.0`
- PyTorch: `2.7.0`
- Maximum length: `128` tokens
- Minimum cosine: **0.999973546**

| # | Sentence | cosine(Core ML, PyTorch) |
|---:|---|---:|
| 1 | The bride and groom exchanged vows beside the lake. | 0.999984465 |
| 2 | Their wedding reception continued long after sunset. | 0.999979864 |
| 3 | A quiet morning began with coffee and a newspaper. | 0.999984860 |
| 4 | The software release includes a smaller inference engine. | 0.999981149 |
| 5 | Waves rolled toward the beach under a bright moon. | 0.999979046 |
| 6 | Sailors watched the tide from the harbor. | 0.999983947 |
| 7 | The telescope revealed a distant spiral galaxy. | 0.999980951 |
| 8 | An astronaut prepared the spacecraft for orbit. | 0.999980791 |
| 9 | Please describe a practical way to learn a new language. | 0.999980151 |
| 10 | A good friend listens carefully and keeps promises. | 0.999979262 |
| 11 | The ceremony ended when the newlyweds left the altar. | 0.999985434 |
| 12 | Coral reefs support diverse marine ecosystems. | 0.999981950 |
| 13 | Mars appears red because iron minerals oxidize. | 0.999981956 |
| 14 | A rainy afternoon can be a good time to read. | 0.999981658 |
| 15 | The team measured latency before choosing an optimization. | 0.999978702 |
| 16 | Fresh bread requires patience while the dough rises. | 0.999976147 |
| 17 | The engagement party included family and close friends. | 0.999979491 |
| 18 | Ocean currents redistribute heat around the planet. | 0.999981444 |
| 19 | Stars form when dense clouds of gas collapse. | 0.999982512 |
| 20 | A short walk through the park cleared her mind. | 0.999973546 |
