# Core ML topic-encoder parity

The same fixed-length token tensors were evaluated by PyTorch and the exported Core ML model.
Cosine similarity is computed between their normalized 384-dimensional embeddings.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Core ML Tools: `9.0`
- PyTorch: `2.7.0`
- Maximum length: `128` tokens
- Compute units: `ALL`
- Weight SHA-256: `f6d040d94a3a476264c26cb3b6d260aa695784121667e6bff6690f56de558d41`
- Minimum cosine: **0.999914432**
- Required minimum: **0.9999**

| # | Sentence | cosine | max abs delta | relative L2 | PyTorch norm | Core ML norm |
|---:|---|---:|---:|---:|---:|---:|
| 1 | The bride and groom exchanged vows beside the lake. | 0.999984465 | 8.780e-04 | 5.577e-03 | 1.000000070 | 1.000176487 |
| 2 | Their wedding reception continued long after sunset. | 0.999979864 | 1.195e-03 | 6.350e-03 | 1.000000038 | 0.999753456 |
| 3 | A quiet morning began with coffee and a newspaper. | 0.999984860 | 9.711e-04 | 5.514e-03 | 1.000000043 | 1.000343067 |
| 4 | The software release includes a smaller inference engine. | 0.999981149 | 1.050e-03 | 6.141e-03 | 0.999999968 | 0.999900519 |
| 5 | Waves rolled toward the beach under a bright moon. | 0.999979046 | 1.060e-03 | 6.479e-03 | 0.999999965 | 0.999716188 |
| 6 | Sailors watched the tide from the harbor. | 0.999983947 | 9.764e-04 | 5.668e-03 | 1.000000002 | 1.000112090 |
| 7 | The telescope revealed a distant spiral galaxy. | 0.999980951 | 8.525e-04 | 6.172e-03 | 0.999999995 | 1.000011551 |
| 8 | An astronaut prepared the spacecraft for orbit. | 0.999980791 | 9.890e-04 | 6.208e-03 | 1.000000017 | 1.000320778 |
| 9 | Please describe a practical way to learn a new language. | 0.999980151 | 9.698e-04 | 6.301e-03 | 0.999999996 | 1.000025495 |
| 10 | A good friend listens carefully and keeps promises. | 0.999979262 | 1.033e-03 | 6.440e-03 | 0.999999951 | 1.000039046 |
| 11 | The ceremony ended when the newlyweds left the altar. | 0.999985434 | 8.717e-04 | 5.399e-03 | 1.000000024 | 0.999849994 |
| 12 | Coral reefs support diverse marine ecosystems. | 0.999981950 | 8.832e-04 | 6.010e-03 | 0.999999993 | 1.000113190 |
| 13 | Mars appears red because iron minerals oxidize. | 0.999981956 | 1.071e-03 | 6.008e-03 | 1.000000013 | 0.999897841 |
| 14 | A rainy afternoon can be a good time to read. | 0.999981658 | 9.677e-04 | 6.058e-03 | 0.999999989 | 1.000088072 |
| 15 | The team measured latency before choosing an optimization. | 0.999978702 | 1.150e-03 | 6.531e-03 | 1.000000020 | 0.999749071 |
| 16 | Fresh bread requires patience while the dough rises. | 0.999976147 | 1.270e-03 | 6.909e-03 | 0.999999997 | 0.999789686 |
| 17 | The engagement party included family and close friends. | 0.999979491 | 9.157e-04 | 6.405e-03 | 0.999999954 | 1.000026461 |
| 18 | Ocean currents redistribute heat around the planet. | 0.999981444 | 1.085e-03 | 6.094e-03 | 1.000000047 | 1.000149334 |
| 19 | Stars form when dense clouds of gas collapse. | 0.999982512 | 1.068e-03 | 5.916e-03 | 0.999999988 | 0.999820615 |
| 20 | A short walk through the park cleared her mind. | 0.999973546 | 1.043e-03 | 7.278e-03 | 1.000000004 | 1.000210830 |
| 21 | A nearly full context checks stable padding and pooling. A nearly full context checks stable ... | 0.999980539 | 1.110e-03 | 6.239e-03 | 1.000000089 | 1.000016131 |
| 22 | Tokens beyond the fixed context must be truncated deterministically. Tokens beyond the fixed ... | 0.999987015 | 7.565e-04 | 5.098e-03 | 1.000000072 | 1.000127702 |
| 23 | *(empty string)* | 0.999914432 | 1.728e-03 | 1.308e-02 | 1.000000006 | 1.000192412 |
| 24 | 海辺で友人たちは静かな結婚式を祝いました。 | 0.999984307 | 8.352e-04 | 5.606e-03 | 0.999999927 | 0.999775622 |
