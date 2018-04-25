# Audio Bit-Depth Super Resolution

Our project focuses on the goal of adapting WaveNet, an audio prediction CNN architecture, to superresolve 8-bit audio clips into 16-bit audio clips, trying to restore lossed dynamic range and as a result cleaning compression artifacts.

We evaluate several modifications to WaveNet, including:

- Discriminative rather than autoregressive prediction
- Non-causal dilations - both input samples from past and future are available during prediction
- Delta prediction - assuming 8-bit audio mostly preserves the 16-bit audio data, we aim to only predict the delta between the two waveforms
- Real-valued prediction - since the amplitude space is inherently continuous (discretized during compression), a real-valued number space is a more natural model than a categorical output passed through softmax.

Improvements are subtle, though we terminated training early due to resource constraints and observed that loss was still decreasing approximately linearly at time of evaluation. We believe there is further improvements to be had with our architecture given sufficient training.

## Final Write up

[Audio Bit Depth Super Resolution Paper](https://github.com/wqi/bdsr/blob/master/docs/paper.pdf)

### Presentation

[Project Presentation](https://github.com/wqi/bdsr/blob/master/docs/presentation.pdf)

### Generated Samples

[Generated Samples](https://drive.google.com/drive/u/0/folders/132KivXYZcXYS2qy4dlK9ksxxLsbA3lQB)

### Source code

[Source Code](https://github.com/wqi/bdsr)

Maintained by Taylor Lundy, Thomas Liu, and William Qi.
