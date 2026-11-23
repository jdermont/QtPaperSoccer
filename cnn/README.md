Generated summary by Claude.

# Conv value networks

`conv32x6.convnet` and `conv64x6.convnet` are residual conv **value** nets (no policy). See
`diagram_conv_network.png` (drawn for the 64-filter net, the 32-filter one has the same layout).
Output: `tanh` value in [-1, 1] for the **player to move**, for non-terminal positions only
(handle the ball in a goal or a blocked ball with the game rules).

## Input: `[9, 11, 9]` (channels, rows y = 0..10, columns x = 0..8)

Only the 9x11 field nodes; the goal rows aren't part of the input.

**Orientation:** if the player to move is ONE, use the board as it is (`Pitch::getPosition`). If it is TWO,
rotate the board 180° (x -> 8 - x, y -> 10 - y, i.e. `EDGES_MIRRORED` / `BALLS_MIRRORED`). After that, the
y = -1 goal (row above row 0) is the goal the player to move **defends**, and the y = 11 goal is the one it attacks.

Per node (row y, column x), in the oriented board:

| channel | meaning |
|---|---|
| 0..7 | 1 if the move in direction d **can't** be played from this node, else 0 |
| 8 | 1 on the ball's node, 0 elsewhere |

The directions (dx, dy) are, with y growing towards the attacked goal:
`0 (0,-1)`, `1 (1,-1)`, `2 (1,0)`, `3 (1,1)`, `4 (0,1)`, `5 (-1,1)`, `6 (-1,0)`, `7 (-1,-1)`.

A direction is blocked if there is no neighbour node there, if the edge isn't in `Pitch::ALL_EDGES` (the 316
playable edges, so lines along the border are blocked), or if the edge is already drawn. Bounces don't count:
a free edge from a visited node is still 0. Example check, empty pitch: the per-channel sums over the field are
`[28, 17, 23, 17, 28, 17, 23, 17, 1]`. The corner (0, 0) has only direction 3 open, and the goal-mouth node (4, 0)
has every direction open.

## File format (`PSCONVB`, little endian)

- `char[8]` `"PSCONVB\0"`
- `int32[17]`: planes (10), filters N (32 / 64), blocks (6), headChannels C (8), nFc (1), then 0, 0, 0, 0,
  valuePatch R (1), valuePatchChannels Cv (32), 0, 0, 0, 0, 0, crop (1)
- `int32 fcWidths[nFc]` (128)
- `float32` tensors, each as weight then bias, in this order (BatchNorm is already folded in):
  - stem `w[N][9][3][3]`, `b[N]`
  - for each block: conv1 `w[N][N][3][3]`, `b[N]`, conv2 `w[N][N][3][3]`, `b[N]`
  - head 1x1 conv `w[C][N]`, `b[C]`
  - value patch 1x1 conv `w[Cv][N]`, `b[Cv]`
  - fc `W[128][C*99 + Cv*9]`, `b[128]`
  - out `W[1][128]`, `b[1]`
- then EOF. Float counts: 253417 (32x6), 589481 (64x6).

The weight layouts match PyTorch directly (`conv2d` `[out][in][kh][kw]`, `linear` `[out][in]`).

## Forward

All 3x3 convs use `padding=1` (zero padding outside the 9x11 field).

```
h = relu(conv3x3(x, stem))
for each block:  h = relu(h + conv3x3(relu(conv3x3(h, conv1)), conv2))
v = relu(conv1x1(h, head)).flatten(1)                       # [B, C*99], channel-major, then row, then column
p = relu(conv1x1(h, vpatch))                                # [B, Cv, 11, 9]
p = zero-pad p by 1, take the 3x3 window centred on the ball (from channel 8 of x)
    -> [B, Cv, 3, 3].flatten(1)                             # [B, Cv*9], row-major from the top left
y = relu(linear(cat([v, p], 1), fc))
value = tanh(linear(y, out))
```

## Notes

- Optional: averaging with the left-right mirror (x -> 8 - x, channels 1<->7, 2<->6, 3<->5) gives slightly
  better values.
- Sanity values for the empty pitch with the ball in the centre (4, 5): conv32x6 `0.000826`, conv64x6 `-0.001608`.
- The forward above was checked against the training code's loader (identical outputs on 4000 positions).
