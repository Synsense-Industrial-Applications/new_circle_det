"""Shared address and channel layout for the current 16-channel Layer-4.

Channels 0..7 are the original directional outputs.  Channels 8..15 repeat
the same four optical-flow directions and 2x2 sub-pixel addresses, but use the
complementary spatial kernel.  Because the source flow direction is unchanged,
each new channel stays in the same D/S vote family as its original counterpart.
"""

from __future__ import annotations


IMAGE_SIZE = 128
LAYER4_SOURCE_SIZE = 64
BASE_FEATURE_COUNT = 8
LAYER4_FEATURE_COUNT = 16

_BASE_FEATURE_TO_DIRECTION = (0, 1, 1, 0, 2, 3, 3, 2)
FEATURE_TO_DIRECTION = _BASE_FEATURE_TO_DIRECTION * 2

# Flow directions 0/2 constrain D = y - x; directions 1/3 constrain S = x + y.
# The second output bank changes its evidence kernel, not its flow direction.
D_HOUGH_FEATURES = frozenset((0, 3, 4, 7, 8, 11, 12, 15))
S_HOUGH_FEATURES = frozenset(range(LAYER4_FEATURE_COUNT)) - D_HOUGH_FEATURES


def decode_layer4_address(x64: int, y64: int, feature: int) -> tuple[int, int]:
    """Decode one 64x64x16 Layer-4 address to a 128x128 pixel address."""

    x64 = int(x64)
    y64 = int(y64)
    feature = int(feature)
    if not 0 <= x64 < LAYER4_SOURCE_SIZE or not 0 <= y64 < LAYER4_SOURCE_SIZE:
        raise ValueError(f"Layer-4 coordinates out of range: ({x64}, {y64})")
    if not 0 <= feature < LAYER4_FEATURE_COUNT:
        raise ValueError(
            f"Layer-4 feature must be in 0..{LAYER4_FEATURE_COUNT - 1}, "
            f"got {feature}"
        )

    # Both eight-channel banks retain the original channel's 2x2 address.
    feature_mod4 = feature % 4
    x128 = 2 * x64 + feature_mod4 // 2
    y128 = 2 * y64 + feature_mod4 % 2
    return x128, y128
