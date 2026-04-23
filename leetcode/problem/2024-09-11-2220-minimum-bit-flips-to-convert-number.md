---
layout: leetcode-entry
title: "2220. Minimum Bit Flips to Convert Number"
permalink: "/leetcode/problem/2024-09-11-2220-minimum-bit-flips-to-convert-number/"
leetcode_ui: true
entry_slug: "2024-09-11-2220-minimum-bit-flips-to-convert-number"
---

[2220. Minimum Bit Flips to Convert Number](https://leetcode.com/problems/minimum-bit-flips-to-convert-number/description/) easy
[blog post](https://leetcode.com/problems/minimum-bit-flips-to-convert-number/solutions/5769518/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11092024-2220-minimum-bit-flips-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BfaJBZ9Hjeo)
![1.webp](/assets/leetcode_daily_images/ee1ecdb0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/731

#### Problem TLDR

Bit diff between two numbers #easy #bit_manipulation

#### Intuition

```j

    // 10 1010
    //  7 0111
    //    ** *

```

To find the bits count there are several hacks:
https://stackoverflow.com/a/109025/23151041

https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

```j

x = (x & 0b1010101010101010101010101010101) + ((x >> 1) & 0b1010101010101010101010101010101);
x = (x & 0b0110011001100110011001100110011) + ((x >> 2) & 0b0110011001100110011001100110011);
x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
x = (x & 0x0000FFFF) + ((x >> 16)& 0x0000FFFF);

+-------------------------------+
| 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |  <- x
|  1 0  |  0 1  |  0 1  |  0 1  |  <- first time merge
|    0 0 1 1    |    0 0 1 0    |  <- second time merge
|        0 0 0 0 0 1 0 1        |  <- third time ( answer = 00000101 = 5)
+-------------------------------+

```

#### Approach

* let's use built-in methods

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minBitFlips(start: Int, goal: Int) =
        (start xor goal).countOneBits()

```
```rust

    pub fn min_bit_flips(start: i32, goal: i32) -> i32 {
        (start ^ goal).count_ones() as i32
    }

```
```c++

    int minBitFlips(int start, int goal) {
        return __builtin_popcount(start ^ goal);
    }

```

