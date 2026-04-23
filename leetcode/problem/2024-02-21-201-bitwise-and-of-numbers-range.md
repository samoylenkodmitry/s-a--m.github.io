---
layout: leetcode-entry
title: "201. Bitwise AND of Numbers Range"
permalink: "/leetcode/problem/2024-02-21-201-bitwise-and-of-numbers-range/"
leetcode_ui: true
entry_slug: "2024-02-21-201-bitwise-and-of-numbers-range"
---

[201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/description) medium
[blog post](https://leetcode.com/problems/bitwise-and-of-numbers-range/solutions/4760909/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21022024-201-bitwise-and-of-numbers?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VXQznHHZnNE)
![image.png](/assets/leetcode_daily_images/36b83a27.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/514

#### Problem TLDR

Bitwise AND for [left..right].

#### Intuition

To understand the problem, let's observe how this works:

```bash
    // 0  0000
    // 1  0001           2^0
    // 2  0010
    // 3  0011
    // 4  0100 3..4 = 0  2^2
    // 5  0101 3..5 = 0
    // 6  0110
    // 7  0111 6..7
    // 8  1000           2^3
    // 9  1001  7..9 = 0
```
Some observations:
* When interval intersects `4`, `8` and so on, it `AND` operation becomes `0`.
* Otherwise, we take the common prefix: `6: 0110 & 7: 0111 = 0110`.

#### Approach

We can take the `most significant bit` and compare it.
In another way, we can just find the common prefix trimming the bits from the right side.

#### Complexity

- Time complexity:
$$O(1)$$, at most 32 calls happens

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun rangeBitwiseAnd(left: Int, right: Int): Int {
    if (left == right) return left
    val l = left.takeHighestOneBit()
    val r = right.takeHighestOneBit()
    return if (l != r) 0 else
      l or rangeBitwiseAnd(left xor l, right xor r)
  }

```
```rust

  pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    if left == right { left }
    else { Self::range_bitwise_and(left / 2, right / 2) * 2 }
  }

```

