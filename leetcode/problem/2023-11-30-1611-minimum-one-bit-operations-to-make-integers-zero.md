---
layout: leetcode-entry
title: "1611. Minimum One Bit Operations to Make Integers Zero"
permalink: "/leetcode/problem/2023-11-30-1611-minimum-one-bit-operations-to-make-integers-zero/"
leetcode_ui: true
entry_slug: "2023-11-30-1611-minimum-one-bit-operations-to-make-integers-zero"
---

[1611. Minimum One Bit Operations to Make Integers Zero](https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/description/) hard
[blog post](https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/solutions/4345560/kotlin-eli-5-two-tricks/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30112023-1611-minimum-one-bit-operations?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/0M5p5KgVGkw)
![image.png](/assets/leetcode_daily_images/f7c44ab7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/421

#### Problem TLDR

Minimum rounds of inverting rightmost bit or left of the rightmost `1` bit to make `n` zero

#### Intuition

Let's observe the example:

```kotlin
  // 6
  // 110
  // 010 b
  // 011 a
  // 001 b
  // 000 a
  // 10 = 2 + f(1) = 2^1 + f(2^0)
  // 11 a
  // 01 b -> f(1)
  // 100 = 4 + f(10) = 2^2 + f(2^1)
  // 101 a
  // 111 b
  // 110 a
  // 010 b -> f(10)
  // 1000 = 8 + f(100) = 2^3 + f(2^2)
  // 1001 a
  // 1011 b
  // 1010 a
  // 1110 b
  // 1111 a
  // 1101 b
  // 1100 a
  // 0100 b -> f(100)
```

There are two tricks we can derive:

1. Each signle-bit number has a recurrent count of operations: f(0b100) = 0b100 + f(0b10) and so on.
2. The hard trick: when we consider the non-single-bit number, like `1101`, we do `f(0b1101) = f(0b1000) - f(0b100) + f(0b1)`.

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(log(n))$$

#### Code

```kotlin

  fun minimumOneBitOperations(n: Int): Int {
    val f = HashMap<Int, Int>()
    f[0] = 0
    f[1] = 1
    var curr = 2
    while (curr > 0) {
      f[curr] = curr + f[curr / 2]!!
      curr *= 2
    }

    var res = 0
    var sign = 1;
    for (i in 0..31) {
      val bit = 1 shl i
      if (n and bit != 0) {
        res += sign * f[bit]!!
        sign = -sign
      }
    }

    return Math.abs(res)
  }

```

