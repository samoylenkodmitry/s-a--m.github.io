---
layout: leetcode-entry
title: "476. Number Complement"
permalink: "/leetcode/problem/2024-08-22-476-number-complement/"
leetcode_ui: true
entry_slug: "2024-08-22-476-number-complement"
---

[476. Number Complement](https://leetcode.com/problems/number-complement/description/) easy
[blog post](https://leetcode.com/problems/number-complement/solutions/5672424/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22082024-476-number-complement?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Q4Xoe2zkJGo)
![1.webp](/assets/leetcode_daily_images/0eb0ffdf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/710

#### Problem TLDR

Invert bits: `101` becomes `010` #easy #bit_manipulation

#### Intuition

One way to do it is inverting all bits and then applying some mask to trim the bits to the left:

```j

0000 0000  0000 0000  0000 0000  0000 0101    5
0000 0000  0000 0000  0000 0000  0000 0100    5.takeHighestOneBit()
0000 0000  0000 0000  0000 0000  0000 1000    5.takeHighestOneBit() shl 1
0000 0000  0000 0000  0000 0000  0000 0111    (5.takeHighestOneBit() shl 1) - 1

```
Now we can use that mask for `(~a&mask)` or just `a ^ mask`.

#### Approach

* Rust has `leading_zeros()`
* There is a cool trick to `paint` the bits to make the mask: `a >> 1 | a, a >> 2 | a, a >> 4 | a, a >> 8 | a, a >> 16 | a`.

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findComplement(num: Int) =
        num xor ((num.takeHighestOneBit() shl 1) - 1)

```
```rust

    pub fn find_complement(num: i32) -> i32 {
        num ^ ((1 << (32 - num.leading_zeros())) - 1)
    }

```

