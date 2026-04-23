---
layout: leetcode-entry
title: "717. 1-bit and 2-bit Characters"
permalink: "/leetcode/problem/2025-11-18-717-1-bit-and-2-bit-characters/"
leetcode_ui: true
entry_slug: "2025-11-18-717-1-bit-and-2-bit-characters"
---

[717. 1-bit and 2-bit Characters](https://leetcode.com/problems/1-bit-and-2-bit-characters/description/) easy
[blog post](https://leetcode.com/problems/1-bit-and-2-bit-characters/solutions/7357068/kotlin-rust-by-samoylenkodmitry-ofyy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18112025-717-1-bit-and-2-bit-characters?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VAqP6LWAzhs)

![16aa5b10-bbe8-4c9a-b673-39462c191a4e (1).webp](/assets/leetcode_daily_images/7cef2cc8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1177

#### Problem TLDR

Is '0' last from characters '0','10','11' concatenation #easy

#### Intuition

I had to start with DP: in each i position if b[i] zero go next, otherwise go next->next.
Then it is just a linear solution without a choice.

#### Approach

* optimization: go from back

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 0ms
    fun isOneBitCharacter(b: IntArray): Boolean {
        var i = b.size - 2
        while (i >= 0 && b[i] > 0) i -= b[i]
        return (b.size - i) % 2 < 1
    }
```
```rust
// 0ms
    pub fn is_one_bit_character(b: Vec<i32>) -> bool {
        b[..b.len()-1].split(|&x|x<1).last().unwrap().len()%2<1
    }
```

