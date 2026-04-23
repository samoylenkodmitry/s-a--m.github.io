---
layout: leetcode-entry
title: "1018. Binary Prefix Divisible By 5"
permalink: "/leetcode/problem/2025-11-24-1018-binary-prefix-divisible-by-5/"
leetcode_ui: true
entry_slug: "2025-11-24-1018-binary-prefix-divisible-by-5"
---

[1018. Binary Prefix Divisible By 5](https://leetcode.com/problems/binary-prefix-divisible-by-5/description/) easy
[blog post](https://leetcode.com/problems/binary-prefix-divisible-by-5/solutions/7371006/kotlin-rust-by-samoylenkodmitry-vs63/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24112025-1018-binary-prefix-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8UwLYNOOoK8)

![b21bafa5-e591-49e3-8c72-f8752080b3a0 (1).webp](/assets/leetcode_daily_images/5375a2c3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1183

#### Problem TLDR

Prefixes divisible by 5 #easy #brainteaser

#### Intuition

```j
    // its not easy; its brainteaser
    // the 10^5 length
    // how to convert base_2 to base_5 directly?
    // to convert to base_10
    // let's try convert to long base_10
    // it goes out of range too fast
    // can we trim it % 5?
```

#### Approach

* the remainder of %5 is enough

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 3ms
    fun prefixesDivBy5(n: IntArray) =
    { var x = 0; n.map {x = (x*2+it)%5; x<1 }}()
```
```rust
// 0ms
    pub fn prefixes_div_by5(n: Vec<i32>) -> Vec<bool> {
        let mut x = 0; n.iter().map(|&n|{x=(x*2+n)%5; x<1}).collect()
    }
```

