---
layout: leetcode-entry
title: "2154. Keep Multiplying Found Values by Two"
permalink: "/leetcode/problem/2025-11-19-2154-keep-multiplying-found-values-by-two/"
leetcode_ui: true
entry_slug: "2025-11-19-2154-keep-multiplying-found-values-by-two"
---

[2154. Keep Multiplying Found Values by Two](https://leetcode.com/problems/keep-multiplying-found-values-by-two/description/) easy
[blog post](https://leetcode.com/problems/keep-multiplying-found-values-by-two/solutions/7359537/kotlin-rust-by-samoylenkodmitry-jcgw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19112025-2154-keep-multiplying-found?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Mke82HIi3uY)

![c4be91d4-0c0e-4b50-bbd4-237ec2f1d49d (1).webp](/assets/leetcode_daily_images/d101e27f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1178

#### Problem TLDR

First `original=original*2` not in array #easy

#### Intuition

Simulate the proces. Use hashset to speedup.

#### Approach

* for 1000 elements O(n^2) is acceptable

#### Complexity

- Time complexity:
$$O(n + log(max))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 15ms
    fun findFinalValue(n: IntArray, o: Int) =
        o shl (0..10).first { o shl it !in n }
```
```rust
// 0ms
    pub fn find_final_value(n: Vec<i32>, o: i32) -> i32 {
       o << (0..11).find(|b| !n.contains(&(o << b))).unwrap()
    }
```

