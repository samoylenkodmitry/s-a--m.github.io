---
layout: leetcode-entry
title: "1848. Minimum Distance to the Target Element"
permalink: "/leetcode/problem/2026-04-13-1848-minimum-distance-to-the-target-element/"
leetcode_ui: true
entry_slug: "2026-04-13-1848-minimum-distance-to-the-target-element"
---

[1848. Minimum Distance to the Target Element](https://leetcode.com/problems/minimum-distance-to-the-target-element/solutions/7887798/kotlin-rust-by-samoylenkodmitry-hb66/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15042026-1848-minimum-distance-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CyoCyNhhEC8)

![13.04.2026.webp](/assets/leetcode_daily_images/13.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1327

#### Problem TLDR

Distance to target from start #easy

#### Intuition

Iterate from left to right.
Or iterate from the start.

#### Approach

* Kotlin 'zip' doesn't work with non-equal ranges
* Rust itertools has 'interleave', abs_diff

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 16ms
    fun getMinDistance(n: IntArray, t: Int, s: Int) =
    n.indices.filter { n[it] == t }.minOf { abs(it - s) }
```
```rust
// 0ms
    pub fn get_min_distance(n: Vec<i32>, t: i32, s: i32) -> i32 {
        (0..=s as usize).rev().interleave(s as usize+1..n.len())
        .find(|&i| n[i] == t).unwrap().abs_diff(s as usize) as _
    }
```

