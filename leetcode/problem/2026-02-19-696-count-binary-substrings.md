---
layout: leetcode-entry
title: "696. Count Binary Substrings"
permalink: "/leetcode/problem/2026-02-19-696-count-binary-substrings/"
leetcode_ui: true
entry_slug: "2026-02-19-696-count-binary-substrings"
---

[696. Count Binary Substrings](https://open.substack.com/pub/dmitriisamoilenko/p/19022026-696-count-binary-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/19022026-696-count-binary-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19022026-696-count-binary-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mm8w6ijHkeA)

![ca5bca1d-de86-418b-815f-1df4df895c62 (1).webp](/assets/leetcode_daily_images/4b06dbaf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1274

#### Problem TLDR

Count substrings of *01* and *10* #easy #sliding_window

#### Intuition

Count consequent zeros and ones. Slide pair-wise res += min(prev, curr).

#### Approach

* the fun solution is to split the string
* itertools in Rust allow nested windows without vec allocation

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 69ms
    fun countBinarySubstrings(s: String) = s
    .replace("01", "0 1").replace("10", "1 0").split(" ")
    .zipWithNext { a, b -> min(a.length, b.length)}.sum()
```
```rust
// 0ms
	pub fn count_binary_substrings(s: String) -> i32 {
        s.as_bytes().chunk_by(|a,b|a==b).map(|c|c.len() as i32)
        .tuple_windows().map(|(a,b)|a.min(b)).sum()
    }
```

