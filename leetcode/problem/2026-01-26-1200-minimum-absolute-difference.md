---
layout: leetcode-entry
title: "1200. Minimum Absolute Difference"
permalink: "/leetcode/problem/2026-01-26-1200-minimum-absolute-difference/"
leetcode_ui: true
entry_slug: "2026-01-26-1200-minimum-absolute-difference"
---

[1200. Minimum Absolute Difference](https://leetcode.com/problems/minimum-absolute-difference/description) easy
[blog post](https://leetcode.com/problems/minimum-absolute-difference/solutions/7525930/kotlin-rust-by-samoylenkodmitry-hhk4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26012026-1200-minimum-absolute-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kCrw4PxV1e8)

![e7c9fdb5-0db6-4819-97ab-47f397a1bf45 (1).webp](/assets/leetcode_daily_images/0d8a1d2f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1249

#### Problem TLDR

Min diff pairs #easy

#### Intuition

Sort, compare adjucent pairs, find min, collect.

#### Approach

* groupBy.min also works
* updating the min diff and collecting can be done in a single go

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 138ms
    fun minimumAbsDifference(a: IntArray) = a.sorted()
    .windowed(2).groupBy{(a,b)->b-a}.minBy{(k,v)->k}.value
```
```rust
// 3ms
    pub fn minimum_abs_difference(mut a: Vec<i32>) -> Vec<Vec<i32>> {
        a.sort(); let mut d = (1..a.len()).map(|i|a[i]-a[i-1]).min().unwrap();
        a.windows(2).filter(|w|w[1]-w[0]==d).map(Vec::from).collect()
    }
```

