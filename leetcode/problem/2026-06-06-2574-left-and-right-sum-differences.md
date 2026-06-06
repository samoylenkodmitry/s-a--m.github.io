---
layout: leetcode-entry
title: "2574. Left and Right Sum Differences"
permalink: "/leetcode/problem/2026-06-06-2574-left-and-right-sum-differences/"
leetcode_ui: true
entry_slug: "2026-06-06-2574-left-and-right-sum-differences"
---

[2574. Left and Right Sum Differences](https://leetcode.com/problems/left-and-right-sum-differences/solutions/8316993/kotlin-rust-by-samoylenkodmitry-we5y/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06062026-2574-left-and-right-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SuQyVx0VcSE)

https://dmitrysamoylenko.com/leetcode/

![06.06.2026.webp](/assets/leetcode_daily_images/06.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1382

#### Problem TLDR

abs(prefix sum - suffix sum)

#### Intuition

* we can use math to just re-use the sum as is

#### Approach

Compute the sum - this is the suffix; use a single variable for prefix sum. Problem is small 1000 elements and 10^5 items, meaning we are in 32 bits

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun leftRightDifference(n: IntArray) =
    n.run {val s = sum(); var l = 0; map {l += it; abs(2*l-it-s)}}
```
```rust
    pub fn left_right_difference(n: Vec<i32>) -> Vec<i32> {
        let (s, mut l) = (n.iter().sum::<i32>(), 0);
        n.iter().map(|x| { l += x; (2*l-x-s).abs()}).collect()
    }
```

