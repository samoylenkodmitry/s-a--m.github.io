---
layout: leetcode-entry
title: "2144. Minimum Cost of Buying Candies With Discount"
permalink: "/leetcode/problem/2026-06-01-2144-minimum-cost-of-buying-candies-with-discount/"
leetcode_ui: true
entry_slug: "2026-06-01-2144-minimum-cost-of-buying-candies-with-discount"
---

[2144. Minimum Cost of Buying Candies With Discount](https://leetcode.com/problems/minimum-cost-of-buying-candies-with-discount/solutions/8305640/kotlin-rust-by-samoylenkodmitry-1dvi/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01062026-2144-minimum-cost-of-buying?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/vDQlS-53V2Y)

https://dmitrysamoylenko.com/leetcode/

![01.06.2026.webp](/assets/leetcode_daily_images/01.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1377

#### Problem TLDR

Min sum taking 2 out of 3

#### Intuition

Sort descending, take greedily

#### Approach

* if you not sure which way is better forward or backward, you can make min(forward,backward)

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n|1)$$

#### Code

```kotlin
    fun minimumCost(c: IntArray) =
        c.sortedDescending().filterIndexed {i,_->i%3<2}.sum()
```
```rust
    pub fn minimum_cost(mut c: Vec<i32>) -> i32 {
        c.sort_unstable_by(|a, b| b.cmp(a));
        c.chunks(3).flat_map(|c| c.iter().take(2)).sum()
    }
```

