---
layout: leetcode-entry
title: "3550. Smallest Index With Digit Sum Equal to Index"
permalink: "/leetcode/problem/2026-09-24-3550-smallest-index-with-digit-sum-equal-to-index/"
leetcode_ui: true
entry_slug: "2026-09-24-3550-smallest-index-with-digit-sum-equal-to-index"
---

[3550. Smallest Index With Digit Sum Equal to Index](https://leetcode.com/problems/smallest-index-with-digit-sum-equal-to-index/solutions/8537505/kotlin-rust-by-samoylenkodmitry-2qmr/) easy
[substack](https://dmitriisamoilenko.substack.com/p/24092026-3550-smallest-index-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1mhYPbjj8cc)

https://dmitrysamoylenko.com/leetcode/

![24.09.2026.webp](/assets/leetcode_daily_images/24.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1492

#### Problem TLDR

Position equal to number digits sum

#### Intuition

Iterate & check. Max position is 27 which is equal to the sum of 999

#### Approach

* Kotlin: find, indexOfFirst, zip
* Rust: find, zip

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun smallestIndex(n: IntArray)=
    n.indices.find{it=="${n[it]}".sumOf{it-'0'}}?:-1
```
```rust
    pub fn smallest_index(n: Vec<i32>) -> i32 {
        (0..).zip(n).find(|(i,x)|*i==x/1000+x/100%10+x/10%10+x%10).map_or(-1,|p|p.0)
    }
```

