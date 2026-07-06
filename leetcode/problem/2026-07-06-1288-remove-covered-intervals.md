---
layout: leetcode-entry
title: "1288. Remove Covered Intervals"
permalink: "/leetcode/problem/2026-07-06-1288-remove-covered-intervals/"
leetcode_ui: true
entry_slug: "2026-07-06-1288-remove-covered-intervals"
---

[1288. Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/solutions/8379059/kotlin-rust-by-samoylenkodmitry-0hva/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06072026-1288-remove-covered-intervals?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DUXM7S8UToE)

https://dmitrysamoylenko.com/leetcode/

![06.07.2026.webp](/assets/leetcode_daily_images/06.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1412

#### Problem TLDR

Count non intersecting intervals

#### Intuition

Sort by (left, - right) to take more spread intervals first and skip others.

#### Approach

* can be a single key a*max-b

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun removeCoveredIntervals(iv: Array<IntArray>)= run {
        var m = 0; iv.sortedWith(compareBy({it[0]},{-it[1]}))
        .count { (l,r) -> r > m.also { m = max(m, r) } }
    }
```
```rust
    pub fn remove_covered_intervals(mut iv: Vec<Vec<i32>>) -> i32 {
        iv.sort_by_key(|i|(i[0],-i[1]));
        iv.iter().fold((0,0),|(c,m),i|(c+(i[1]>m)as i32,m.max(i[1]))).0
    }
```

