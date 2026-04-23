---
layout: leetcode-entry
title: "2110. Number of Smooth Descent Periods of a Stock"
permalink: "/leetcode/problem/2025-12-15-2110-number-of-smooth-descent-periods-of-a-stock/"
leetcode_ui: true
entry_slug: "2025-12-15-2110-number-of-smooth-descent-periods-of-a-stock"
---

[2110. Number of Smooth Descent Periods of a Stock](https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock/description/) medium
[blog post](https://leetcode.com/problems/number-of-smooth-descent-periods-of-a-stock/solutions/7415252/kotlin-rust-by-samoylenkodmitry-a4jj/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15122025-2110-number-of-smooth-descent?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pICkmctr-WM)

![d210a905-4fd3-482d-a2a6-97376705d4cb (1).webp](/assets/leetcode_daily_images/fa9a1603.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1205

#### Problem TLDR

Decreasing subarrays #medium

#### Intuition

Just count them like and arithmetic progression.

#### Approach

* res += count++

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 40ms
    fun getDescentPeriods(p: IntArray) =
    p.indices.fold(0 to 0L) { (cnt, res), i ->
        val c = 1 + if (i > 0 && p[i] == p[i-1] - 1) cnt else 0
        c to (res + c)
    }.second
```
```rust
// 0ms
    pub fn get_descent_periods(p: Vec<i32>) -> i64 {
        (0..p.len()).fold((0,0), |(cnt,res), i| {
            let c = 1 + if i > 0 && p[i] == p[i-1]-1 { cnt } else { 0 };
            (c, res + c)
        }).1
    }
```

