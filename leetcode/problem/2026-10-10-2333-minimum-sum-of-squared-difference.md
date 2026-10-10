---
layout: leetcode-entry
title: "2333. Minimum Sum of Squared Difference"
permalink: "/leetcode/problem/2026-10-10-2333-minimum-sum-of-squared-difference/"
leetcode_ui: true
entry_slug: "2026-10-10-2333-minimum-sum-of-squared-difference"
---

[2333. Minimum Sum of Squared Difference](https://leetcode.com/problems/minimum-sum-of-squared-difference/solutions/8565472/kotlin-rust-by-samoylenkodmitry-lgwp/) medium
[substack](https://dmitriisamoilenko.substack.com/p/10102026-2333-minimum-sum-of-squared?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HzF_Y4lO2mk)

https://dmitrysamoylenko.com/leetcode/

![10.10.2026.webp](/assets/leetcode_daily_images/10.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1508

#### Problem TLDR

Min sum of squared difs after adjusting each array by k

#### Intuition

Didn't solved myself.
Binary search solution: find the lowest threshold max diff. Calculate leftover operations and spread them at most one per item if it equal the threshold.
Counting solution: decrement the counts taking as much as you can and moving to the count-1 position. Do the final sweep.

#### Approach

* the leftover ops are guaranteed to spread at most one per item, because of math: (x-2)^2+y^2>(x-1)^2+(y-1)^2, the binary search would move the threshold otherwise

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minSumSquareDiff(n1: IntArray, n2: IntArray, k1: Int, k2: Int): Long {
        val c = LongArray(100001); for (i in n1.indices) c[abs(n1[i] - n2[i])]++
        var k = 1L * k1 + k2
        for (i in 100000 downTo 1) { val t = min(k, c[i]); c[i] -= t; c[i - 1] += t; k -= t }
        return c.indices.sumOf { c[it] * it * it }
    }
```
```rust
    pub fn min_sum_square_diff(a: Vec<i32>, b: Vec<i32>, k1: i32, k2: i32) -> i64 {
        let (mut count, mut k) = (vec![0i64; 100001], (k1 + k2) as i64);
        for (x, y) in a.into_iter().zip(b) { count[(x - y).abs() as usize] += 1 }
        for i in (1..=100000).rev() { if count[i] > 0 && k > 0 {
            let take = k.min(count[i]); count[i] -= take; count[i - 1] += take; k -= take
        }}
        (0..).zip(count).map(|(i, c)| c * i * i).sum()
    }
```

