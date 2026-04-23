---
layout: leetcode-entry
title: "1482. Minimum Number of Days to Make m Bouquets"
permalink: "/leetcode/problem/2024-06-19-1482-minimum-number-of-days-to-make-m-bouquets/"
leetcode_ui: true
entry_slug: "2024-06-19-1482-minimum-number-of-days-to-make-m-bouquets"
---

[1482. Minimum Number of Days to Make m Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/solutions/5334796/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19062024-1482-minimum-number-of-days?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MGRlPAJmtc4)
![2024-06-19_06-00_1.webp](/assets/leetcode_daily_images/72eac202.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/644

#### Problem TLDR

Min days to take `m` `k`-subarrays #medium #binary_search

#### Intuition

```j

    //   1 10  3 10  2         m=3 k=1
    //   1
    //               2
    //         3
    //     10    10

    //   7  7  7  7 12  7  7   m=2 k=3
    //  [7  7  7] 7     7  7   +1
    //           [  12   ]     +2

```

We can binary search in space of days as function grows from `not possible` to `possible` with increase of days.

#### Approach

Don't forget the `-1` case.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minDays(bloomDay: IntArray, m: Int, k: Int): Int {
        var lo = 0; var hi = bloomDay.max(); var min = Int.MAX_VALUE
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            var curr = 0; var count = 0
            for (d in bloomDay) {
                if (d > mid) curr = 0 else curr++
                if (curr == k) { curr = 0; count++ }
            }
            if (count >= m) { hi = mid - 1; min = min(min, mid) }
            else lo = mid + 1
        }
        return if (min == Int.MAX_VALUE) -1 else min
    }

```
```rust

    pub fn min_days(bloom_day: Vec<i32>, m: i32, k: i32) -> i32 {
        let (mut lo, mut hi, mut min) = (0, *bloom_day.iter().max().unwrap(), i32::MAX);
        while lo <= hi {
            let (mid, mut curr, mut count) = (lo + (hi - lo) / 2, 0, 0);
            for &d in &bloom_day {
                curr = if d > mid { 0 } else { curr + 1 };
                if curr == k { curr = 0; count += 1 }
            }
            if count >= m { hi = mid - 1; min = min.min(mid) }
            else { lo = mid + 1 }
        }
        if min == i32::MAX { -1 } else { min }
    }

```

