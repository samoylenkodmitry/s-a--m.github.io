---
layout: leetcode-entry
title: "1386. Cinema Seat Allocation"
permalink: "/leetcode/problem/2026-08-19-1386-cinema-seat-allocation/"
leetcode_ui: true
entry_slug: "2026-08-19-1386-cinema-seat-allocation"
---

[1386. Cinema Seat Allocation](https://leetcode.com/problems/cinema-seat-allocation/solutions/8469690/kotlin-rust-by-samoylenkodmitry-5vgt/) medium
[substack](https://dmitriisamoilenko.substack.com/p/19082026-1386-cinema-seat-allocation?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/OQO27pDuFoc)

https://dmitrysamoylenko.com/leetcode/

![19.08.2026.webp](/assets/leetcode_daily_images/19.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1456

#### Problem TLDR

Max non-excluded 4-groups in n rows

#### Intuition

Total max groups are 2*n. Exclude groups by comparing bitmasks.

#### Approach

* we can safely use sum() instead of OR

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxNumberOfFamilies(n: Int, rs: Array<IntArray>) = n * 2 -
    rs.groupBy({ it[0] }) { 1 shl it[1] }.values.sumOf { v ->
        2 - (setOf(60, 960, 240).count { v.sum() and it == 0 } + 1) / 2
    }
```
```rust
    pub fn max_number_of_families(n: i32, rs: Vec<Vec<i32>>) -> i32 {
        n * 2 - rs.into_iter().map(|s| (s[0], 1 << s[1])).into_group_map().values().map(|v| {
            let m = v.iter().sum::<i32>();
            2 - (m & 1020 == 0) as i32 - ((m & 60) * (m & 960) * (m & 240) == 0) as i32
        }).sum::<i32>()
    }
```

