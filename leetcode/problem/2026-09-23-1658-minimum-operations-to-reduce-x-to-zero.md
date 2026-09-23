---
layout: leetcode-entry
title: "1658. Minimum Operations to Reduce X to Zero"
permalink: "/leetcode/problem/2026-09-23-1658-minimum-operations-to-reduce-x-to-zero/"
leetcode_ui: true
entry_slug: "2026-09-23-1658-minimum-operations-to-reduce-x-to-zero"
---

[1658. Minimum Operations to Reduce X to Zero](https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/solutions/8536078/kotlin-rust-by-samoylenkodmitry-wqdt/) medium
[substack](https://dmitriisamoilenko.substack.com/p/23092026-1658-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7u1twJHot1Q)

https://dmitrysamoylenko.com/leetcode/

![23.09.2026.webp](/assets/leetcode_daily_images/23.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1491

#### Problem TLDR

Min operations to remove first or last elements sum of x

#### Intuition

Invert the problem: longest subarray with sum equal to sum()-x

#### Approach

* use the target itself as a sum variable, compare with 0

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun minOperations(n: IntArray, x: Int) = n.run {
        var t = sum() - x; var j = 0
        indices.maxOf { i ->
            t -= n[i]; while (t < 0 && j <= i) t += n[j++]
            if (t == 0) i - j + 1 else -1
        }.let { if (it < 0) -1 else size - it }
    }
```
```rust
    pub fn min_operations(n: Vec<i32>, x: i32) -> i32 {
        let (mut t, mut j, l) = (n.iter().sum::<i32>() - x, 0, n.len());
        (0..l).filter_map(|i| {
            t -= n[i]; while t < 0 && j <= i { t += n[j]; j += 1 }
            (t == 0).then_some((l + j - i - 1) as i32)
        }).min().unwrap_or(-1)
    }
```

