---
layout: leetcode-entry
title: "3507. Minimum Pair Removal to Sort Array I"
permalink: "/leetcode/problem/2026-01-22-3507-minimum-pair-removal-to-sort-array-i/"
leetcode_ui: true
entry_slug: "2026-01-22-3507-minimum-pair-removal-to-sort-array-i"
---

[3507. Minimum Pair Removal to Sort Array I](https://leetcode.com/problems/minimum-pair-removal-to-sort-array-i/description/) easy
[blog post](https://leetcode.com/problems/minimum-pair-removal-to-sort-array-i/solutions/7514933/kotlin-rust-by-samoylenkodmitry-fwmb/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22012026-3507-minimum-pair-removal?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xaSCtlhNe6o)

![64566dd5-2f58-4333-8e15-3c6318b38b2f (1).webp](/assets/leetcode_daily_images/1de8a99b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1245

#### Problem TLDR

Sort by removing lowest sum pairs #easy

#### Intuition

Simulate the process.

#### Approach

* just create a new array each time
* or in Rust we can actually remove by index in-place

#### Complexity

- Time complexity:
$$O(n^2)$$ n^2log(n) for golfing in Kotlin

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 76ms
    fun minimumPairRemoval(l: IntArray): Int {
        val l = l.toList()
        if (l == l.sorted()) return 0
        val j = (1..<l.size).minBy { l[it-1] + l[it] }
        return 1 + minimumPairRemoval((l.take(j-1) + (l[j-1]+l[j]) + l.drop(j+1)).toIntArray())
    }
```
```rust
// 0ms
    pub fn minimum_pair_removal(mut l: Vec<i32>) -> i32 {
        (0..).find(|_| {
            let ok = l.windows(2).all(|w| w[0] <= w[1]);
            if let Some(j) = (1..l.len()).min_by_key(|&i| l[i-1] + l[i]) { l[j-1] += l[j];  l.remove(j); }
            ok
        }).unwrap()
    }
```

