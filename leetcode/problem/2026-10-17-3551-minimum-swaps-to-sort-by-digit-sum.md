---
layout: leetcode-entry
title: "3551. Minimum Swaps to Sort by Digit Sum"
permalink: "/leetcode/problem/2026-10-17-3551-minimum-swaps-to-sort-by-digit-sum/"
leetcode_ui: true
entry_slug: "2026-10-17-3551-minimum-swaps-to-sort-by-digit-sum"
---

[3551. Minimum Swaps to Sort by Digit Sum](https://leetcode.com/problems/minimum-swaps-to-sort-by-digit-sum/solutions/8576231/kotlin-rust-by-samoylenkodmitry-wo5p/) medium
[substack](https://dmitriisamoilenko.substack.com/p/17102026-3551-minimum-swaps-to-sort?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FQjNg1hn-P8)

https://dmitrysamoylenko.com/leetcode/

![17.10.2026.webp](/assets/leetcode_daily_images/17.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1514

#### Problem TLDR

Minimum swaps to sort by digits sum & num

#### Intuition

The cycle sort is possible when you know the final order. That means, count the cycle jumps.

#### Approach

* numbers of swaps is cycle length-1

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minSwaps(n: IntArray): Int {
        val si = n.indices.sortedWith(compareBy({"${n[it]}".sumOf{it-'0'}},{n[it]}))
        val v = HashSet<Int>()
        return n.indices.sumOf{
            var i = it; var c = 0
            while (v.add(i)) {i = si[i]; c++}; max(c-1,0)
        }
    }
```
```rust
    pub fn min_swaps(n: Vec<i32>) -> i32 {
        let mut p: Vec<_> = (0..n.len()).collect(); let mut v = vec![false; n.len()];
        p.sort_by_key(|&i| (n[i].to_string().bytes().map(|b| (b - 48) as i32).sum::<i32>(), n[i]));
        (0..n.len()).map(|mut i| {
            let mut c = 0;
            while !v[i] { v[i] = true; i = p[i]; c += 1 }; 0.max(c - 1)
        }).sum()
    }
```

