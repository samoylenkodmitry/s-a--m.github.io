---
layout: leetcode-entry
title: "1356. Sort Integers by The Number of 1 Bits"
permalink: "/leetcode/problem/2026-02-25-1356-sort-integers-by-the-number-of-1-bits/"
leetcode_ui: true
entry_slug: "2026-02-25-1356-sort-integers-by-the-number-of-1-bits"
---

[1356. Sort Integers by The Number of 1 Bits](https://open.substack.com/pub/dmitriisamoilenko/p/25022026-1356-sort-integers-by-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/25022026-1356-sort-integers-by-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25022026-1356-sort-integers-by-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aQ89OpjetAU)

![9a58f8d1-aaaa-4b92-a674-3a6792b31633 (1).webp](/assets/leetcode_daily_images/5c5fe2e3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1280

#### Problem TLDR

Sort #easy

#### Intuition

Sort.

#### Approach

* you can use 15 buckets (32 bits, and even less for 10^4 max test case)
* still have to sort

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 18ms
    fun sortByBits(a: IntArray) =
    a.sortedBy{it.countOneBits()*1e5+it}
```
```rust
// 0ms
    pub fn sort_by_bits(mut a: Vec<i32>) -> Vec<i32> {
        a.sort_by_key(|&x|(x.count_ones(),x)); a
    }
```

