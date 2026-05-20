---
layout: leetcode-entry
title: "2657. Find the Prefix Common Array of Two Arrays"
permalink: "/leetcode/problem/2026-05-20-2657-find-the-prefix-common-array-of-two-arrays/"
leetcode_ui: true
entry_slug: "2026-05-20-2657-find-the-prefix-common-array-of-two-arrays"
---

[2657. Find the Prefix Common Array of Two Arrays](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/solutions/8279775/kotlin-rust-by-samoylenkodmitry-ubgu/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20052026-2657-find-the-prefix-common?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zB2ujV53xqs)

https://dmitrysamoylenko.com/leetcode/

![20.05.2026.webp](/assets/leetcode_daily_images/20.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1365

#### Problem TLDR

Duplicates in prefixes

#### Intuition

Only 50 elements, brute-force works.
We can use HashSets or bitmasks

#### Approach

* shortest version in Kotlin is O(n^2)
* in Rust we can use zip+scan, or a simple map

#### Complexity

- Time complexity:
$$O(n^2|n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun findThePrefixCommonArray(a: IntArray, b: IntArray) =
    (1..a.size).map {a.take(it).intersect(b.take(it)).size}
```
```rust
    pub fn find_the_prefix_common_array(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
        let (mut c, mut d) = (0, 0i64);
        (0..a.len()).map(|i|{c|=1<<a[i];d|=1<<b[i];(c&d).count_ones() as _}).collect()
    }
```

