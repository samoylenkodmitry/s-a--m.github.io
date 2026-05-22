---
layout: leetcode-entry
title: "33. Search in Rotated Sorted Array"
permalink: "/leetcode/problem/2026-05-22-33-search-in-rotated-sorted-array/"
leetcode_ui: true
entry_slug: "2026-05-22-33-search-in-rotated-sorted-array"
---

[33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/8285804/kotlin-rust-by-samoylenkodmitry-bqxx/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22052026-33-search-in-rotated-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eDQYY0qkkc0)

https://dmitrysamoylenko.com/leetcode/

![22.05.2026.webp](/assets/leetcode_daily_images/22.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1367

#### Problem TLDR

Binary search in shifted array

#### Intuition

* find the split point
* binary search in two parts

#### Approach

* insertion point: the first value is less than first part and bigger than second part
* if target is less than first value - it is in the second part

#### Complexity

- Time complexity:
$$O(logn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun search(n: IntArray, t: Int) = n.asList().run {
        val s = binarySearch { if(it < n[0])1 else -1}.inv()
        maxOf(-1, binarySearch(t, 0, s), binarySearch(t, s))
    }
```
```rust
    pub fn search(n: Vec<i32>, t: i32) -> i32 {
        let (s,b) = (n.partition_point(|&x| x >= n[0]), (t<n[0]) as usize);
        [&n[..s],&n[s..]][b].binary_search(&t).map_or(-1, |i|(i+s*b)as _)
    }
```

