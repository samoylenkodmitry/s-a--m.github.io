---
layout: leetcode-entry
title: "1331. Rank Transform of an Array"
permalink: "/leetcode/problem/2026-07-12-1331-rank-transform-of-an-array/"
leetcode_ui: true
entry_slug: "2026-07-12-1331-rank-transform-of-an-array"
---

[1331. Rank Transform of an Array](https://leetcode.com/problems/rank-transform-of-an-array/solutions/8391969/kotlin-rust-by-samoylenkodmitry-zbz7/) easy
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12072026-1331-rank-transform-of-an?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8yd87OStcow)

https://dmitrysamoylenko.com/leetcode/

![12.07.2026.webp](/assets/leetcode_daily_images/12.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1418

#### Problem TLDR

Rank of items in an array

#### Intuition

Sort. Dedup. Binary search each number.

#### Approach

* we can use toSortedMap().run { ... headSet(it).size } but leetcode give TLE, still O(nlogn)
* Rust: use itertools and collect tuples to the hashmap

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun arrayRankTransform(a: IntArray) =
    a.toSet().sorted().run { a.map { binarySearch(it) + 1 } }
```
```rust
    pub fn array_rank_transform(a: Vec<i32>) -> Vec<i32> {
        let m: HashMap<_,_> = a.iter().copied().sorted().dedup().zip(1..).collect();
        a.iter().map(|x| m[x]).collect()
    }
```

