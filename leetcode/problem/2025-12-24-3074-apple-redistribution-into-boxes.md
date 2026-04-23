---
layout: leetcode-entry
title: "3074. Apple Redistribution into Boxes"
permalink: "/leetcode/problem/2025-12-24-3074-apple-redistribution-into-boxes/"
leetcode_ui: true
entry_slug: "2025-12-24-3074-apple-redistribution-into-boxes"
---

[3074. Apple Redistribution into Boxes](https://leetcode.com/problems/apple-redistribution-into-boxes/description/) easy
[blog post](https://leetcode.com/problems/apple-redistribution-into-boxes/solutions/7435158/kotlin-rust-by-samoylenkodmitry-muoc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24122025-3074-apple-redistribution?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xaP4i6FooFM)

![b4baba22-df27-4293-96e0-a74d93d0bb8d (1).webp](/assets/leetcode_daily_images/0a8df32d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1214

#### Problem TLDR

Min containers for all apples #easy #counting_sort

#### Intuition

Sum all apples. Take largest containers first.

#### Approach

* (1..51).first for number of containers
* sort by counting

#### Complexity

- Time complexity:
$$O(sort)$$, the sort can be NlogN or N for counting

- Space complexity:
$$O(sort)$$, Kotlin's IntArray.sort is O(1) space complexity

#### Code

```kotlin
// 29ms
    fun minimumBoxes(a: IntArray, c: IntArray) =
        (1..c.size).first { a.sum() <= c.sortedDescending().take(it).sum() }
```
```rust
// 0ms
    pub fn minimum_boxes(a: Vec<i32>, c: Vec<i32>) -> i32 {
        let mut s = a.iter().sum::<i32>(); let mut f = [0;51];
        for c in c { f[c as usize] += 1 }; let mut j = 50;
        (1..51).find(|i| { while f[j] < 1 { j-=1 }; s -= j as i32; f[j] -= 1; s <= 0}).unwrap() as _
    }
```

