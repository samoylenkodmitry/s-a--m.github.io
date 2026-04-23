---
layout: leetcode-entry
title: "955. Delete Columns to Make Sorted II"
permalink: "/leetcode/problem/2025-12-21-955-delete-columns-to-make-sorted-ii/"
leetcode_ui: true
entry_slug: "2025-12-21-955-delete-columns-to-make-sorted-ii"
---

[955. Delete Columns to Make Sorted II](https://leetcode.com/problems/delete-columns-to-make-sorted-ii/description) medium
[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted-ii/solutions/7428350/kotlin-rust-by-samoylenkodmitry-zo5m/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21122025-955-delete-columns-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZIaR7OLp83Q)

![189f24be-def7-4a1c-9ebe-7cdf61e00950 (1).webp](/assets/leetcode_daily_images/10d94eaa.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1211

#### Problem TLDR

Remove columns to sort strings #medium

#### Intuition

The brute-force works.
Build the strings column by column and either accept the current column or not.

#### Approach

* optimization: keep only rows that have equal chars

#### Complexity

- Time complexity:
$$O(n^3)$$ or n^2 optimized

- Space complexity:
$$O(n^2)$$ or n oprimized

#### Code

```kotlin
// 32ms
    fun minDeletionSize(s: Array<String>): Int {
        var res = List(s.size) { "" }
        return s[0].indices.count { i ->
            val r = res.zip(s).map { (a,b) -> a + b[i] }
            (1..<s.size).any { r[it-1] > r[it] }.also { if (!it) res = r }
        }
    }
```
```rust
// 0ms
    pub fn min_deletion_size(s: Vec<String>) -> i32 {
        let mut js: Vec<_> = (1..s.len()).collect();
        (0..s[0].len()).filter(|&i| {
            let rm = js.iter().any(|&j| s[j-1].as_bytes()[i] > s[j].as_bytes()[i]);
            if !rm { js.retain(|&j| s[j-1].as_bytes()[i] == s[j].as_bytes()[i]) }; rm
        }).count() as _
    }
```

